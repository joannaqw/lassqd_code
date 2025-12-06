"""
Functions for optimizing the molecular orbitals.

.. currentmodule:: sqsd.qsci.orbital_optimization

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   optimize_orbitals
   rotate_integrals
"""

from __future__ import annotations

import numpy as np
from jax import grad, jit
from jax import numpy as jnp
from jax._src.typing import Array
from jax.scipy.linalg import expm
from pyscf import fci
from scipy import linalg as LA


def optimize_orbitals(
    addresses: tuple[list[int], list[int]],
    hcore: np.ndarray,
    eri: np.ndarray,
    p_flat: np.ndarray,
    num_up: int,
    num_dn: int,
    spin_sq: int,
    num_iters: int,
    max_cycles_qsci: int,
    num_steps_grad: int,
    learning_rate: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Optimize orbitals.

    Optimize orbitals to produce a minimal ground state. The process involves
    iterating over 3 steps:

        For ``num_iters`` iterations:
            - Rotate the integrals with respect to the parameters, ``p_flat``
            - Run QSCI to estimate groundstate energy and wavefunction amplitudes
            - Optimize ``p_flat`` using gradient descent and the wavefunction
              amplitudes found in Step 2

    Refer to `Sec. II A 4 <https://arxiv.org/pdf/2405.05068>`_ for more detailed
    discussion on this orbital optimization technique.

    Args:
        addresses: A length-2 tuple of lists containing base-10 representations
            of bitstrings, which ``pyscf`` refers to as determinant addresses.
            The first list represents configurations of the alpha particles,
            and the second list represents that of the beta particles.
        hcore: Core Hamiltonian matrix representing single-electron integrals
        eri: Electronic repulsion integrals representing two-electron integrals
        p_flat: 1D array defining the orbital transform. This array will be reshaped
            to be of shape (# orbitals, # orbitals) before being used as a
            similarity transform on the orbitals.
        num_up: number of spin-up electrons
        num_dn: number of spin-down electrons
        spin_sq: Target value for the total spin squared for the ground state
        num_iters: The number of iterations of orbital optimization to perform
        max_cycles_qsci: The maximum number of cycles of Davidson's algorithm to
            perform at each iteration during QSCI.
        num_steps_grad: The number of steps of gradient descent to perform
            during each optimization iteration
        learning_rate: The learning rate to use during gradient descent

    Returns:
        A tuple containing:
            - The groundstate energy found during the last optimization iteration
            - An optimized 1D array defining the orbital transform
            - A 1D array representing the one-body reduced density matrix generated
              during the last optimization iteration
    """
    # TODO: Need metadata showing the optimization history
    ## hcore and eri in physicist ordering
    num_orbitals = hcore.shape[0]
    p_flat = p_flat.copy()
    eri_phys = np.asarray(eri.transpose(0, 2, 3, 1), order="C")  # physicist ordering
    for _ in range(num_iters):
        # Rotate integrals
        hcore_rot, eri_rot = rotate_integrals(hcore, eri_phys, num_orbitals, p_flat)
        eri_rot_chem = np.asarray(eri_rot.transpose(0, 3, 1, 2), order="C")  # chemist ordering

        # Solve for ground state with respect to optimized integrals
        myci = fci.selected_ci.SelectedCI()
        myci = fci.addons.fix_spin_(myci, ss=spin_sq)
        e_qsci, amplitudes = fci.selected_ci.kernel_fixed_space(
            myci,
            hcore_rot,
            eri_rot_chem,
            num_orbitals,
            (num_up, num_dn),
            ci_strs=addresses,
            max_cycle=max_cycles_qsci,
        )

        # Generate the one and two-body reduced density matrices from latest wavefunction amplitudes
        dm1, dm2_chem = myci.make_rdm12(amplitudes, num_orbitals, (num_up, num_up))
        dm2 = np.asarray(dm2_chem.transpose(0, 2, 3, 1), order="C")

        # TODO: Expose the momentum parameter as an input option
        # Optimize the basis rotations
        _optimize_orbitals_sci(
            p_flat, learning_rate, 0.9, num_steps_grad, dm1, dm2, hcore, eri_phys
        )

    return e_qsci, p_flat, np.diagonal(dm1)


def rotate_integrals(
    hcore: np.ndarray, eri: np.ndarray, num_orbitals: int, p_flat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform a similarity transform on the integrals.

    This transform is described in `Sec. II A 4 <https://arxiv.org/pdf/2405.05068>`_.
    """
    p = np.reshape(p_flat, (num_orbitals, num_orbitals))
    K = (p - np.transpose(p)) / 2.0
    U = LA.expm(K)
    hcore_rot = np.matmul(np.transpose(U), np.matmul(hcore, U))
    eri_rot = np.einsum("pqrs, pi, qj, rk, sl->ijkl", eri, U, U, U, U, optimize=True)

    return np.array(hcore_rot), np.array(eri_rot)


def _optimize_orbitals_sci(
    p_flat: np.ndarray,
    learning_rate: float,
    momentum: float,
    num_steps: int,
    dm1: np.ndarray,
    dm2: np.ndarray,
    hcore: np.ndarray,
    eri: np.ndarray,
) -> None:
    """
    Optimize orbital rotation parameters in-place using gradient descent.

    This procedure is described in `Sec. II A 4 <https://arxiv.org/pdf/2405.05068>`_.
    """
    prev_update = np.zeros(len(p_flat))
    num_orbitals = dm1.shape[0]
    for _ in range(num_steps):
        grad = _SCISCF_Energy_contract_grad(dm1, dm2, hcore, eri, num_orbitals, p_flat)
        prev_update = learning_rate * grad + momentum * prev_update
        p_flat -= prev_update


def _SCISCF_Energy_contract(
    dm1: np.ndarray,
    dm2: np.ndarray,
    hcore: np.ndarray,
    eri: np.ndarray,
    num_orbitals: int,
    p_flat: np.ndarray,
) -> Array:
    """
    Calculate gradient.

    The gradient can be calculated by contracting the bare one and two-body
    reduced density matrices with the gradients of the of the one and two-body
    integrals with respect to the rotation parameters, ``p_flat``.
    """
    p = jnp.reshape(p_flat, (num_orbitals, num_orbitals))
    K = (p - jnp.transpose(p)) / 2.0
    U = expm(K)
    hcore_rot = jnp.matmul(jnp.transpose(U), jnp.matmul(hcore, U))
    eri_rot = jnp.einsum("pqrs, pi, qj, rk, sl->ijkl", eri, U, U, U, U)
    grad = jnp.sum(dm1 * hcore_rot) + jnp.sum(dm2 * eri_rot / 2.0)

    return grad


_SCISCF_Energy_contract_grad = jit(grad(_SCISCF_Energy_contract, argnums=5), static_argnums=4)
