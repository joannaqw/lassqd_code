"""
Functions for using pyscf to estimate the ground state of a molecule from noisy configuration samples.

.. currentmodule:: sqsd.qsci.pyscf_solver

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   solve_pyscf
"""

from __future__ import annotations

import numpy as np
from pyscf import fci


def solve_pyscf2(
    addresses: tuple[list[int], list[int]],
    hcore: np.ndarray,
    eri: np.ndarray,
    num_up: int,
    num_dn: int,
    spin_sq: int | None = None,
    max_davidson: int = 100,
    verbose: int | None = None,
    tol: float | None = None,
    nroots: int | None = None,
    shift: float |None=None
) -> tuple[float, np.ndarray, list[np.ndarray], float, np.ndarray,np.ndarray]:
    """
    Approximate the ground state given molecular integrals and Slater determinant addresses.

    Args:
        addresses: A length-2 tuple of lists containing sorted, base-10 representations
            of bitstrings, which ``pyscf`` refers to as determinant addresses.
            The first list represents configurations of the alpha particles,
            and the second list represents that of the beta particles. Note,
            these addresses must be sorted in order to obtain meaningful results.
        hcore: Core Hamiltonian matrix representing single-electron integrals
        eri: Electronic repulsion integrals representing two-electron integrals
        num_up: number of spin-up electrons
        num_dn: number of spin-down electrons
        spin_sq: Target value for the total spin squared for the ground state.
            If ``None``, no spin will be imposed.
        max_davidson: The maximum number of cycles of Davidson's algorithm
        verbose: A verbosity level between 0 and 10 to pass to PySCF

        Returns:
            A tuple containing:
                - Minimum energy from SCI calculation
                - SCI coefficients
                - Average orbital occupancy
                - Expectation value of spin-squared
    """
    # Number of molecular orbitals
    norb = hcore.shape[0]
    #print('norb',flush=True)
    # Call the projection + eigenstate finder
    myci = fci.selected_ci.SelectedCI()
    #print('myci',flush=True)
    if spin_sq is not None:
        myci = fci.addons.fix_spin_(myci,shift=shift,ss=spin_sq)
    #print('fix spin',flush = True)
    # fci.selected_ci.kernel_fixed_space( 
    e_sci, coeffs_sci = fci.selected_ci.kernel_fixed_space(      
        myci,
        hcore,
        eri,
        norb,
        (num_up, num_dn),
        ci_strs=addresses,
        verbose=verbose,
        max_cycle=max_davidson,
        tol=tol,
        nroots=nroots
    )
    # print('fci.selected_ci.kernel',flush=True)
    # Calculate the avg occupancy of each orbital
    dm1 = myci.make_rdm1s(coeffs_sci[0], norb, (num_up, num_dn))
    dm11 = myci.make_rdm1(coeffs_sci[0], norb, (num_up, num_dn))
    #print('dm1',flush=True)
    avg_occupancy = [np.diagonal(dm1[0]), np.diagonal(dm1[1])]
    #print('after avg_occupancy',flush=True)
    dm2 = myci.make_rdm12(coeffs_sci[0], norb,(num_up,num_dn))[1]
    dm22 = myci.make_rdm2(coeffs_sci[0], norb, (num_up, num_dn))
    #print('dm2',flush = True)
    e_sci = np.einsum("pr,pr->", dm11, hcore) + 0.5 * np.einsum("prqs,prqs->", dm22, eri)
    # Compute total spin
    spin_squared = myci.spin_square(coeffs_sci[0], norb, (num_up, num_dn))[0]

    return e_sci, coeffs_sci[0], avg_occupancy, spin_squared, dm1, dm2
