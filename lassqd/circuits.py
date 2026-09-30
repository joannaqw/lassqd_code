"""Fragment state-preparation circuits."""

from collections.abc import Callable

import numpy as np
from pyscf import cc
from qiskit import QuantumCircuit, QuantumRegister

from lassqd.basis import fragment_mo_basis, fragment_rohf

# ``circuit_fn(h1, h2, norb, nelec) -> circuit``, e.g. :func:`lucj_circuit`.
CircuitFn = Callable[[np.ndarray, np.ndarray, int, tuple[int, int]], QuantumCircuit]


def lucj_circuit(
    h1: np.ndarray,
    h2: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    n_reps: int = 1,
) -> QuantumCircuit:
    """Build a spin-unbalanced LUCJ circuit for one fragment Hamiltonian.

    The circuit is Hartree-Fock state preparation followed by the UCJ operator,
    with no measurements. The UCJ operator starts from CCSD amplitudes on the
    fragment's ROHF reference and is then optimized with ffsim's linear method.
    Qubit ordering is ffsim's Jordan-Wigner convention: alpha orbitals on qubits
    ``0..norb-1``, beta on ``norb..2*norb-1``.

    Args:
        h1: One-electron integrals in the LAS basis, shape ``(norb, norb)``.
        h2: Two-electron integrals in the LAS basis, in chemists' notation, shape
            ``(norb,) * 4``.
        norb: Number of fragment orbitals. Numpy integers are accepted.
        nelec: ``(neleca, nelecb)``. Numpy integers are accepted.
        n_reps: Number of UCJ layers.

    Returns:
        A ``2 * norb``-qubit circuit preparing the optimized LUCJ state.
    """
    import ffsim
    from ffsim.optimize import minimize_linear_method

    # qiskit rejects numpy integers such as those in las.ncas_sub
    norb = int(norb)
    nelec = tuple(int(n) for n in nelec)
    _, h1_mo, h2_mo = fragment_mo_basis(h1, h2, norb, nelec)
    mf_mo = fragment_rohf(h1_mo, h2_mo, norb, nelec)
    ccsd = cc.CCSD(mf_mo)
    ccsd.verbose = 0
    ccsd.kernel()

    hamiltonian = ffsim.linear_operator(
        ffsim.MolecularHamiltonian(h1_mo, h2_mo, mf_mo.mol.energy_nuc()),
        norb=norb,
        nelec=nelec,
    )
    reference_state = ffsim.hartree_fock_state(norb, nelec)
    interaction_pairs = (
        [(p, p + 1) for p in range(norb - 1)],  # alpha-alpha
        [(p, p) for p in range(0, norb, 4)],  # alpha-beta
        [(p, p + 1) for p in range(norb - 1)],  # beta-beta
    )

    def operator(x):
        return ffsim.UCJOpSpinUnbalanced.from_parameters(
            x,
            norb=norb,
            n_reps=n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=True,
        )

    def params_to_vec(x):
        return ffsim.apply_unitary(reference_state, operator(x), norb=norb, nelec=nelec)

    x0 = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        ccsd.t2, n_reps=n_reps, t1=ccsd.t1
    ).to_parameters(interaction_pairs=interaction_pairs)
    result = minimize_linear_method(params_to_vec, hamiltonian, x0=x0)

    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinUnbalancedJW(operator(result.x)), qubits)
    return circuit
