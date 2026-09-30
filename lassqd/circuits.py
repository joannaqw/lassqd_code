"""Fragment state-preparation circuits."""

from collections.abc import Callable

import ffsim
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
    with no measurements. The UCJ operator is initialized by compressed double
    factorization of CCSD amplitudes on the fragment's ROHF reference.
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
        A ``2 * norb``-qubit circuit preparing the CCSD-initialized LUCJ state.
    """

    # qiskit rejects numpy integers such as those in las.ncas_sub
    norb = int(norb)
    nelec = tuple(int(n) for n in nelec)
    _, h1_mo, h2_mo = fragment_mo_basis(h1, h2, norb, nelec)
    mf_mo = fragment_rohf(h1_mo, h2_mo, norb, nelec)
    ccsd = cc.CCSD(mf_mo)
    ccsd.verbose = 0
    ccsd.kernel()

    interaction_pairs = (
        [(p, p + 1) for p in range(norb - 1)],  # alpha-alpha
        [(p, p) for p in range(0, norb, 4)],  # alpha-beta
        [(p, p + 1) for p in range(norb - 1)],  # beta-beta
    )

    ucj_op = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        ccsd.t2,
        t1=ccsd.t1,
        n_reps=n_reps,
        interaction_pairs=interaction_pairs,
        optimize=True,
        options={"maxiter": 100},
    )

    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op), qubits)
    return circuit
