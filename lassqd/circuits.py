"""Fragment circuits, and running them all in one job.

Each fragment gets its own state-preparation circuit. The fragment circuits are
glued side by side onto one register, with one classical register per fragment,
so a single job samples every fragment; the counts are then cut back apart.
"""

import numpy as np
from pyscf import cc
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from lassqd.basis import fragment_mo_basis, fragment_rohf


def lucj_circuit(h1, h2, norb, nelec, *, n_reps=1):
    """Spin-unbalanced LUCJ circuit (Hartree-Fock + UCJ, no measurements) for one
    fragment Hamiltonian ``(h1, h2)`` in the LAS basis.

    The UCJ operator starts from CCSD amplitudes on the fragment's ROHF reference
    and is then optimized with ffsim's linear method. Qubit ordering is ffsim's
    Jordan-Wigner convention: alpha orbitals on qubits ``0..norb-1``, beta on
    ``norb..2*norb-1``.
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


def glue_circuits(circuits):
    """Place ``circuits`` side by side on one register and measure each one into
    its own classical register, in order."""
    widths = [qc.num_qubits for qc in circuits]
    glued = QuantumCircuit(sum(widths))
    cregs = [ClassicalRegister(n) for n in widths]
    for creg in cregs:
        glued.add_register(creg)
    start = 0
    for qc, creg, n in zip(circuits, cregs, widths):
        glued.append(qc, range(start, start + n))
        for i in range(n):
            glued.measure(start + i, creg[i])
        start += n
    return glued


def cut_counts(counts):
    """Split the counts of a glued circuit into one counts dict per fragment.

    Qiskit joins the classical registers with spaces, last register first, so the
    split keys are reversed to put fragment 0 first.
    """
    per_fragment = None
    for key, count in counts.items():
        parts = key.split()[::-1]
        if per_fragment is None:
            per_fragment = [{} for _ in parts]
        for frag_counts, part in zip(per_fragment, parts):
            frag_counts[part] = frag_counts.get(part, 0) + count
    return per_fragment
