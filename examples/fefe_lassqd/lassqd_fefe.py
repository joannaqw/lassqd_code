"""Hybrid LASSQD for the FeFe complex, (6e,10o) per Fe, with LUCJ circuits sampled
classically with ffsim and determinant carryover between cycles.

"""

from collections.abc import Sequence
from functools import partial
from pathlib import Path

import ffsim
import numpy as np
from pyscf import cc, fci, gto, lib, scf
from pyscf.mcscf import avas
from qiskit import QuantumCircuit
from qiskit_addon_sqd.fermion import SCIResult, SCIState

from lassqd import FragmentSQD, LASSCFNoSymm, run_lassqd
from lassqd.basis import fragment_mo_basis, fragment_rohf


def prepare_fragment(
    h1: np.ndarray, h2: np.ndarray, norb: int, nelec: tuple[int, int]
) -> QuantumCircuit:
    """Prepare this example's CCSD-initialized LUCJ state in the SQD ROHF basis."""
    norb = int(norb)
    nelec = tuple(int(n) for n in nelec)
    _, h1_mo, h2_mo = fragment_mo_basis(h1, h2, norb, nelec)
    mf_mo = fragment_rohf(h1_mo, h2_mo, norb, nelec)
    ccsd = cc.CCSD(mf_mo)
    ccsd.verbose = 0
    ccsd.kernel()

    # Ansatz choices for this calculation; change these to explore other circuits.
    interaction_pairs = (
        [(p, p + 1) for p in range(norb - 1)],  # alpha-alpha
        [(p, p) for p in range(0, norb, 4)],  # alpha-beta
        [(p, p + 1) for p in range(norb - 1)],  # beta-beta
    )
    ucj_op = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        ccsd.t2,
        t1=ccsd.t1,
        n_reps=1,
        interaction_pairs=interaction_pairs,
        optimize=True,
    )
    circuit = QuantumCircuit(2 * norb)
    # Alpha orbitals precede beta orbitals, as required by FragmentSQD.
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), circuit.qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op), circuit.qubits)
    return circuit


def solve_sci_batch(
    ci_strings: list[tuple[np.ndarray, np.ndarray]],
    h1: np.ndarray,
    h2: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    nroots: int = 10,
    max_cycle: int = 200,
    tol: float = 1e-16,
) -> list[SCIResult]:
    """Solve several Davidson roots per batch and return only the first root.

    The spin penalty targets
    ``S(S + 1)`` with ``S = abs(neleca - nelecb) / 2``. The addon's RDM processing
    assumes one eigenvector, so multiple roots are handled directly with PySCF.
    """
    spin = abs(nelec[0] - nelec[1]) / 2
    results = []
    for strings in ci_strings:
        myci = fci.addons.fix_spin_(fci.selected_ci.SelectedCI(), ss=spin * (spin + 1))
        _, sci_vecs = fci.selected_ci.kernel_fixed_space(
            myci,
            h1,
            h2,
            norb,
            nelec,
            ci_strs=strings,
            nroots=nroots,
            max_cycle=max_cycle,
            tol=tol,
        )
        sci_vec = sci_vecs if nroots == 1 else sci_vecs[0]
        dm1s = myci.make_rdm1s(sci_vec, norb, nelec)
        dm1 = myci.make_rdm1(sci_vec, norb, nelec)
        dm2 = myci.make_rdm2(sci_vec, norb, nelec)
        energy = np.einsum("pr,pr->", dm1, h1) + 0.5 * np.einsum("prqs,prqs->", dm2, h2)
        state = SCIState(
            amplitudes=np.array(sci_vec),
            ci_strs_a=sci_vec._strs[0],
            ci_strs_b=sci_vec._strs[1],
            norb=norb,
            nelec=nelec,
        )
        results.append(
            SCIResult(
                energy,
                state,
                orbital_occupancies=(np.diagonal(dm1s[0]), np.diagonal(dm1s[1])),
                rdm1=dm1,
                rdm2=dm2,
            )
        )
    return results


def record_iteration(history: dict[str, list], results: list[SCIResult]) -> None:
    """Keep small summaries of each batch, without retaining its wave function."""
    best = min(results, key=lambda result: result.energy)
    rows = {
        "e_hist": [r.energy for r in results],
        "s_hist": [r.sci_state.spin_square() for r in results],
        "d_hist": [r.sci_state.amplitudes.size for r in results],
        "a_hist": [len(r.sci_state.ci_strs_a) for r in results],
        "b_hist": [len(r.sci_state.ci_strs_b) for r in results],
        "occupancy_hist": np.concatenate(best.orbital_occupancies),
    }
    for name, row in rows.items():
        history.setdefault(name, []).append(row)


def save_cycle(
    cycle: int,
    las: LASSCFNoSymm,
    *,
    solvers: Sequence[FragmentSQD],
    histories: Sequence[dict[str, list]],
    output_dir: str | Path = ".",
) -> None:
    """Overwrite checkpoints with this cycle's data, then clear history buffers.

    Called by ``run_lassqd`` after the fragment solves and orbital update. Fragment
    states retain their SQD ROHF basis; ``current_orb`` holds the updated LAS basis.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "current_orb", las.mo_coeff)
    for ifrag, (solver, history) in enumerate(zip(solvers, histories)):
        directory = output_dir / f"data_carryover_frag{ifrag}"
        directory.mkdir(exist_ok=True)
        solver.sci_state.save(directory / "sci_vec")
        for name, rows in history.items():
            np.save(directory / name, np.asarray(rows))
        if solver.carryover_strings is not None:
            alpha, beta = solver.carryover_strings
            np.save(directory / "alpha_strings", alpha)
            np.save(directory / "beta_strings", beta)
    for history in histories:
        history.clear()


lib.logger.TIMER_LEVEL = lib.logger.INFO
basis = {"Fe": "6-31g", "C": "6-31g", "H": "6-31g", "O": "6-31g", "N": "6-31g"}
mol = gto.M(atom="fefe.xyz", verbose=4, spin=0, charge=4, basis=basis)
mf = scf.ROHF(mol)
mf.init_guess = "atom"
mf = mf.density_fit()
mf.kernel()
ncas, nelecas, guess_mo_coeff = avas.kernel(mf, ["Fe 3d"], minao=mol.basis)

las = LASSCFNoSymm(mf, (10, 10), ((4, 2), (2, 4)), spin_sub=(3, 3))
guess_mo_sorted = las.sort_mo(list(range(100, 120)), guess_mo_coeff)
mo_localized = las.localize_init_guess(([0], [1]), guess_mo_sorted)

histories = [{} for _ in range(las.nfrags)]
solvers = [
    FragmentSQD(
        max_iterations=6,
        num_batches=15,
        samples_per_batch=170,
        sci_solver=solve_sci_batch,
        carryover_threshold=1e-3,
        callback=partial(record_iteration, histories[ifrag]),
    )
    for ifrag in range(las.nfrags)
]


result = run_lassqd(
    las,
    mo_localized,
    solvers,
    ffsim.qiskit.FfsimSampler(),
    circuit_builder=prepare_fragment,
    glue_circuits=False,
    shots=100_000,
    max_cycles=50,
    conv_tol=1e-5,
    callback=partial(save_cycle, solvers=solvers, histories=histories),
)
print(
    f"LASSQD energy {result.e_tot:.10f} ({'' if result.converged else 'not '}converged)"
)
