"""Hybrid LASSQD for the FeFe complex, (6e,5o) per Fe, on IBM Quantum hardware,
followed by an SQD-PDFT (tPBE) energy.

Save an IBM Quantum account first (``QiskitRuntimeService.save_account(...)``).
Each cycle writes a checkpoint, the current LAS wave function (orbitals +
fragment RDMs) as ``casdm.h5``, a :class:`lassqd.LASState` that
``pdft_from_rdms.py`` can reuse, and each fragment's solver state as
``sqd_state{i}.h5``. The files go into a new directory under ``RDMS/``, and the
symlink ``RDMS/latest`` is then switched to it in one step, so ``RDMS/latest``
always holds one whole checkpoint. A later run continues from it when it exists.
"""

import os
import shutil
import tempfile
from functools import partial
from pathlib import Path

import ffsim
from pyscf import cc, gto, lib, scf
from pyscf.mcscf import avas
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_addon_sqd.fermion import solve_sci_batch
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session

from lassqd import (
    LASSCF,
    CycleResult,
    Fragment,
    FragmentSQD,
    LASState,
    SQDState,
    initial_state,
    lassqd_pdft_energy,
    run_lassqd,
)


def prepare_fragment(fragment: Fragment) -> QuantumCircuit:
    """Prepare this example's CCSD-initialized LUCJ state in the sampling basis."""
    norb, nelec = fragment.norb, fragment.nelec
    ccsd = cc.CCSD(fragment.rohf())
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


lib.logger.TIMER_LEVEL = lib.logger.INFO
basis = {"Fe": "6-31g", "C": "6-31g", "H": "6-31g", "O": "6-31g", "N": "6-31g"}
mol = gto.M(atom="fefe.xyz", verbose=4, spin=0, charge=4, basis=basis)
mf = scf.ROHF(mol)
mf.init_guess = "atom"
mf = mf.density_fit()
mf.kernel()
ncas, nelecas, guess_mo_coeff = avas.kernel(mf, ["Fe 3d"], minao=mol.basis)

las = LASSCF(mf, (5, 5), ((4, 2), (2, 4)), spin_sub=(3, 3))
guess_mo_sorted = las.sort_mo(list(range(100, 110)), guess_mo_coeff)
mo_localized = las.localize_init_guess(([0], [1]), guess_mo_sorted)

solvers = [
    FragmentSQD(
        max_iterations=6,
        num_batches=15,
        samples_per_batch=50,
        sci_solver=partial(solve_sci_batch, spin_sq=2.0, max_cycle=200, tol=1e-12),
    )
    for ifrag in range(las.nfrags)
]

# Physical qubits for fragment 0 then fragment 1 (alpha then beta orbitals each).
spin_a_layout = [60, 61, 62, 72, 81, 82, 83, 92, 102, 103]
spin_b_layout = [58, 71, 77, 78, 79, 91, 98, 99, 100, 101]
backend = QiskitRuntimeService().backend("ibm_sherbrooke")
pass_manager = generate_preset_pass_manager(
    backend=backend,
    optimization_level=3,
    initial_layout=spin_a_layout + spin_b_layout,
)
pass_manager.pre_init = ffsim.qiskit.PRE_INIT


checkpoints = Path("RDMS")
latest = checkpoints / "latest"


def save_checkpoint(cycle: int, step: CycleResult) -> None:
    """Write the whole checkpoint, then point ``RDMS/latest`` at it."""
    checkpoints.mkdir(exist_ok=True)
    new = Path(tempfile.mkdtemp(prefix="checkpoint-", dir=checkpoints))
    step.state.save(new / "casdm.h5")
    for ifrag, solver in enumerate(solvers):
        solver.state.save(new / f"sqd_state{ifrag}.h5")
    previous = latest.resolve() if latest.exists() else None
    link = checkpoints / "latest.new"
    link.unlink(missing_ok=True)
    link.symlink_to(new.name)
    # Renaming the link replaces RDMS/latest atomically: a crash before this
    # leaves the previous checkpoint in place.
    os.replace(link, latest)
    if previous is not None:
        shutil.rmtree(previous)


# Continue from a previous run's checkpoint if there is one.
if latest.exists():
    state = LASState.load(latest / "casdm.h5")
    for ifrag, solver in enumerate(solvers):
        solver.state = SQDState.load(latest / f"sqd_state{ifrag}.h5")
else:
    state = initial_state(las, mo_localized)
with Session(backend=backend) as session:
    sampler = SamplerV2(mode=session)
    sampler.options.max_execution_time = 10800
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = True
    result = run_lassqd(
        las,
        state,
        solvers,
        sampler,
        circuit_builder=prepare_fragment,
        pass_manager=pass_manager,
        shots=30000,
        max_cycles=1,
        callback=save_checkpoint,
    )
e_pdft, _, _ = lassqd_pdft_energy(las, result.state, ot="tPBE")
print("SQDPDFT energy", e_pdft)
