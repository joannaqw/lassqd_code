"""Hybrid LASSQD for the FeFe complex, (5e,5o) per Fe, on IBM Quantum hardware,
followed by an SQD-PDFT (tPBE) energy.

Save an IBM Quantum account first (``QiskitRuntimeService.save_account(...)``).
Each cycle writes the current LAS wave function (orbitals + fragment RDMs) to
``RDMS/casdm.h5``, which ``pdft_from_rdms.py`` can reuse.
"""

from functools import partial
from pathlib import Path

import ffsim
import numpy as np
from pyscf import gto, lib, scf
from pyscf.mcscf import avas
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_addon_sqd.fermion import solve_sci_batch
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session

from lassqd import (
    FragmentSQD,
    LASSCFNoSymm,
    lassqd_pdft_energy,
    run_lassqd,
    save_rdms,
)

lib.logger.TIMER_LEVEL = lib.logger.INFO
basis = {"Fe": "6-31g", "C": "6-31g", "H": "6-31g", "O": "6-31g", "N": "6-31g"}
mol = gto.M(atom="fefe.xyz", verbose=4, spin=0, charge=4, basis=basis)
mf = scf.ROHF(mol)
mf.init_guess = "atom"
mf = mf.density_fit()
mf.kernel()
ncas, nelecas, guess_mo_coeff = avas.kernel(mf, ["Fe 3d"], minao=mol.basis)

las = LASSCFNoSymm(mf, (5, 5), ((4, 2), (2, 4)), spin_sub=(3, 3))
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


def save_wave_function(cycle, las):
    np.save("current_orb", las.mo_coeff)
    Path("RDMS").mkdir(exist_ok=True)
    save_rdms("RDMS/casdm.h5", las.casdm1frs, las.casdm2fr, mo_coeff=las.mo_coeff)


# Restart from the orbitals of a previous run; use mo_localized to start fresh.
mo_init = np.load("current_orb.npy")
with Session(backend=backend) as session:
    sampler = SamplerV2(mode=session)
    sampler.options.max_execution_time = 10800
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = True
    result = run_lassqd(
        las,
        mo_init,
        solvers,
        sampler,
        pass_manager=pass_manager,
        shots=30000,
        max_cycles=1,
        callback=save_wave_function,
    )
e_pdft = lassqd_pdft_energy(
    las, result.casdm1frs, result.casdm2fr, result.mo_coeff, ot="tPBE"
)
print("SQDPDFT energy", e_pdft)
