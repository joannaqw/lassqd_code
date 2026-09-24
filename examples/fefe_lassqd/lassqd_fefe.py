"""Hybrid LASSQD for the FeFe complex, (10e,10o) per Fe, with LUCJ circuits sampled
classically on Aer and determinant carryover between cycles.

Needs ``fefe_as.npy`` and ``as_increase_avas.npy`` (initial orbitals) in the
working directory; they are not tracked in the repository.
"""

import numpy as np
from pyscf import gto, lib, scf

from lassqd import FragmentSQD, LASSCFNoSymm, aer_sampler, run_lassqd

lib.logger.TIMER_LEVEL = lib.logger.INFO
basis = {"Fe": "6-31g", "C": "6-31g", "H": "6-31g", "O": "6-31g", "N": "6-31g"}
mol = gto.M(atom="fefe.xyz", verbose=4, spin=0, charge=4, basis=basis)
mf = scf.ROHF(mol)
mf.init_guess = "atom"
mf = mf.density_fit()
mf.mo_coeff = np.load("fefe_as.npy")
mf.kernel()
guess_mo_coeff = np.load("as_increase_avas.npy")

las = LASSCFNoSymm(mf, (10, 10), ((4, 2), (2, 4)), spin_sub=(3, 3))
guess_mo_sorted = las.sort_mo(list(range(100, 120)), guess_mo_coeff)
mo_localized = las.localize_init_guess(([0], [1]), guess_mo_sorted)

solvers = [
    FragmentSQD(
        iterations=6,
        n_batches=15,
        samples_per_batch=170,
        max_davidson_cycles=200,
        tol=1e-16,
        nroots=10,
        carryover_threshold=1e-3,
        output_dir=f"data_carryover_frag{ifrag}",
    )
    for ifrag in range(las.nfrags)
]


def save_orbitals(cycle, las):
    np.save("current_orb", las.mo_coeff)


result = run_lassqd(
    las,
    mo_localized,
    solvers,
    aer_sampler(shots=100_000),
    max_cycles=50,
    conv_tol=1e-5,
    callback=save_orbitals,
)
print(f"LASSQD energy {result.e_tot:.10f} ({'' if result.converged else 'not '}converged)")
