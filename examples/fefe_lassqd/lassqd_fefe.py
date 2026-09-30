"""Hybrid LASSQD for the FeFe complex, (6e,10o) per Fe, with LUCJ circuits sampled
classically with ffsim and determinant carryover between cycles.

Needs ``fefe_as.npy`` and ``as_increase_avas.npy`` (initial orbitals) in the
working directory; they are not tracked in the repository.
"""

import ffsim
import numpy as np
from pyscf import fci, gto, lib, scf
from qiskit_addon_sqd.fermion import SCIResult, SCIState

from lassqd import FragmentSQD, LASSCFNoSymm, run_lassqd


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

    This preserves the FeFe example's solver settings. The spin penalty targets
    ``S(S + 1)`` with ``S = abs(neleca - nelecb) / 2``. The addon's RDM processing
    assumes one eigenvector, so multiple roots are handled directly with PySCF.
    Energies are recomputed without the spin penalty or constant energy term.
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
        max_iterations=6,
        num_batches=15,
        samples_per_batch=170,
        sci_solver=solve_sci_batch,
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
    ffsim.qiskit.FfsimSampler(),
    glue_circuits=False,
    shots=100_000,
    max_cycles=50,
    conv_tol=1e-5,
    callback=save_orbitals,
)
print(
    f"LASSQD energy {result.e_tot:.10f} ({'' if result.converged else 'not '}converged)"
)
