import numpy as np
from conftest import empty_circuit, full_space_sampler

from lassqd import FragmentSQD, LASSCFNoSymm, run_lassqd


def run(mf, las, mo, **kwargs):
    solvers = [FragmentSQD(2, 2, 100, seed=0) for _ in range(las.nfrags)]
    return run_lassqd(
        las,
        mo,
        solvers,
        full_space_sampler(las),
        circuit_fn=empty_circuit,
        conv_tol=1e-10,
        max_cycles=100,
        **kwargs,
    )


def test_full_space_lassqd_matches_lasscf(h4):
    mf, las, mo = h4
    ref = LASSCFNoSymm(mf, (2, 2), ((1, 1), (1, 1)), spin_sub=(1, 1))
    ref.kernel(mo)
    result = run(mf, las, mo)
    assert result.converged
    assert abs(result.e_tot - ref.e_tot) < 1e-7


def test_result_is_a_consistent_wave_function(h6):
    mf, las, mo = h6
    cycles = []
    result = run(mf, las, mo, callback=lambda cycle, las: cycles.append(las.e_tot))
    assert result.converged
    assert cycles == result.e_hist
    # The RDMs are expressed in result.mo_coeff's active orbitals.
    e = las.energy_nuc() + las.energy_elec(
        mo_coeff=result.mo_coeff, casdm1frs=result.casdm1frs, casdm2fr=result.casdm2fr
    )
    assert np.isclose(e, result.e_tot)
    # run_lassqd restores the settings it overrides.
    assert las.max_cycle_macro != 1
