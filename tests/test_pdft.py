import numpy as np
import pytest
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

from lassqd import LASState, lassqd_pdft_energy
from lassqd.pdft import make_casdm1s, make_casdm2


@pytest.fixture
def ci_las(h6):
    """Conventional (CI-vector) LASSCF, whose full active-space RDMs mrh builds itself."""
    mf, _, mo = h6
    las = LASSCF(mf, (3, 3), ((2, 1), (1, 2)), spin_sub=(2, 2))
    las.kernel(mo)
    casdm1frs = [dm[None] for dm in las.make_casdm1s_sub()]
    casdm2fr = [dm[None] for dm in las.make_casdm2_sub()]
    return las, casdm1frs, casdm2fr


def test_rdm_assembly_matches_mrh(ci_las):
    las, casdm1frs, casdm2fr = ci_las
    assert np.allclose(make_casdm1s(casdm1frs), las.make_casdm1s())
    assert np.allclose(make_casdm2(casdm1frs, casdm2fr), las.make_casdm2())


def test_pdft_energy_matches_las_pdft(ci_las):
    from mrh.my_pyscf.mcpdft.laspdft import get_mcpdft_child_class

    las, casdm1frs, casdm2fr = ci_las
    e_tot_ref, e_ot_ref, e_states_ref = get_mcpdft_child_class(
        las, ot="tPBE"
    ).compute_pdft_energy_()
    state = LASState(las.mo_coeff, casdm1frs, casdm2fr)
    e_tot, e_ot, e_states = lassqd_pdft_energy(las, state)
    assert np.isclose(e_tot, e_tot_ref)
    assert np.isclose(e_ot, e_ot_ref)
    assert np.allclose(e_states, e_states_ref)
