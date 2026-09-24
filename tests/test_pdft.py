import numpy as np
import pytest
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

from lassqd import lassqd_pdft_energy, load_rdms, save_rdms
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
    ref = get_mcpdft_child_class(las, ot="tPBE").compute_pdft_energy_()[0]
    assert np.isclose(lassqd_pdft_energy(las, casdm1frs, casdm2fr, las.mo_coeff), ref)


def test_save_load_round_trip(tmp_path, ci_las):
    las, casdm1frs, casdm2fr = ci_las
    save_rdms(tmp_path / "rdms.h5", casdm1frs, casdm2fr, mo_coeff=las.mo_coeff)
    dm1, dm2, mo = load_rdms(tmp_path / "rdms.h5")
    assert all(np.array_equal(a, b) for a, b in zip(dm1, casdm1frs))
    assert all(np.array_equal(a, b) for a, b in zip(dm2, casdm2fr))
    assert np.array_equal(mo, las.mo_coeff)

    # Unbatched RDMs are stored with a leading root axis; orbitals are optional.
    save_rdms(tmp_path / "bare.h5", [d[0] for d in casdm1frs], [d[0] for d in casdm2fr])
    dm1, _, mo = load_rdms(tmp_path / "bare.h5")
    assert dm1[0].shape == casdm1frs[0].shape
    assert mo is None
