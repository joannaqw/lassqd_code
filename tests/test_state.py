import h5py
import numpy as np
import pytest

from lassqd import LASState, fragment_hamiltonians, initial_state
from lassqd.las import _capture_hamiltonians


@pytest.fixture
def state(h6):
    _, las, mo = h6
    return initial_state(las, mo)


def test_save_load_round_trip(tmp_path, state):
    state.save(tmp_path / "state.h5")
    loaded = LASState.load(tmp_path / "state.h5")
    assert np.array_equal(loaded.mo_coeff, state.mo_coeff)
    assert loaded.nfrags == state.nfrags == 2
    for a, b in zip(
        loaded.casdm1frs + loaded.casdm2fr, state.casdm1frs + state.casdm2fr
    ):
        assert np.array_equal(a, b)


def test_rdms_get_a_root_axis_and_are_read_only(state):
    bare = LASState(
        state.mo_coeff,
        [dm[0] for dm in state.casdm1frs],
        [dm[0] for dm in state.casdm2fr],
    )
    assert [dm.shape for dm in bare.casdm1frs] == [dm.shape for dm in state.casdm1frs]
    assert [dm.shape for dm in bare.casdm2fr] == [dm.shape for dm in state.casdm2fr]
    with pytest.raises(ValueError):
        bare.casdm1frs[0][...] = 0.0
    with pytest.raises(ValueError):
        bare.mo_coeff[...] = 0.0


def test_inputs_are_copied(state):
    mo = state.mo_coeff.copy()
    copied = LASState(mo, state.casdm1frs, state.casdm2fr)
    mo[...] = 0.0
    assert np.array_equal(copied.mo_coeff, state.mo_coeff)


def test_mismatched_fragment_counts_raise(state):
    with pytest.raises(ValueError, match="fragment"):
        LASState(state.mo_coeff, state.casdm1frs, state.casdm2fr[:1])


def test_load_rejects_other_files(tmp_path, state):
    # The pre-record layout: RDM groups without a header or orbitals.
    with h5py.File(tmp_path / "old.h5", "w") as f:
        f.create_group("casdm1frs").create_dataset("0", data=state.casdm1frs[0])
    with pytest.raises(ValueError, match="LASState"):
        LASState.load(tmp_path / "old.h5")

    state.save(tmp_path / "future.h5")
    with h5py.File(tmp_path / "future.h5", "a") as f:
        f.attrs["format_version"] = LASState.FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="version"):
        LASState.load(tmp_path / "future.h5")


def test_initial_state_gives_mrh_initial_guess_hamiltonians(h6):
    _, las, mo = h6
    explicit = fragment_hamiltonians(las, initial_state(las, mo))
    implicit = _capture_hamiltonians(las, mo, None, None)
    for ham, ref in zip(explicit, implicit):
        for a, b in zip(ham, ref):
            assert np.array_equal(a, b)
