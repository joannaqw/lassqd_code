import h5py
import numpy as np
import pytest

from lassqd import (
    Embedding,
    Fragment,
    FragmentSamples,
    LASState,
    capture_embedding,
    fragment_hamiltonians,
    initial_state,
)
from lassqd.fragment import fragment_mo_basis, fragment_rohf, from_mo
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


@pytest.fixture
def embedding(h6):
    _, las, mo = h6
    return capture_embedding(las, initial_state(las, mo))


def test_fragment_basis_matches_fragment_mo_basis(embedding):
    for fragment in embedding.fragments:
        args = (fragment.h1s_las[0], fragment.h2_las, fragment.norb, fragment.nelec)
        basis, h1, h2 = fragment_mo_basis(*args)
        assert np.array_equal(fragment.basis, basis)
        assert np.array_equal(fragment.h1, h1)
        assert np.array_equal(fragment.h2, h2)
        mf = fragment_rohf(*args)
        assert np.array_equal(fragment.mo_occ, mf.mo_occ)
        assert np.array_equal(fragment.mo_energy, mf.mo_energy)
        assert fragment.rohf_converged


def test_embedding_carries_las_orbitals_and_overlap(h6, embedding):
    mf, las, _ = h6
    start = las.ncore
    for i, (fragment, norb) in enumerate(zip(embedding.fragments, las.ncas_sub)):
        assert fragment.index == i
        assert (fragment.norb, fragment.nelec) == (norb, tuple(las.nelecas_sub[i]))
        assert type(fragment.norb) is int
        assert all(type(n) is int for n in fragment.nelec)
        mo = embedding.state.mo_coeff[:, start : start + norb]
        assert np.array_equal(fragment.las_orbitals, mo)
        assert np.array_equal(fragment.sampling_orbitals, mo @ fragment.basis)
        start += norb
    assert np.array_equal(embedding.ao_ovlp, mf.get_ovlp())


def test_fragment_defaults_and_conversions(embedding):
    ref = embedding.fragments[0]
    norb, nelec = ref.norb, ref.nelec
    # A spin-free h1 is used for both spins; the LAS basis defaults to fixed.
    fragment = Fragment.from_hamiltonian(norb, nelec, 0.5, ref.h1s_las[0], ref.h2_las)
    assert fragment.h1s_las.shape == (2, norb, norb)
    assert fragment.spin_asymmetry == 0.0
    assert np.array_equal(fragment.las_orbitals, np.eye(norb))
    assert np.array_equal(fragment.basis, ref.basis)
    assert ref.spin_asymmetry == np.max(np.abs(ref.h1s_las[0] - ref.h1s_las[1]))
    dm1s = np.stack([np.diag(ref.mo_occ > 0), np.diag(ref.mo_occ > 1)]).astype(float)
    dm2 = np.zeros((norb,) * 4)
    for a, b in zip(ref.rdms_to_las(dm1s, dm2), from_mo(ref.basis, dm1s, dm2)):
        assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        ref.h1[...] = 0.0


def test_fragment_digest_identifies_the_problem(embedding):
    ref = embedding.fragments[0]
    same = Fragment.from_hamiltonian(
        ref.norb, ref.nelec, ref.h0, ref.h1s_las, ref.h2_las
    )
    assert same.digest() == ref.digest()
    h1s = ref.h1s_las.copy()
    h1s[0, 0, 0] += 1e-12
    changed = Fragment.from_hamiltonian(ref.norb, ref.nelec, ref.h0, h1s, ref.h2_las)
    assert changed.digest() != ref.digest()
    assert embedding.fragments[1].digest() != ref.digest()


def test_embedding_and_samples_validate_fragments(embedding):
    fragments = embedding.fragments
    with pytest.raises(ValueError, match="fragments"):
        Embedding(embedding.state, fragments[:1], embedding.ao_ovlp)
    with pytest.raises(ValueError, match="index"):
        Embedding(embedding.state, fragments[::-1], embedding.ao_ovlp)
    with pytest.raises(ValueError, match="counts"):
        FragmentSamples(embedding, [{"0": 1}])
