import numpy as np
import pytest
from conftest import full_counts
from pyscf import fci
from qiskit_addon_sqd.fermion import solve_sci

from lassqd import FragmentSQD, fragment_hamiltonians, solve_sci_nroots
from lassqd.basis import fragment_mo_basis
from lassqd.sqd import permute_carryover


@pytest.fixture
def fragment(h6):
    _, las, mo = h6
    h0, h1s, h2 = fragment_hamiltonians(las, mo)[0]
    return las.ncas_sub[0], las.nelecas_sub[0], h0, h1s, h2


def fci_energy(norb, nelec, h1, h2, spin_sq):
    solver = fci.addons.fix_spin_(fci.direct_spin1.FCI(), ss=spin_sq)
    return solver.kernel(h1, h2, norb, nelec)[0]


def all_strings(norb, nelec):
    return tuple(fci.cistring.make_strings(range(norb), n) for n in nelec)


def test_solve_sci_nroots(fragment):
    norb, nelec, _, h1s, h2 = fragment
    _, h1, h2 = fragment_mo_basis(h1s[0], h2, norb, nelec)
    strings = all_strings(norb, nelec)
    ref = solve_sci(strings, h1, h2, norb, nelec, spin_sq=0.75)
    assert solve_sci_nroots(strings, h1, h2, norb, nelec, spin_sq=0.75).energy == ref.energy
    multi = solve_sci_nroots(strings, h1, h2, norb, nelec, spin_sq=0.75, nroots=3)
    assert np.isclose(multi.energy, ref.energy)
    assert np.isclose(multi.sci_state.spin_square(), 0.75)


@pytest.mark.parametrize("carryover_threshold", [None, 1e-3])
def test_fragment_sqd_full_space_is_exact(fragment, carryover_threshold):
    norb, nelec, h0, h1s, h2 = fragment
    solver = FragmentSQD(2, 2, 50, seed=0, carryover_threshold=carryover_threshold)
    solver.counts = full_counts(norb, nelec)
    e, dm1s, dm2 = solver(norb, nelec, h0, h1s, h2)

    assert np.isclose(e, h0 + fci_energy(norb, nelec, h1s[0], h2, 0.75))
    # The returned RDMs, in the LAS basis, belong to the returned state.
    e_rdm = (
        h0
        + np.einsum("ij,ij", h1s[0], dm1s[0] + dm1s[1])
        + 0.5 * np.einsum("ijkl,ijkl", h2, dm2)
    )
    assert np.isclose(e, e_rdm)
    assert dm1s.shape == (2, norb, norb)
    assert np.isclose(np.trace(dm1s[0]), nelec[0])
    assert np.isclose(np.trace(dm1s[1]), nelec[1])
    assert solver.e_hist.shape == (2, 2)


def test_fragment_sqd_carryover_persists_between_calls(fragment, tmp_path):
    norb, nelec, h0, h1s, h2 = fragment
    solver = FragmentSQD(1, 2, 50, seed=0, carryover_threshold=1e-3, output_dir=tmp_path)
    solver.counts = full_counts(norb, nelec)
    e1, _, _ = solver(norb, nelec, h0, h1s, h2)
    assert len(solver.carryover_strings[0]) > 0
    # Second call: only one determinant is sampled, the rest must come from carryover.
    solver.counts = {next(iter(solver.counts)): 1}
    e2, _, _ = solver(norb, nelec, h0, h1s, h2)
    assert np.isclose(e1, e2)
    assert (tmp_path / "alpha_strings.npy").exists()
    assert (tmp_path / "e_hist.npy").exists()


def test_fragment_sqd_requires_counts(fragment):
    norb, nelec, h0, h1s, h2 = fragment
    with pytest.raises(RuntimeError):
        FragmentSQD(1, 1, 10)(norb, nelec, h0, h1s, h2)


def test_permute_carryover():
    norb = 3
    strings_a = np.array([0b011, 0b101])
    strings_b = np.array([0b001, 0b001])
    same_a, same_b = permute_carryover(strings_a, strings_b, np.eye(norb), norb)
    assert set(same_a) == {0b011, 0b101}
    assert set(same_b) == {0b001}
    # New orbital p is old orbital 2 - p.
    reverse = np.eye(norb)[::-1]
    new_a, new_b = permute_carryover(strings_a, strings_b, reverse, norb)
    assert set(new_a) == {0b110, 0b101}
    assert set(new_b) == {0b100}
