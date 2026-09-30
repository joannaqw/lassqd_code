import numpy as np
import pytest
from conftest import full_counts
from pyscf import fci
from qiskit_addon_sqd.fermion import SCIResult, solve_sci

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


@pytest.mark.parametrize("nroots", [1, 3])
def test_solve_sci_nroots(fragment, nroots):
    norb, nelec, _, h1s, h2 = fragment
    _, h1, h2 = fragment_mo_basis(h1s[0], h2, norb, nelec)
    strings = all_strings(norb, nelec)
    ref = solve_sci(strings, h1, h2, norb, nelec, spin_sq=0.75)
    assert (
        solve_sci_nroots(strings, h1, h2, norb, nelec, spin_sq=0.75).energy
        == ref.energy
    )
    multi = solve_sci_nroots(strings, h1, h2, norb, nelec, spin_sq=0.75, nroots=nroots)
    assert np.isclose(multi.energy, ref.energy)
    assert np.isclose(multi.sci_state.spin_square(), 0.75)


@pytest.mark.parametrize("carryover_threshold", [None, 1e-3])
@pytest.mark.parametrize("nroots", [None, 1, 3])
def test_fragment_sqd_full_space_is_exact(fragment, carryover_threshold, nroots):
    norb, nelec, h0, h1s, h2 = fragment
    solver = FragmentSQD(
        2, 2, 50, seed=0, carryover_threshold=carryover_threshold, nroots=nroots
    )
    solver.counts = full_counts(norb, nelec)
    # Wrong electron counts must be postselected initially and recovered later.
    solver.counts.update({"0" * (2 * norb): 100, "1" * (2 * norb): 100})
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
    assert np.allclose(solver.e_hist, e - h0)
    assert np.allclose(solver.s_hist, 0.75)
    assert np.all(solver.a_hist == 3)
    assert np.all(solver.b_hist == 3)
    assert np.all(solver.d_hist == 9)
    assert np.allclose(solver.occupancy_hist[:, :norb].sum(axis=1), nelec[0])
    assert np.allclose(solver.occupancy_hist[:, norb:].sum(axis=1), nelec[1])


@pytest.mark.parametrize("carryover_threshold", [None, 1e-3])
def test_fragment_sqd_returns_best_across_iterations(carryover_threshold):
    # Two doubly occupied determinants with energies 0 and 2. This seed samples
    # the ground state early, but only the excited determinant in the last round.
    h0 = 2.0
    h1 = np.diag([0.0, 1.0])
    h2 = np.zeros((2,) * 4)
    solver = FragmentSQD(4, 2, 1, seed=0, carryover_threshold=carryover_threshold)
    solver.counts = {"0101": 10, "1010": 10}
    e, dm1s, dm2 = solver(2, (1, 1), h0, h1, h2)

    assert np.isclose(e, h0)
    assert np.isclose(e, h0 + solver.e_hist.min())
    assert np.allclose(dm1s, [np.diag([1, 0])] * 2)
    assert np.isclose(e, h0 + np.einsum("ij,sij->", h1, dm1s))
    assert np.isclose(dm2[0, 0, 0, 0], 2)
    assert solver.e_hist.shape == (4, 2)
    assert np.all(solver.d_hist == solver.a_hist * solver.b_hist)
    if carryover_threshold is None:
        assert solver.e_hist[-1].min() > e - h0
        assert np.all(solver.d_hist == 1)
        assert solver.carryover_strings is None
        assert np.allclose(solver.occupancy_hist[-1], [0, 1, 0, 1])
    else:
        # Carryover retains the ground determinant even when it is not sampled.
        assert np.allclose(solver.e_hist[-1], 0)
        assert np.all(solver.d_hist[-1] == 4)
        assert all(np.array_equal(strings, [1]) for strings in solver.carryover_strings)
        assert np.allclose(solver.occupancy_hist[-1], [1, 0, 1, 0])


def test_fragment_sqd_carryover_persists_between_calls(fragment, tmp_path):
    norb, nelec, h0, h1s, h2 = fragment
    solver = FragmentSQD(
        1, 2, 50, seed=0, carryover_threshold=1e-3, output_dir=tmp_path
    )
    solver.counts = full_counts(norb, nelec)
    e1, _, _ = solver(norb, nelec, h0, h1s, h2)
    assert len(solver.carryover_strings[0]) > 0
    # Second call: only one determinant is sampled, the rest must come from carryover.
    solver.counts = {next(iter(solver.counts)): 1}
    e2, _, _ = solver(norb, nelec, h0, h1s, h2)
    assert np.isclose(e1, e2)
    assert (tmp_path / "alpha_strings.npy").exists()
    assert (tmp_path / "e_hist.npy").exists()


@pytest.mark.parametrize(
    "energy_tol,occupancies_tol,expected_iterations",
    [(0.0, 0.0, 5), (1e-8, 0.0, 5), (0.0, 1e-5, 5), (1e-8, 1e-5, 2)],
)
def test_fragment_sqd_convergence_and_callback(
    energy_tol, occupancies_tol, expected_iterations, tmp_path
):
    seen = []

    def callback(results):
        # Internal histories are recorded before the user's callback runs.
        assert np.allclose(solver.e_hist[len(seen)], [r.energy for r in results])
        seen.append(results)

    solver = FragmentSQD(
        5,
        2,
        10,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        callback=callback,
        output_dir=tmp_path,
    )
    solver.counts = {"1010": 10}
    h1 = np.diag([0.0, 1.0])
    e, _, _ = solver(2, (1, 1), 3.0, h1, np.zeros((2,) * 4))
    assert np.isclose(e, 5.0)
    assert len(seen) == expected_iterations
    for name in ("e_hist", "s_hist", "d_hist", "a_hist", "b_hist"):
        assert getattr(solver, name).shape == (expected_iterations, 2)
    assert solver.occupancy_hist.shape == (expected_iterations, 4)
    assert np.allclose(solver.e_hist, 2.0)
    assert np.array_equal(np.load(tmp_path / "e_hist.npy"), solver.e_hist)
    # Reusing a solver resets its histories, including after early convergence.
    seen.clear()
    solver(2, (1, 1), 3.0, h1, np.zeros((2,) * 4))
    assert len(seen) == expected_iterations
    assert solver.e_hist.shape == (expected_iterations, 2)


def test_fragment_sqd_custom_solver_without_rdms(fragment):
    norb, nelec, h0, h1s, h2 = fragment
    batches = []
    callbacks = []

    def sci_solver(ci_strings, h1, h2, norb, nelec):
        results = []
        for strings in ci_strings:
            result = solve_sci(strings, h1, h2, norb, nelec, spin_sq=0.75)
            # RDMs are optional in the addon's custom-solver contract.
            results.append(
                SCIResult(result.energy, result.sci_state, result.orbital_occupancies)
            )
        batches.append(results)
        return results

    solver = FragmentSQD(
        2,
        3,
        50,
        sci_solver=sci_solver,
        callback=callbacks.append,
        # These default-solver settings must not override the custom solver.
        nroots=100,
        spin_sq=100.0,
        max_davidson_cycles=0,
    )
    solver.counts = full_counts(norb, nelec)
    e, dm1s, dm2 = solver(norb, nelec, h0, h1s, h2)
    assert len(batches) == len(callbacks) == 2
    assert all(a is b and len(a) == 3 for a, b in zip(batches, callbacks))
    assert np.isclose(e, h0 + fci_energy(norb, nelec, h1s[0], h2, 0.75))
    assert np.isclose(
        e,
        h0
        + np.einsum("ij,sij->", h1s[0], dm1s)
        + 0.5 * np.einsum("ijkl,ijkl", h2, dm2),
    )
    assert np.allclose(solver.s_hist, 0.75)


def test_fragment_sqd_initial_occupancies_recover_entirely_noisy_counts():
    h1 = np.diag([0.0, 1.0])
    h2 = np.zeros((2,) * 4)
    solver = FragmentSQD(1, 1, 10, seed=0)
    solver.counts = {"0000": 10, "1111": 10}
    with pytest.raises(ValueError, match="valid bitstrings"):
        solver(2, (1, 1), 0.0, h1, h2)
    solver.initial_occupancies = (np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    e, dm1s, _ = solver(2, (1, 1), 0.0, h1, h2)
    assert np.isclose(e, 0.0)
    assert np.allclose(dm1s, [np.diag([1.0, 0.0])] * 2)


@pytest.mark.parametrize("symmetrize_spin", [False, True])
def test_fragment_sqd_spin_symmetry(symmetrize_spin):
    solver = FragmentSQD(1, 1, 1, symmetrize_spin=symmetrize_spin)
    solver.counts = {"1001": 10}
    e, _, _ = solver(2, (1, 1), 0.0, np.diag([0.0, 1.0]), np.zeros((2,) * 4))
    assert np.isclose(e, 0.0 if symmetrize_spin else 1.0)
    assert solver.d_hist[0, 0] == (4 if symmetrize_spin else 1)
    if symmetrize_spin:
        assert np.array_equal(solver.sci_state.ci_strs_a, solver.sci_state.ci_strs_b)


@pytest.mark.parametrize("carryover_threshold", [None, 1e-3])
@pytest.mark.parametrize("include", [[1], np.array([1]), ([1], [2])])
def test_fragment_sqd_included_configurations_and_carryover(
    include, carryover_threshold
):
    h1 = np.diag([0.0, 1.0, 2.0])
    h2 = np.zeros((3,) * 4)
    solver = FragmentSQD(1, 1, 1, carryover_threshold=carryover_threshold)
    solver.counts = {"010010": 10}
    solver(3, (1, 1), 0.0, h1, h2)
    solver.include_configurations = include
    solver.counts = {"100100": 10}
    e, _, _ = solver(3, (1, 1), 0.0, h1, h2)
    separate_spins = isinstance(include, tuple)
    expected_a = {1, 4}
    expected_b = {2, 4} if separate_spins else {1, 4}
    if carryover_threshold is not None:
        expected_a.add(2)
        expected_b.add(2)
    assert set(solver.sci_state.ci_strs_a) == expected_a
    assert set(solver.sci_state.ci_strs_b) == expected_b
    assert np.isclose(e, 1.0 if separate_spins else 0.0)


@pytest.mark.parametrize("max_dim,expected_dims", [(1, (1, 1)), ((2, 1), (2, 1))])
def test_fragment_sqd_max_dim(max_dim, expected_dims):
    solver = FragmentSQD(2, 2, 10, max_dim=max_dim, include_configurations=[1, 2, 4])
    solver.counts = full_counts(3, (1, 1))
    e, _, _ = solver(3, (1, 1), 0.0, np.diag([0.0, 1.0, 2.0]), np.zeros((3,) * 4))
    assert np.isclose(e, 0.0)
    assert np.all(solver.a_hist == expected_dims[0])
    assert np.all(solver.b_hist == expected_dims[1])
    assert np.all(solver.d_hist == np.prod(expected_dims))


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
