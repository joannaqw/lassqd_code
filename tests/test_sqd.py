from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
from conftest import full_counts
from pyscf import fci
from qiskit_addon_sqd.fermion import (
    SCIResult,
    diagonalize_fermionic_hamiltonian,
    solve_sci,
    solve_sci_batch,
)

from lassqd import FragmentSQD, fragment_hamiltonians
from lassqd.sqd import permute_carryover


@pytest.fixture
def fragment(h6):
    _, las, mo = h6
    h0, h1s, h2 = fragment_hamiltonians(las, mo)[0]
    return las.ncas_sub[0], las.nelecas_sub[0], h0, h1s, h2


@pytest.fixture
def iterations():
    return []


def summarize(iterations):
    """Derive diagnostics from the actual results received by a caller callback."""
    return {
        "e_hist": np.array([[r.energy for r in results] for results in iterations]),
        "s_hist": np.array(
            [[r.sci_state.spin_square() for r in results] for results in iterations]
        ),
        "a_hist": np.array(
            [[len(r.sci_state.ci_strs_a) for r in results] for results in iterations]
        ),
        "b_hist": np.array(
            [[len(r.sci_state.ci_strs_b) for r in results] for results in iterations]
        ),
        "d_hist": np.array(
            [[r.sci_state.amplitudes.size for r in results] for results in iterations]
        ),
        "occupancy_hist": np.array(
            [
                np.concatenate(min(results, key=lambda r: r.energy).orbital_occupancies)
                for results in iterations
            ]
        ),
    }


def fci_energy(norb, nelec, h1, h2, spin_sq):
    solver = fci.addons.fix_spin_(fci.direct_spin1.FCI(), ss=spin_sq)
    return solver.kernel(h1, h2, norb, nelec)[0]


def test_fragment_sqd_defaults_converge_and_carry_over(iterations):
    solver = FragmentSQD(10, callback=iterations.append)
    h1 = np.diag([0.0, 1.0])
    h2 = np.zeros((2,) * 4)
    solver.counts = {"0101": 10}
    e1, _, _ = solver(2, (1, 1), 0.0, h1, h2)
    history = summarize(iterations)
    assert np.isclose(e1, 0.0)
    assert history["e_hist"].shape == (2, 1)
    assert history["occupancy_hist"].shape == (2, 4)
    assert all(np.array_equal(strings, [1]) for strings in solver.carryover_strings)

    # Default carryover retains the ground state when only an excited state is sampled.
    solver.counts = {"1010": 10}
    iterations.clear()
    e2, _, _ = solver(2, (1, 1), 0.0, h1, h2)
    history = summarize(iterations)
    assert np.isclose(e2, e1)
    assert history["e_hist"].shape == (2, 1)
    assert np.all(history["d_hist"] == 4)


@pytest.mark.parametrize("carryover_threshold", [None, np.inf])
def test_fragment_sqd_random_stream_advances_between_calls(carryover_threshold):
    observations = [[], []]
    solvers = [
        FragmentSQD(
            1,
            num_batches=12,
            max_iterations=1,
            carryover_threshold=carryover_threshold,
            seed=seed,
            callback=observed.append,
        )
        for seed, observed in zip((0, np.random.default_rng(0)), observations)
    ]
    h1 = np.diag([0.0, 1.0])
    h2 = np.zeros((2,) * 4)
    previous = None
    for _ in range(3):
        for solver, observed in zip(solvers, observations):
            observed.clear()
            solver.counts = {"0101": 10, "1010": 10}
            solver(2, (1, 1), 0.0, h1, h2)
            assert solver.carryover_strings is None
        # Integer seeds and supplied generators yield the same sequence of runs.
        energies = [summarize(observed)["e_hist"] for observed in observations]
        assert np.array_equal(*energies)
        if previous is not None:
            assert not np.array_equal(energies[0], previous)
        previous = energies[0].copy()


@pytest.mark.parametrize("carryover_threshold", [None, 1e-3])
def test_fragment_sqd_full_space_is_exact(fragment, carryover_threshold, iterations):
    norb, nelec, h0, h1s, h2 = fragment
    solver = FragmentSQD(
        50,
        max_iterations=2,
        num_batches=2,
        seed=0,
        carryover_threshold=carryover_threshold,
        callback=iterations.append,
    )
    solver.counts = full_counts(norb, nelec)
    # Wrong electron counts must be postselected initially and recovered later.
    solver.counts.update({"0" * (2 * norb): 100, "1" * (2 * norb): 100})
    e, dm1s, dm2 = solver(norb, nelec, h0, h1s, h2)
    history = summarize(iterations)

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
    assert history["e_hist"].shape == (2, 2)
    assert np.allclose(history["e_hist"], e - h0)
    assert np.allclose(history["s_hist"], 0.75)
    assert np.all(history["a_hist"] == 3)
    assert np.all(history["b_hist"] == 3)
    assert np.all(history["d_hist"] == 9)
    assert np.allclose(history["occupancy_hist"][:, :norb].sum(axis=1), nelec[0])
    assert np.allclose(history["occupancy_hist"][:, norb:].sum(axis=1), nelec[1])


@pytest.mark.parametrize("carryover_threshold", [None, 1e-3])
def test_fragment_sqd_returns_best_across_iterations(carryover_threshold, iterations):
    # Two doubly occupied determinants with energies 0 and 2. This seed samples
    # the ground state early, but only the excited determinant in the last round.
    h0 = 2.0
    h1 = np.diag([0.0, 1.0])
    h2 = np.zeros((2,) * 4)
    solver = FragmentSQD(
        1,
        max_iterations=4,
        num_batches=2,
        energy_tol=0.0,
        seed=0,
        carryover_threshold=carryover_threshold,
        callback=iterations.append,
    )
    solver.counts = {"0101": 10, "1010": 10}
    e, dm1s, dm2 = solver(2, (1, 1), h0, h1, h2)
    history = summarize(iterations)

    assert np.isclose(e, h0)
    assert np.isclose(e, h0 + history["e_hist"].min())
    assert np.allclose(dm1s, [np.diag([1, 0])] * 2)
    assert np.isclose(e, h0 + np.einsum("ij,sij->", h1, dm1s))
    assert np.isclose(dm2[0, 0, 0, 0], 2)
    assert history["e_hist"].shape == (4, 2)
    assert np.all(history["d_hist"] == history["a_hist"] * history["b_hist"])
    if carryover_threshold is None:
        assert history["e_hist"][-1].min() > e - h0
        assert np.all(history["d_hist"] == 1)
        assert solver.carryover_strings is None
        assert np.allclose(history["occupancy_hist"][-1], [0, 1, 0, 1])
    else:
        # Carryover retains the ground determinant even when it is not sampled.
        assert np.allclose(history["e_hist"][-1], 0)
        assert np.all(history["d_hist"][-1] == 4)
        assert all(np.array_equal(strings, [1]) for strings in solver.carryover_strings)
        assert np.allclose(history["occupancy_hist"][-1], [1, 0, 1, 0])


def test_fragment_sqd_carryover_persists_between_calls(fragment):
    norb, nelec, h0, h1s, h2 = fragment
    solver = FragmentSQD(
        50,
        max_iterations=1,
        num_batches=2,
        seed=0,
        carryover_threshold=1e-3,
    )
    solver.counts = full_counts(norb, nelec)
    e1, _, _ = solver(norb, nelec, h0, h1s, h2)
    assert len(solver.carryover_strings[0]) > 0
    # Second call: only one determinant is sampled, the rest must come from carryover.
    solver.counts = {next(iter(solver.counts)): 1}
    e2, _, _ = solver(norb, nelec, h0, h1s, h2)
    assert np.isclose(e1, e2)


@pytest.mark.parametrize(
    "energy_tol,occupancies_tol,expected_iterations",
    [(0.0, 0.0, 5), (1e-8, 0.0, 5), (0.0, 1e-5, 5), (1e-8, 1e-5, 2)],
)
def test_fragment_sqd_convergence_and_callback(
    energy_tol, occupancies_tol, expected_iterations, iterations
):
    callback = iterations.append
    solver = FragmentSQD(
        10,
        max_iterations=5,
        num_batches=2,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        callback=callback,
    )
    solver.counts = {"1010": 10}
    h1 = np.diag([0.0, 1.0])
    # The addon receives the caller's callback itself, without an internal wrapper.
    with patch(
        "lassqd.sqd.diagonalize_fermionic_hamiltonian",
        wraps=diagonalize_fermionic_hamiltonian,
    ) as diagonalize:
        e, _, _ = solver(2, (1, 1), 3.0, h1, np.zeros((2,) * 4))
    assert diagonalize.call_args.kwargs["callback"] is callback
    assert np.isclose(e, 5.0)
    assert len(iterations) == expected_iterations
    history = summarize(iterations)
    for name in ("e_hist", "s_hist", "d_hist", "a_hist", "b_hist"):
        assert history[name].shape == (expected_iterations, 2)
    assert history["occupancy_hist"].shape == (expected_iterations, 4)
    assert np.allclose(history["e_hist"], 2.0)

    # The caller controls whether to retain observations between solves.
    solver(2, (1, 1), 3.0, h1, np.zeros((2,) * 4))
    assert len(iterations) == 2 * expected_iterations
    iterations.clear()
    solver(2, (1, 1), 3.0, h1, np.zeros((2,) * 4))
    assert len(iterations) == expected_iterations

    # Both the callback and batch dimensions can change between calls.
    replacement = []
    solver.sqd_options.update(
        max_iterations=1, num_batches=3, callback=replacement.append
    )
    solver(2, (1, 1), 3.0, h1, np.zeros((2,) * 4))
    assert len(iterations) == expected_iterations
    assert len(replacement) == 1
    assert summarize(replacement)["e_hist"].shape == (1, 3)
    assert summarize(replacement)["occupancy_hist"].shape == (1, 4)

    solver.sqd_options["callback"] = None
    solver(2, (1, 1), 3.0, h1, np.zeros((2,) * 4))
    assert len(replacement) == 1


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
        50,
        max_iterations=2,
        num_batches=3,
        sci_solver=sci_solver,
        callback=callbacks.append,
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
    assert all(
        np.isclose(r.sci_state.spin_square(), 0.75)
        for results in callbacks
        for r in results
    )


@pytest.mark.parametrize("spin_sq,energy", [(None, 1.05), (0.0, 1.15)])
def test_fragment_sqd_spin_constraint_is_configured_through_sci_solver(spin_sq, energy):
    # Positive exchange favors a triplet. The addon default should find it;
    # a configured solver can instead target the higher-energy singlet.
    h1 = np.diag([0.0, 0.1])
    h2 = np.zeros((2,) * 4)
    h2[0, 0, 0, 0] = h2[1, 1, 1, 1] = 2.0
    h2[0, 0, 1, 1] = h2[1, 1, 0, 0] = 1.0
    for index in [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)]:
        h2[index] = 0.05
    sci_solver = (
        None
        if spin_sq is None
        else partial(solve_sci_batch, spin_sq=spin_sq, max_cycle=200, tol=1e-12)
    )
    solver = FragmentSQD(4, max_iterations=1, sci_solver=sci_solver)
    solver.counts = full_counts(2, (1, 1))
    e, dm1s, dm2 = solver(2, (1, 1), 0.0, h1, h2)
    assert np.isclose(e, energy)
    assert np.isclose(
        solver.sci_state.spin_square(), 2.0 if spin_sq is None else spin_sq
    )
    assert np.isclose(
        e, np.einsum("ij,sij->", h1, dm1s) + 0.5 * np.einsum("ijkl,ijkl", h2, dm2)
    )


def test_fragment_sqd_initial_occupancies_recover_entirely_noisy_counts():
    h1 = np.diag([0.0, 1.0])
    h2 = np.zeros((2,) * 4)
    solver = FragmentSQD(10, max_iterations=1, seed=0)
    solver.counts = {"0000": 10, "1111": 10}
    with pytest.raises(ValueError, match="valid bitstrings"):
        solver(2, (1, 1), 0.0, h1, h2)
    solver.sqd_options["initial_occupancies"] = (
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
    )
    e, dm1s, _ = solver(2, (1, 1), 0.0, h1, h2)
    assert np.isclose(e, 0.0)
    assert np.allclose(dm1s, [np.diag([1.0, 0.0])] * 2)


@pytest.mark.parametrize("symmetrize_spin", [False, True])
def test_fragment_sqd_spin_symmetry(symmetrize_spin):
    solver = FragmentSQD(1, max_iterations=1, symmetrize_spin=symmetrize_spin)
    solver.counts = {"1001": 10}
    e, _, _ = solver(2, (1, 1), 0.0, np.diag([0.0, 1.0]), np.zeros((2,) * 4))
    assert np.isclose(e, 0.0 if symmetrize_spin else 1.0)
    assert solver.sci_state.amplitudes.size == (4 if symmetrize_spin else 1)
    if symmetrize_spin:
        assert np.array_equal(solver.sci_state.ci_strs_a, solver.sci_state.ci_strs_b)


@pytest.mark.parametrize("carryover_threshold", [None, 1e-3])
@pytest.mark.parametrize("include", [[1], np.array([1]), ([1], [2])])
def test_fragment_sqd_included_configurations_and_carryover(
    include, carryover_threshold
):
    h1 = np.diag([0.0, 1.0, 2.0])
    h2 = np.zeros((3,) * 4)
    solver = FragmentSQD(1, max_iterations=1, carryover_threshold=carryover_threshold)
    solver.counts = {"010010": 10}
    solver(3, (1, 1), 0.0, h1, h2)
    solver.sqd_options["include_configurations"] = include
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
def test_fragment_sqd_max_dim(max_dim, expected_dims, iterations):
    solver = FragmentSQD(
        10,
        max_iterations=2,
        num_batches=2,
        max_dim=max_dim,
        include_configurations=[1, 2, 4],
        callback=iterations.append,
    )
    solver.counts = full_counts(3, (1, 1))
    e, _, _ = solver(3, (1, 1), 0.0, np.diag([0.0, 1.0, 2.0]), np.zeros((3,) * 4))
    history = summarize(iterations)
    assert np.isclose(e, 0.0)
    assert np.all(history["a_hist"] == expected_dims[0])
    assert np.all(history["b_hist"] == expected_dims[1])
    assert np.all(history["d_hist"] == np.prod(expected_dims))


def test_fragment_sqd_requires_counts(fragment):
    norb, nelec, h0, h1s, h2 = fragment
    with pytest.raises(RuntimeError):
        FragmentSQD(10)(norb, nelec, h0, h1s, h2)


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
