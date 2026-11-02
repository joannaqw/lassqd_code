from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
from conftest import full_counts, solve_hamiltonian
from pyscf import fci
from qiskit_addon_sqd.fermion import (
    SCIResult,
    SCIState,
    diagonalize_fermionic_hamiltonian,
    solve_sci,
    solve_sci_batch,
)

from lassqd import (
    Fragment,
    FragmentSQD,
    SQDState,
    capture_embedding,
    initial_state,
)
from lassqd.sqd import permute_carryover


@pytest.fixture
def fragment(h6):
    """The first h6 fragment, ``(norb, nelec, h0, h1s, h2)`` in the LAS basis."""
    _, las, mo = h6
    f = capture_embedding(las, initial_state(las, mo)).fragments[0]
    return f.norb, f.nelec, f.h0, f.h1s_las, f.h2_las


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
    e1 = solve_hamiltonian(solver, {"0101": 10}, 2, (1, 1), 0.0, h1, h2).e_tot
    history = summarize(iterations)
    assert np.isclose(e1, 0.0)
    assert history["e_hist"].shape == (2, 1)
    assert history["occupancy_hist"].shape == (2, 4)
    assert all(
        np.array_equal(strings, [1]) for strings in solver.state.carryover_strings
    )

    # Default carryover retains the ground state when only an excited state is sampled.
    iterations.clear()
    e2 = solve_hamiltonian(solver, {"1010": 10}, 2, (1, 1), 0.0, h1, h2).e_tot
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
            counts = {"0101": 10, "1010": 10}
            solve_hamiltonian(solver, counts, 2, (1, 1), 0.0, h1, h2)
            assert solver.state.carryover_strings is None
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
    counts = full_counts(norb, nelec)
    # Wrong electron counts must be postselected initially and recovered later.
    counts.update({"0" * (2 * norb): 100, "1" * (2 * norb): 100})
    result = solve_hamiltonian(solver, counts, norb, nelec, h0, h1s, h2)
    e, dm1s, dm2 = result.e_tot, result.dm1s, result.dm2
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
    result = solve_hamiltonian(solver, {"0101": 10, "1010": 10}, 2, (1, 1), h0, h1, h2)
    e, dm1s, dm2 = result.e_tot, result.dm1s, result.dm2
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
        assert solver.state.carryover_strings is None
        assert np.allclose(history["occupancy_hist"][-1], [0, 1, 0, 1])
    else:
        # Carryover retains the ground determinant even when it is not sampled.
        assert np.allclose(history["e_hist"][-1], 0)
        assert np.all(history["d_hist"][-1] == 4)
        assert all(
            np.array_equal(strings, [1]) for strings in solver.state.carryover_strings
        )
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
    counts = full_counts(norb, nelec)
    e1 = solve_hamiltonian(solver, counts, norb, nelec, h0, h1s, h2).e_tot
    assert len(solver.state.carryover_strings[0]) > 0
    # Second solve: only one determinant is sampled, the rest must come from carryover.
    counts = {next(iter(counts)): 1}
    e2 = solve_hamiltonian(solver, counts, norb, nelec, h0, h1s, h2).e_tot
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
    h1 = np.diag([0.0, 1.0])
    h2 = np.zeros((2,) * 4)

    def solve():
        return solve_hamiltonian(solver, {"1010": 10}, 2, (1, 1), 3.0, h1, h2).e_tot

    # The addon receives the caller's callback itself, without an internal wrapper.
    with patch(
        "lassqd.sqd.diagonalize_fermionic_hamiltonian",
        wraps=diagonalize_fermionic_hamiltonian,
    ) as diagonalize:
        e = solve()
    assert diagonalize.call_args.kwargs["callback"] is callback
    assert np.isclose(e, 5.0)
    assert len(iterations) == expected_iterations
    history = summarize(iterations)
    for name in ("e_hist", "s_hist", "d_hist", "a_hist", "b_hist"):
        assert history[name].shape == (expected_iterations, 2)
    assert history["occupancy_hist"].shape == (expected_iterations, 4)
    assert np.allclose(history["e_hist"], 2.0)

    # The caller controls whether to retain observations between solves.
    solve()
    assert len(iterations) == 2 * expected_iterations
    iterations.clear()
    solve()
    assert len(iterations) == expected_iterations

    # Both the callback and batch dimensions can change between solves.
    replacement = []
    solver.options.update(max_iterations=1, num_batches=3)
    solver.callback = replacement.append
    solve()
    assert len(iterations) == expected_iterations
    assert len(replacement) == 1
    assert summarize(replacement)["e_hist"].shape == (1, 3)
    assert summarize(replacement)["occupancy_hist"].shape == (1, 4)

    solver.callback = None
    solve()
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
    result = solve_hamiltonian(
        solver, full_counts(norb, nelec), norb, nelec, h0, h1s, h2
    )
    e, dm1s, dm2 = result.e_tot, result.dm1s, result.dm2
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
    result = solve_hamiltonian(solver, full_counts(2, (1, 1)), 2, (1, 1), 0.0, h1, h2)
    e, dm1s, dm2 = result.e_tot, result.dm1s, result.dm2
    assert np.isclose(e, energy)
    assert np.isclose(
        result.sci_state.spin_square(), 2.0 if spin_sq is None else spin_sq
    )
    assert np.isclose(
        e, np.einsum("ij,sij->", h1, dm1s) + 0.5 * np.einsum("ijkl,ijkl", h2, dm2)
    )


def test_fragment_sqd_initial_occupancies_recover_entirely_noisy_counts():
    h1 = np.diag([0.0, 1.0])
    h2 = np.zeros((2,) * 4)
    solver = FragmentSQD(10, max_iterations=1, seed=0)
    counts = {"0000": 10, "1111": 10}
    with pytest.raises(ValueError, match="valid bitstrings"):
        solve_hamiltonian(solver, counts, 2, (1, 1), 0.0, h1, h2)
    solver.options["initial_occupancies"] = (
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
    )
    result = solve_hamiltonian(solver, counts, 2, (1, 1), 0.0, h1, h2)
    e, dm1s = result.e_tot, result.dm1s
    assert np.isclose(e, 0.0)
    assert np.allclose(dm1s, [np.diag([1.0, 0.0])] * 2)


@pytest.mark.parametrize("symmetrize_spin", [False, True])
def test_fragment_sqd_spin_symmetry(symmetrize_spin):
    solver = FragmentSQD(1, max_iterations=1, symmetrize_spin=symmetrize_spin)
    result = solve_hamiltonian(
        solver, {"1001": 10}, 2, (1, 1), 0.0, np.diag([0.0, 1.0]), np.zeros((2,) * 4)
    )
    assert np.isclose(result.e_tot, 0.0 if symmetrize_spin else 1.0)
    assert result.amplitudes.size == (4 if symmetrize_spin else 1)
    if symmetrize_spin:
        assert np.array_equal(result.ci_strs_a, result.ci_strs_b)


@pytest.mark.parametrize("carryover_threshold", [None, 1e-3])
@pytest.mark.parametrize("include", [[1], np.array([1]), ([1], [2])])
def test_fragment_sqd_included_configurations_and_carryover(
    include, carryover_threshold
):
    h1 = np.diag([0.0, 1.0, 2.0])
    h2 = np.zeros((3,) * 4)
    solver = FragmentSQD(1, max_iterations=1, carryover_threshold=carryover_threshold)
    solve_hamiltonian(solver, {"010010": 10}, 3, (1, 1), 0.0, h1, h2)
    solver.options["include_configurations"] = include
    result = solve_hamiltonian(solver, {"100100": 10}, 3, (1, 1), 0.0, h1, h2)
    separate_spins = isinstance(include, tuple)
    expected_a = {1, 4}
    expected_b = {2, 4} if separate_spins else {1, 4}
    if carryover_threshold is not None:
        expected_a.add(2)
        expected_b.add(2)
    assert set(result.ci_strs_a) == expected_a
    assert set(result.ci_strs_b) == expected_b
    assert np.isclose(result.e_tot, 1.0 if separate_spins else 0.0)


@pytest.mark.parametrize(
    "reorder_las_basis,expected_strings,expected_energy",
    [
        # Same LAS orbitals; LAS orbitals 0 and 1 swap energy order, so the
        # carried ROHF orbital 1 (LAS orbital 1) becomes ROHF orbital 0.
        (False, {1, 4}, 0.0),
        # LAS orbitals 0 and 1 swap places, as LASSCF canonicalization can do; the
        # Hamiltonian is the same physically, so the carried orbital is unchanged.
        (True, {2, 4}, 2.0),
    ],
)
def test_fragment_sqd_carryover_follows_orbitals_in_ao_basis(
    reorder_las_basis, expected_strings, expected_energy
):
    # Non-orthonormal AO basis; the LAS orbitals are S-orthonormal.
    ao_ovlp = np.diag([1.0, 2.0, 4.0])
    las_orbitals = np.diag(ao_ovlp.diagonal() ** -0.5)
    h2 = np.zeros((3,) * 4)
    solver = FragmentSQD(1, max_iterations=1, carryover_threshold=1e-3)
    solve_hamiltonian(
        solver,
        {"010010": 10},
        3,
        (1, 1),
        0.0,
        np.diag([0.0, 1.0, 2.0]),
        h2,
        las_orbitals=las_orbitals,
        ao_ovlp=ao_ovlp,
    )
    # Carried over: ROHF orbital 1 (energy 1) doubly occupied.
    assert all(
        np.array_equal(strings, [2]) for strings in solver.state.carryover_strings
    )
    if reorder_las_basis:
        las_orbitals = las_orbitals[:, [1, 0, 2]]
    result = solve_hamiltonian(
        solver,
        {"100100": 10},
        3,
        (1, 1),
        0.0,
        np.diag([1.0, 0.0, 2.0]),
        h2,
        las_orbitals=las_orbitals,
        ao_ovlp=ao_ovlp,
    )
    assert set(result.ci_strs_a) == expected_strings
    assert set(result.ci_strs_b) == expected_strings
    assert np.isclose(result.e_tot, expected_energy)


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
    e = solve_hamiltonian(
        solver,
        full_counts(3, (1, 1)),
        3,
        (1, 1),
        0.0,
        np.diag([0.0, 1.0, 2.0]),
        np.zeros((3,) * 4),
    ).e_tot
    history = summarize(iterations)
    assert np.isclose(e, 0.0)
    assert np.all(history["a_hist"] == expected_dims[0])
    assert np.all(history["b_hist"] == expected_dims[1])
    assert np.all(history["d_hist"] == np.prod(expected_dims))


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


def test_failed_solve_leaves_state_unchanged():
    solver = FragmentSQD(10, max_iterations=1, seed=0)
    before = solver.state
    with pytest.raises(ValueError, match="valid bitstrings"):
        solve_hamiltonian(
            solver,
            {"0000": 10, "1111": 10},
            2,
            (1, 1),
            0.0,
            np.diag([0.0, 1.0]),
            np.zeros((2,) * 4),
        )
    assert solver.state is before


def assert_same_state(a, b):
    assert a.rng_state == b.rng_state
    assert (a.carryover_strings is None) == (b.carryover_strings is None)
    if a.carryover_strings is not None:
        for x, y in zip(a.carryover_strings, b.carryover_strings):
            assert np.array_equal(x, y)
        assert np.array_equal(a.carryover_orbitals, b.carryover_orbitals)


def assert_same_result(a, b):
    assert a.e_tot == b.e_tot
    for name in ("dm1s", "dm2", "amplitudes", "ci_strs_a", "ci_strs_b"):
        assert np.array_equal(getattr(a, name), getattr(b, name))
    assert a.fragment_digest == b.fragment_digest


def test_restoring_state_reproduces_a_solve(fragment):
    norb, nelec, h0, h1s, h2 = fragment
    solver = FragmentSQD(5, max_iterations=2, num_batches=2, seed=0)
    counts = full_counts(norb, nelec)
    solve_hamiltonian(solver, counts, norb, nelec, h0, h1s, h2)
    # The second solve continues the random stream and the carryover.
    saved = solver.state
    first = solve_hamiltonian(solver, counts, norb, nelec, h0, h1s, h2)
    after_first = solver.state
    solver.state = saved
    again = solve_hamiltonian(solver, counts, norb, nelec, h0, h1s, h2)
    assert_same_result(first, again)
    assert_same_state(after_first, solver.state)
    # Without restoring, the next solve draws different batches.
    assert solver.state.rng_state != saved.rng_state


def test_prepare_run_finalize_is_step(fragment):
    norb, nelec, h0, h1s, h2 = fragment
    f = Fragment.from_hamiltonian(norb, nelec, h0, h1s, h2)
    solver = FragmentSQD(5, max_iterations=2, num_batches=2, seed=0)
    counts = full_counts(norb, nelec)
    state = solver.state
    inputs = solver.prepare(f, counts, state)
    split = solver.finalize(f, inputs, inputs.run())
    together = solver.step(f, counts, state)
    assert_same_result(split[0], together[0])
    assert_same_state(split[1], together[1])
    # step and prepare leave the solver's own state alone.
    assert solver.state is state


def test_fragment_result_sci_state(fragment):
    norb, nelec, h0, h1s, h2 = fragment
    solver = FragmentSQD(50, max_iterations=1)
    result = solve_hamiltonian(
        solver, full_counts(norb, nelec), norb, nelec, h0, h1s, h2
    )
    state = result.sci_state
    assert isinstance(state, SCIState)
    assert np.array_equal(state.amplitudes, result.amplitudes)
    assert (state.norb, tuple(state.nelec)) == (norb, nelec)
    assert np.isclose(np.sum(state.amplitudes**2), 1.0)


def test_sqd_state_copies_generators():
    rng = np.random.default_rng(0)
    state = SQDState.from_seed(rng)
    rng.random(10)
    assert state.rng().random() == np.random.default_rng(0).random()
    assert SQDState.from_seed(0).rng_state == state.rng_state
    with pytest.raises(ValueError, match="together"):
        SQDState(state.rng_state, (np.array([1]), np.array([1])))


def popcounts(strings):
    return [int(s).bit_count() for s in strings]


@pytest.mark.parametrize(
    "overlap,old,new",
    [
        # Both new orbitals overlap old orbital 1 most; the review's example
        # turned alpha 0b10 into 0b11 and beta 0b01 into 0b0.
        pytest.param(
            [[0.6, 0.8], [-0.7, 0.71]], (0b10, 0b01), ({0b01}, {0b10}), id="2x2"
        ),
        # New orbitals 0 and 2 both overlap old orbital 1 most.
        pytest.param(
            [[0.3, 0.9, 0.3], [0.1, 0.2, 0.97], [0.95, 0.96, 0.1]],
            (0b110, 0b001),
            ({0b011}, {0b100}),
            id="3x3",
        ),
    ],
)
def test_permute_carryover_preserves_electrons_when_orbitals_mix(overlap, old, new):
    norb = len(overlap)
    strings_a, strings_b = np.array([old[0]]), np.array([old[1]])
    expected_a, expected_b = new
    new_a, new_b = permute_carryover(strings_a, strings_b, np.array(overlap), norb)
    assert popcounts(new_a) == popcounts(strings_a)
    assert popcounts(new_b) == popcounts(strings_b)
    assert set(new_a) == expected_a
    assert set(new_b) == expected_b


def test_permute_carryover_preserves_electrons_for_random_rotations():
    rng = np.random.default_rng(0)
    norb = 8
    strings = np.array(
        [
            sum(1 << int(p) for p in rng.choice(norb, 3, replace=False))
            for _ in range(20)
        ]
    )
    for _ in range(50):
        q, _ = np.linalg.qr(rng.normal(size=(norb, norb)))
        new_a, new_b = permute_carryover(strings, strings, q, norb)
        assert set(popcounts(new_a)) == set(popcounts(new_b)) == {3}


def test_permute_carryover_keeps_the_largest_overlap_when_it_is_a_permutation():
    rng = np.random.default_rng(1)
    norb = 6
    strings = np.arange(1 << norb)
    for _ in range(20):
        perm = rng.permutation(norb)
        signs = rng.choice([-1.0, 1.0], norb)
        overlap = np.eye(norb)[perm] * signs[:, None] + 0.2 * rng.normal(
            size=(norb, norb)
        )
        rho = np.argmax(np.abs(overlap), axis=1)
        if len(set(rho.tolist())) < norb:
            continue
        expected = np.unique(((strings[:, None] >> rho) & 1) @ (1 << np.arange(norb)))
        new_a, _ = permute_carryover(strings, strings, overlap, norb)
        assert np.array_equal(new_a, expected)


def test_permute_carryover_rejects_64_orbitals():
    with pytest.raises(ValueError, match="63 orbitals"):
        permute_carryover(np.array([1]), np.array([1]), np.eye(64), 64)


def test_batch_solver_receives_writable_integrals(fragment):
    # Records are read-only, but batch solvers such as Fulqrum's need writable
    # buffers.
    norb, nelec, h0, h1s, h2 = fragment
    received = []

    def sci_solver(ci_strings, h1, h2, norb, nelec):
        received.append((h1.flags.writeable, h2.flags.writeable))
        return solve_sci_batch(ci_strings, h1, h2, norb, nelec)

    solver = FragmentSQD(50, max_iterations=1, sci_solver=sci_solver)
    solve_hamiltonian(solver, full_counts(norb, nelec), norb, nelec, h0, h1s, h2)
    assert received == [(True, True)]
