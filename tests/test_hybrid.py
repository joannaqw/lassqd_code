from unittest.mock import patch

import numpy as np
import pytest
from conftest import FullSpaceSampler, empty_circuit
from ffsim.qiskit import (
    PRE_INIT,
    FfsimSampler,
    PrepareHartreeFockJW,
    UCJOpSpinUnbalancedJW,
)
from ffsim.random import random_ucj_op_spin_unbalanced
from mrh.my_pyscf.mcscf.lasci import _DFLASCI
from mrh.my_pyscf.mcscf.lasscf_rdm import LASSCFNoSymm
from pyscf import gto, scf
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

import lassqd.las
from lassqd import (
    LASSCF,
    Embedding,
    FragmentResult,
    FragmentSamples,
    FragmentSQD,
    LASState,
    SQDState,
    build_circuits,
    capture_embedding,
    collect_samples,
    initial_state,
    replay_orbital_step,
    run_lassqd,
    solve_fragments,
    submit_circuits,
    update_orbitals,
)


def make_solvers(las):
    return [
        FragmentSQD(100, max_iterations=2, num_batches=2, seed=0)
        for _ in range(las.nfrags)
    ]


def h6_las(mf):
    """A new LAS object with the h6 fixture's fragments."""
    return LASSCF(mf, (3, 3), ((2, 1), (1, 2)), spin_sub=(2, 2))


def sample(las, state):
    """Capture the embedding of ``state`` and sample it with full-space counts."""
    circuits = build_circuits(capture_embedding(las, state), empty_circuit)
    job = submit_circuits(FullSpaceSampler(las), circuits)
    return collect_samples(circuits, job.result(), job_ids=job.job_id())


def cycle_by_stages(las, state, solvers):
    """One cycle from the stages, starting from ``state``."""
    samples = sample(las, state)
    results = solve_fragments(samples, solvers)
    return samples, update_orbitals(las, samples.embedding, results)


def perturbed(result, **changes):
    fields = {name: getattr(result, name) for name in result.__dataclass_fields__}
    return FragmentResult(**{**fields, **changes})


def run(mf, las, mo, **kwargs):
    return run_lassqd(
        las,
        initial_state(las, mo),
        make_solvers(las),
        FullSpaceSampler(las),
        circuit_builder=empty_circuit,
        conv_tol=1e-10,
        max_cycles=100,
        **kwargs,
    )


def test_full_space_lassqd_matches_lasscf(h4):
    mf, las, mo = h4
    ref = LASSCF(mf, (2, 2), ((1, 1), (1, 1)), spin_sub=(1, 1))
    ref.kernel(mo)
    result = run(mf, las, mo)
    assert result.converged
    assert abs(result.e_tot - ref.e_tot) < 1e-7


def test_exact_fragments_match_lasscf_for_coupled_fragments():
    # Two H6 chains 2 A apart, as in the tutorial. Their embedding in each other
    # matters, so each cycle has to continue from the previous cycle's RDMs.
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    mol = gto.M(atom=[("H", (x, 0, 0)) for x in xs], basis="sto-6g", verbose=0)
    mf = scf.RHF(mol).run()
    las = LASSCF(mf, (6, 6), ((3, 3), (3, 3)), spin_sub=(1, 1))
    mo = las.localize_init_guess((list(range(6)), list(range(6, 12))), mf.mo_coeff)
    ref = LASSCF(mf, (6, 6), ((3, 3), (3, 3)), spin_sub=(1, 1))
    ref.kernel(mo.copy())
    # Full-space counts and batches: each fragment solve is exact.
    solvers = [
        FragmentSQD(1000, max_iterations=1, carryover_threshold=None)
        for _ in range(las.nfrags)
    ]
    result = run_lassqd(
        las,
        initial_state(las, mo),
        solvers,
        FullSpaceSampler(las),
        circuit_builder=empty_circuit,
        conv_tol=1e-12,
    )
    assert result.converged
    # Restarting each cycle from mrh's initial-guess RDMs ends ~2e-7 above LASSCF.
    assert abs(result.e_tot - ref.e_tot) < 1e-8


def test_result_is_a_consistent_wave_function(h6):
    mf, las, mo = h6
    cycles = []
    result = run(mf, las, mo, callback=lambda cycle, step: cycles.append(step.e_tot))
    assert result.converged
    assert cycles == result.e_hist
    # The RDMs are expressed in the active orbitals of result.state.mo_coeff.
    state = result.state
    e = las.energy_nuc() + las.energy_elec(
        mo_coeff=state.mo_coeff,
        casdm1frs=list(state.casdm1frs),
        casdm2fr=list(state.casdm2fr),
    )
    assert np.isclose(e, result.e_tot)
    assert len(result.results) == las.nfrags
    # run_lassqd restores the settings and fragment solvers it overrides.
    assert las.max_cycle_macro != 1
    assert las.max_cycle_rdmjk != 0


def test_carryover_is_kept_in_the_fragment_ao_basis(h6):
    _, las, mo = h6
    solvers = [FragmentSQD(100, max_iterations=1, seed=0) for _ in range(las.nfrags)]
    state = initial_state(las, mo)
    samples, _ = cycle_by_stages(las, state, solvers)
    # One cycle: the fragments were solved at the input orbitals.
    start = las.ncore
    for solver, fragment, norb in zip(
        solvers, samples.embedding.fragments, las.ncas_sub
    ):
        assert np.array_equal(fragment.las_orbitals, mo[:, start : start + norb])
        assert np.array_equal(
            solver.state.carryover_orbitals, fragment.sampling_orbitals
        )
        start += norb


def test_stages_reproduce_run_lassqd(h6):
    mf, las, mo = h6
    result = run_lassqd(
        las,
        initial_state(las, mo),
        make_solvers(las),
        FullSpaceSampler(las),
        circuit_builder=empty_circuit,
        max_cycles=3,
        conv_tol=0.0,
    )
    assert len(result.job_ids) == 3
    las = h6_las(mf)
    solvers = make_solvers(las)
    e_hist = []
    state = initial_state(las, mo)
    for _ in range(3):
        _, step = cycle_by_stages(las, state, solvers)
        state = step.state
        e_hist.append(step.e_tot)
    np.testing.assert_allclose(e_hist, result.e_hist, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        state.mo_coeff, result.state.mo_coeff, rtol=0, atol=1e-10
    )
    rdms = zip(
        state.casdm1frs + state.casdm2fr,
        result.state.casdm1frs + result.state.casdm2fr,
    )
    for dm, ref in rdms:
        np.testing.assert_allclose(dm, ref, rtol=0, atol=1e-10)


def test_replay_receives_the_captured_hamiltonians(h6):
    _, las, mo = h6
    solvers = make_solvers(las)
    checked = []
    real_check = lassqd.las._check_hamiltonian

    def check(fragment, norb, nelec, h0, h1s, h2):
        captured = (fragment.h0, fragment.h1s_las, fragment.h2_las)
        checked.append(
            all(np.array_equal(a, b) for a, b in zip((h0, h1s, h2), captured))
        )
        return real_check(fragment, norb, nelec, h0, h1s, h2)

    # The second cycle continues from the first cycle's orbitals and RDMs.
    state = initial_state(las, mo)
    with patch("lassqd.las._check_hamiltonian", check):
        for _ in range(2):
            _, step = cycle_by_stages(las, state, solvers)
            state = step.state
    # mrh passes bit-identical Hamiltonians, one per fragment and cycle.
    assert checked == [True] * (2 * las.nfrags)


def test_replay_uses_only_the_fragment_rdms(h6):
    # mrh computes the energy from the RDMs; fragment energies are only logged.
    _, las, mo = h6
    samples = sample(las, initial_state(las, mo))
    results = solve_fragments(samples, make_solvers(las))
    step = update_orbitals(las, samples.embedding, results)
    shifted = [perturbed(r, e_tot=r.e_tot + 1.0) for r in results]
    again = update_orbitals(las, samples.embedding, shifted)
    assert again.e_tot == step.e_tot
    assert np.array_equal(again.state.mo_coeff, step.state.mo_coeff)
    for a, b in zip(
        again.state.casdm1frs + again.state.casdm2fr,
        step.state.casdm1frs + step.state.casdm2fr,
    ):
        assert np.array_equal(a, b)


def test_saved_records_resume_identically_in_a_new_las(h6, tmp_path):
    mf, las, mo = h6
    solvers = make_solvers(las)
    # One cycle first, so the solvers have carryover and orbital state.
    _, first = cycle_by_stages(las, initial_state(las, mo), solvers)
    first.state.save(tmp_path / "state.h5")
    samples = sample(las, first.state)
    samples.save(tmp_path / "samples.h5")
    for i, solver in enumerate(solvers):
        solver.state.save(tmp_path / f"sqd_state_{i}.h5")
    step = update_orbitals(las, samples.embedding, solve_fragments(samples, solvers))

    # Resume after a failure: a new LAS object and the saved records.
    las = h6_las(mf)
    state = LASState.load(tmp_path / "state.h5")
    samples = FragmentSamples.load(tmp_path / "samples.h5")
    assert [f.digest() for f in capture_embedding(las, state).fragments] == [
        f.digest() for f in samples.embedding.fragments
    ]
    resumed_solvers = make_solvers(las)
    for i, solver in enumerate(resumed_solvers):
        solver.state = SQDState.load(tmp_path / f"sqd_state_{i}.h5")
    resumed = update_orbitals(
        las, samples.embedding, solve_fragments(samples, resumed_solvers)
    )

    assert resumed.e_tot == step.e_tot
    assert np.array_equal(resumed.state.mo_coeff, step.state.mo_coeff)
    for a, b in zip(
        resumed.state.casdm1frs + resumed.state.casdm2fr,
        step.state.casdm1frs + step.state.casdm2fr,
    ):
        assert np.array_equal(a, b)
    for resumed_solver, solver in zip(resumed_solvers, solvers):
        assert resumed_solver.state.rng_state == solver.state.rng_state
        for a, b in zip(
            resumed_solver.state.carryover_strings, solver.state.carryover_strings
        ):
            assert np.array_equal(a, b)


def test_update_orbitals_checks_results(h6):
    _, las, mo = h6
    samples = sample(las, initial_state(las, mo))
    embedding = samples.embedding
    results = solve_fragments(samples, make_solvers(las))
    with pytest.raises(ValueError, match="results for"):
        update_orbitals(las, embedding, results[:1])
    swapped = [perturbed(results[0], fragment_digest=results[1].fragment_digest)]
    with pytest.raises(ValueError, match="different fragment problem"):
        update_orbitals(las, embedding, swapped + results[1:])
    with pytest.raises(ValueError, match="solvers for"):
        solve_fragments(samples, make_solvers(las)[:1])


@pytest.mark.parametrize("fail_in", ["setup", "hamiltonian"])
def test_replay_restores_las_on_failure(h6, fail_in):
    _, las, mo = h6
    saved = las.fciboxes, las.max_cycle_macro, las.max_cycle_rdmjk
    samples = sample(las, initial_state(las, mo))
    results = solve_fragments(samples, make_solvers(las))
    if fail_in == "setup":
        failure = patch(
            "lassqd.las._set_fragment_kernels",
            side_effect=RuntimeError("setup failed"),
        )
        error = pytest.raises(RuntimeError, match="setup failed")
    else:
        # A Hamiltonian that does not match the one mrh passes during replay.
        def check(fragment, *args):
            raise ValueError("differs from the captured one")

        failure = patch("lassqd.las._check_hamiltonian", check)
        error = pytest.raises(ValueError, match="captured")
    with failure, error:
        update_orbitals(las, samples.embedding, results)
    assert (las.fciboxes, las.max_cycle_macro, las.max_cycle_rdmjk) == saved


def test_replay_rejects_a_different_hamiltonian(h6):
    _, las, mo = h6
    samples = sample(las, initial_state(las, mo))
    embedding = samples.embedding
    results = solve_fragments(samples, make_solvers(las))
    # Other starting RDMs give other embedded Hamiltonians than those captured.
    moved = update_orbitals(las, embedding, results).state
    other_rdms = LASState(embedding.state.mo_coeff, moved.casdm1frs, moved.casdm2fr)
    mismatched = Embedding(other_rdms, embedding.fragments, embedding.ao_ovlp)
    with pytest.raises(ValueError, match="differs from the captured one"):
        replay_orbital_step(las, mismatched, results)


@pytest.mark.parametrize(
    "sampler_type,groups,shots",
    [
        pytest.param(FfsimSampler, [[0], [1]], None, id="ffsim-default-shots"),
        pytest.param(FfsimSampler, [[0], [1]], 512, id="ffsim-explicit-shots"),
        pytest.param(SamplerV2, None, 512, id="aer-glued"),
        pytest.param(SamplerV2, [[0], [1]], 512, id="aer-separate"),
    ],
)
def test_lucj_sampling_in_hybrid_loop(h6, sampler_type, groups, shots):
    _, las, mo = h6
    solvers = [
        FragmentSQD(100, max_iterations=1, num_batches=1, seed=0)
        for _ in range(las.nfrags)
    ]
    sampler = sampler_type(default_shots=256, seed=0)
    pass_manager = None
    if sampler_type is SamplerV2:
        pass_manager = generate_preset_pass_manager(
            backend=AerSimulator(), optimization_level=3, seed_transpiler=0
        )
        pass_manager.pre_init = PRE_INIT

    def circuit_builder(fragment):
        # Exercise native LUCJ gates without repeating the variational optimization.
        norb, nelec = fragment.norb, fragment.nelec
        op = random_ucj_op_spin_unbalanced(
            norb,
            interaction_pairs=(
                [(p, p + 1) for p in range(norb - 1)],
                [(p, p) for p in range(0, norb, 4)],
                [(p, p + 1) for p in range(norb - 1)],
            ),
            with_final_orbital_rotation=True,
            seed=0,
        )
        qc = QuantumCircuit(2 * norb)
        qc.append(PrepareHartreeFockJW(norb, nelec), qc.qubits)
        qc.append(UCJOpSpinUnbalancedJW(op), qc.qubits)
        return qc

    with (
        patch.object(sampler, "run", wraps=sampler.run) as submit,
        patch("lassqd.hybrid.solve_fragments", wraps=solve_fragments) as solve,
    ):
        result = run_lassqd(
            las,
            initial_state(las, mo),
            solvers,
            sampler,
            pass_manager=pass_manager,
            groups=groups,
            shots=shots,
            circuit_builder=circuit_builder,
            max_cycles=2,
            conv_tol=0.0,
        )
    counts_per_cycle = [call.args[0].counts for call in solve.call_args_list]

    # One job per cycle, with either two independent circuits or one glued circuit.
    assert submit.call_count == 2
    assert len(result.job_ids) == 2
    for call in submit.call_args_list:
        (circuits,) = call.args
        assert [qc.num_qubits for qc in circuits] == (
            [12] if groups is None else [6, 6]
        )
    assert len(result.e_hist) == len(counts_per_cycle) == 2
    assert np.all(np.isfinite(result.e_hist))
    for counts in counts_per_cycle:
        for count, norb, nelec in zip(counts, las.ncas_sub, las.nelecas_sub):
            assert sum(count.values()) == (256 if shots is None else shots)
            # Opposite fragment spin polarizations detect swapped results or registers.
            for key in count:
                assert len(key) == 2 * norb
                assert key[norb:].count("1") == nelec[0]
                assert key[:norb].count("1") == nelec[1]


def h4_density_fitted():
    """The h4 fixture's molecule with a density-fitted mean field."""
    mol = gto.M(
        atom="H 0 0 0; H 0.8 0 0; H 3.0 0 0; H 3.8 0 0", basis="6-31g", verbose=0
    )
    return scf.RHF(mol).density_fit().run()


def h4_las(build, mf):
    """``build(mf, ...)`` with the h4 fixture's fragments, and its initial orbitals."""
    las = build(mf, (2, 2), ((1, 1), (1, 1)), spin_sub=(1, 1))
    return las, las.localize_init_guess(([0, 1], [2, 3]), las.sort_mo([1, 2, 3, 4]))


def test_density_fitted_lassqd_matches_density_fitted_lasscf():
    mf = h4_density_fitted()
    las, mo = h4_las(LASSCF, mf)
    assert isinstance(las, _DFLASCI)
    ref, _ = h4_las(LASSCF, mf)
    ref.kernel(mo.copy())
    result = run(mf, las, mo)
    # Full-space counts make each fragment solve exact.
    assert result.converged
    assert abs(result.e_tot - ref.e_tot) < 1e-7


def test_las_without_its_mean_fields_density_fitting_is_rejected():
    mf = h4_density_fitted()
    las, mo = h4_las(LASSCF, mf)
    samples = sample(las, initial_state(las, mo))
    results = solve_fragments(samples, make_solvers(las))
    # Built directly, the LAS object would use exact active-space integrals with
    # the mean field's density-fitted potentials.
    mixed, _ = h4_las(LASSCFNoSymm, mf)
    saved = mixed.fciboxes, mixed.max_cycle_macro, mixed.max_cycle_rdmjk
    match = "LAS object does not"
    with pytest.raises(ValueError, match=match):
        initial_state(mixed, mo)
    with pytest.raises(ValueError, match=match):
        capture_embedding(mixed, samples.embedding.state)
    with pytest.raises(ValueError, match=match):
        replay_orbital_step(mixed, samples.embedding, results)
    assert (mixed.fciboxes, mixed.max_cycle_macro, mixed.max_cycle_rdmjk) == saved
