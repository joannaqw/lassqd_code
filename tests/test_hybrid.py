import copy
import pickle
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pytest
from conftest import FullSpaceSampler, empty_circuit, full_counts
from ffsim.qiskit import (
    PRE_INIT,
    FfsimSampler,
    PrepareHartreeFockJW,
    UCJOpSpinUnbalancedJW,
)
from ffsim.random import random_ucj_op_spin_unbalanced
from pyscf import gto, scf
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

from lassqd import (
    FragmentSamples,
    FragmentSQD,
    LASSCFNoSymm,
    build_circuits,
    collect_samples,
    run_lassqd,
    solve_cycle,
    submit_circuits,
)
from lassqd.hybrid import _glue_circuits


def make_solvers(las):
    return [
        FragmentSQD(100, max_iterations=2, num_batches=2, seed=0)
        for _ in range(las.nfrags)
    ]


def h6_las(mf):
    """A new LAS object with the h6 fixture's fragments."""
    return LASSCFNoSymm(mf, (3, 3), ((2, 1), (1, 2)), spin_sub=(2, 2))


def cycle_by_stages(las, mo, solvers, previous=None):
    """One cycle from the stages, continuing from the ``previous`` CycleResult."""
    rdms = {}
    if previous is not None:
        rdms = dict(casdm1frs=previous.casdm1frs, casdm2fr=previous.casdm2fr)
    circuits = build_circuits(las, mo, empty_circuit, **rdms)
    job = submit_circuits(FullSpaceSampler(las), circuits)
    samples = collect_samples(circuits, job.result(), job_id=job.job_id())
    return circuits, samples, solve_cycle(las, samples, solvers)


class RecordingSQD(FragmentSQD):
    """Records the Hamiltonian each call receives."""

    def __call__(self, norb, nelec, h0, h1s, h2):
        self.hamiltonian = (h0, h1s, h2)
        return super().__call__(norb, nelec, h0, h1s, h2)


class FailingSolver:
    def __call__(self, norb, nelec, h0, h1s, h2):
        raise RuntimeError("fragment solve failed")


def run(mf, las, mo, **kwargs):
    return run_lassqd(
        las,
        mo,
        make_solvers(las),
        FullSpaceSampler(las),
        circuit_builder=empty_circuit,
        conv_tol=1e-10,
        max_cycles=100,
        **kwargs,
    )


def test_glue_and_sample_recovers_each_fragment():
    def x_circuit(n, flipped):
        qc = QuantumCircuit(n)
        for q in flipped:
            qc.x(q)
        return qc

    circuits = [x_circuit(2, [0]), x_circuit(3, [2]), x_circuit(2, [0, 1])]
    glued = _glue_circuits(circuits)
    pass_manager = generate_preset_pass_manager(
        backend=AerSimulator(), optimization_level=3
    )
    isa = pass_manager.run(glued)
    data = SamplerV2().run([isa], shots=64).result()[0].data
    # Within a fragment, qubit 0 is the rightmost bit.
    counts = [data[creg.name].get_counts() for creg in glued.cregs]
    assert counts == [{"01": 64}, {"100": 64}, {"11": 64}]


def test_full_space_lassqd_matches_lasscf(h4):
    mf, las, mo = h4
    ref = LASSCFNoSymm(mf, (2, 2), ((1, 1), (1, 1)), spin_sub=(1, 1))
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
    las = LASSCFNoSymm(mf, (6, 6), ((3, 3), (3, 3)), spin_sub=(1, 1))
    mo = las.localize_init_guess((list(range(6)), list(range(6, 12))), mf.mo_coeff)
    ref = LASSCFNoSymm(mf, (6, 6), ((3, 3), (3, 3)), spin_sub=(1, 1))
    ref.kernel(mo.copy())
    # Full-space counts and batches: each fragment solve is exact.
    solvers = [
        FragmentSQD(1000, max_iterations=1, carryover_threshold=None)
        for _ in range(las.nfrags)
    ]
    result = run_lassqd(
        las,
        mo,
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


def test_solvers_get_their_fragment_orbitals(h6):
    mf, las, mo = h6
    solvers = [FragmentSQD(100, max_iterations=1, seed=0) for _ in range(las.nfrags)]
    run_lassqd(
        las,
        mo,
        solvers,
        FullSpaceSampler(las),
        circuit_builder=empty_circuit,
        max_cycles=1,
    )
    # One cycle: the fragments were solved at the input orbitals.
    start = las.ncore
    for solver, norb in zip(solvers, las.ncas_sub):
        assert np.array_equal(solver.las_orbitals, mo[:, start : start + norb])
        assert np.allclose(solver.ao_ovlp, mf.get_ovlp())
        start += norb


def test_stages_reproduce_run_lassqd(h6):
    mf, las, mo = h6
    result = run_lassqd(
        las,
        mo,
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
    step = None
    for _ in range(3):
        _, _, step = cycle_by_stages(las, mo, solvers, step)
        mo = step.mo_coeff
        e_hist.append(step.e_tot)
    np.testing.assert_allclose(e_hist, result.e_hist, rtol=0, atol=1e-10)
    np.testing.assert_allclose(step.mo_coeff, result.mo_coeff, rtol=0, atol=1e-10)
    rdms = zip(
        step.casdm1frs + step.casdm2fr, result.casdm1frs + result.casdm2fr
    )
    for dm, ref in rdms:
        np.testing.assert_allclose(dm, ref, rtol=0, atol=1e-10)


def test_captured_hamiltonians_are_the_solved_ones(h6):
    _, las, mo = h6
    solvers = [RecordingSQD(100, max_iterations=1, seed=0) for _ in range(las.nfrags)]
    # The second cycle continues from the first cycle's orbitals and RDMs.
    step = None
    for _ in range(2):
        circuits, _, step = cycle_by_stages(las, mo, solvers, step)
        mo = step.mo_coeff
        for solver, captured in zip(solvers, circuits.hamiltonians):
            for solved, ref in zip(solver.hamiltonian, captured):
                np.testing.assert_allclose(solved, ref, rtol=0, atol=1e-12)


def test_saved_samples_resolve_identically_in_a_new_las(h6):
    mf, las, mo = h6
    solvers = make_solvers(las)
    # One cycle first, so the solvers have carryover and orbital state.
    _, _, first = cycle_by_stages(las, mo, solvers)
    circuits = build_circuits(
        las,
        first.mo_coeff,
        empty_circuit,
        casdm1frs=first.casdm1frs,
        casdm2fr=first.casdm2fr,
    )
    samples = collect_samples(
        circuits, submit_circuits(FullSpaceSampler(las), circuits).result()
    )
    saved_samples, saved_solvers = pickle.dumps(samples), copy.deepcopy(solvers)
    step = solve_cycle(las, samples, solvers)
    # Resume after a failure: a new LAS object, the saved samples and solver state.
    resumed = solve_cycle(h6_las(mf), pickle.loads(saved_samples), saved_solvers)
    assert np.isclose(resumed.e_tot, step.e_tot, rtol=0, atol=1e-10)
    np.testing.assert_allclose(resumed.mo_coeff, step.mo_coeff, rtol=0, atol=1e-8)
    for solver, saved in zip(solvers, saved_solvers):
        for strings, saved_strings in zip(
            solver.carryover_strings, saved.carryover_strings
        ):
            np.testing.assert_array_equal(strings, saved_strings)


@pytest.mark.parametrize("fail_in", ["setup", "solve"])
def test_solve_cycle_restores_las_settings_on_failure(h6, fail_in):
    _, las, mo = h6
    saved = las.max_cycle_macro, las.max_cycle_rdmjk
    counts = [full_counts(n, ne) for n, ne in zip(las.ncas_sub, las.nelecas_sub)]
    failure = (
        patch.object(las._scf, "get_ovlp", side_effect=RuntimeError("setup failed"))
        if fail_in == "setup"
        else nullcontext()
    )
    with failure, pytest.raises(RuntimeError):
        solve_cycle(las, FragmentSamples(mo, counts), [FailingSolver(), FailingSolver()])
    assert (las.max_cycle_macro, las.max_cycle_rdmjk) == saved


@pytest.mark.parametrize(
    "sampler_type,glue_circuits,shots",
    [
        pytest.param(FfsimSampler, False, None, id="ffsim-default-shots"),
        pytest.param(FfsimSampler, False, 512, id="ffsim-explicit-shots"),
        pytest.param(SamplerV2, True, 512, id="aer-glued"),
        pytest.param(SamplerV2, False, 512, id="aer-separate"),
    ],
)
def test_lucj_sampling_in_hybrid_loop(h6, sampler_type, glue_circuits, shots):
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

    counts_per_cycle = []

    def circuit_builder(h1, h2, norb, nelec):
        # Exercise native LUCJ gates without repeating the variational optimization.
        norb = int(norb)
        nelec = tuple(int(n) for n in nelec)
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

    def record_counts(cycle, las):
        counts_per_cycle.append([solver.counts.copy() for solver in solvers])

    with patch.object(sampler, "run", wraps=sampler.run) as submit:
        result = run_lassqd(
            las,
            mo,
            solvers,
            sampler,
            pass_manager=pass_manager,
            glue_circuits=glue_circuits,
            shots=shots,
            circuit_builder=circuit_builder,
            max_cycles=2,
            conv_tol=0.0,
            callback=record_counts,
        )

    # One job per cycle, with either two independent circuits or one glued circuit.
    assert submit.call_count == 2
    assert len(result.job_ids) == 2
    for call in submit.call_args_list:
        (circuits,) = call.args
        assert [qc.num_qubits for qc in circuits] == ([12] if glue_circuits else [6, 6])
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
