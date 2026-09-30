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
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

from lassqd import FragmentSQD, LASSCFNoSymm, run_lassqd
from lassqd.hybrid import _glue_circuits


def run(mf, las, mo, **kwargs):
    solvers = [
        FragmentSQD(100, max_iterations=2, num_batches=2, seed=0)
        for _ in range(las.nfrags)
    ]
    return run_lassqd(
        las,
        mo,
        solvers,
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
