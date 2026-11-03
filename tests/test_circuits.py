import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

from lassqd import (
    Embedding,
    Fragment,
    FragmentCircuits,
    LASState,
    build_circuits,
    collect_samples,
    submit_circuits,
)
from lassqd.circuits import _glue_circuits

# Qubits flipped in each fragment's circuit; qubit 0 is the rightmost bit.
FLIPS = {0: [0, 2], 1: [1, 3], 2: [0, 3]}
EXPECTED = ["0101", "1010", "1001"]


@pytest.fixture
def embedding():
    """Three (2e, 2o) fragments with trivial Hamiltonians."""
    nfrags, norb = 3, 2
    h1 = np.diag([0.0, 1.0])
    h2 = np.zeros((norb,) * 4)
    orbitals = np.eye(nfrags * norb)
    fragments = [
        Fragment.from_hamiltonian(
            norb,
            (1, 1),
            0.0,
            h1,
            h2,
            index=i,
            las_orbitals=orbitals[:, norb * i : norb * (i + 1)],
        )
        for i in range(nfrags)
    ]
    state = LASState(
        orbitals,
        [np.zeros((2, norb, norb))] * nfrags,
        [np.zeros((norb,) * 4)] * nfrags,
    )
    return Embedding(state, fragments, np.eye(nfrags * norb))


def x_builder(fragment):
    qc = QuantumCircuit(2 * fragment.norb)
    for q in FLIPS[fragment.index]:
        qc.x(q)
    return qc


@pytest.fixture
def pass_manager():
    return generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1)


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


@pytest.mark.parametrize(
    "groups,widths",
    [
        pytest.param(None, [12], id="all-glued"),
        pytest.param([[0], [1], [2]], [4, 4, 4], id="separate"),
        pytest.param([[0, 2], [1]], [8, 4], id="mixed"),
        pytest.param([[2, 0], [1]], [8, 4], id="mixed-reordered"),
    ],
)
def test_groups_recover_each_fragment(embedding, pass_manager, groups, widths):
    circuits = build_circuits(
        embedding, x_builder, pass_manager=pass_manager, groups=groups
    )
    assert [qc.num_qubits for qc in circuits.circuits] == widths
    job = submit_circuits(SamplerV2(), circuits, shots=16)
    samples = collect_samples(circuits, job.result(), job_ids=job.job_id())
    assert samples.counts == tuple({key: 16} for key in EXPECTED)
    assert samples.job_ids == (job.job_id(),) * len(widths)
    assert samples.embedding is embedding


def test_groups_can_be_sampled_in_separate_jobs(embedding, pass_manager):
    circuits = build_circuits(
        embedding, x_builder, pass_manager=pass_manager, groups=[[0, 2], [1]]
    )
    sampler = SamplerV2()
    jobs = [submit_circuits(sampler, circuits, shots=8, indices=[i]) for i in (0, 1)]
    results = [job.result()[0] for job in jobs]
    samples = collect_samples(circuits, results, job_ids=[job.job_id() for job in jobs])
    assert samples.counts == tuple({key: 8} for key in EXPECTED)
    assert samples.job_ids == tuple(job.job_id() for job in jobs)


@pytest.mark.parametrize(
    "groups",
    [pytest.param(None, id="all-glued"), pytest.param([[0, 2], [1]], id="mixed")],
)
def test_circuits_round_trip(tmp_path, embedding, pass_manager, groups):
    # A QPU gateway samples circuits prepared and saved elsewhere.
    circuits = build_circuits(
        embedding, x_builder, pass_manager=pass_manager, groups=groups
    )
    circuits.save(tmp_path / "circuits.h5")
    loaded = FragmentCircuits.load(tmp_path / "circuits.h5")
    assert loaded.circuits == circuits.circuits
    assert loaded.output_registers == circuits.output_registers
    assert [f.digest() for f in loaded.embedding.fragments] == [
        f.digest() for f in embedding.fragments
    ]
    job = submit_circuits(SamplerV2(), loaded, shots=16)
    samples = collect_samples(loaded, job.result(), job_ids=job.job_id())
    assert samples.counts == tuple({key: 16} for key in EXPECTED)


@pytest.mark.parametrize(
    "groups",
    [
        pytest.param([[0], [1]], id="missing"),
        pytest.param([[0, 1], [1, 2]], id="duplicated"),
        pytest.param([[0, 1, 2], []], id="empty"),
        pytest.param([[0, 1, 3]], id="out-of-range"),
    ],
)
def test_invalid_groups_raise(embedding, groups):
    with pytest.raises(ValueError, match="groups"):
        build_circuits(embedding, x_builder, groups=groups)


def test_collect_samples_checks_lengths(embedding, pass_manager):
    circuits = build_circuits(
        embedding, x_builder, pass_manager=pass_manager, groups=[[0, 2], [1]]
    )
    result = submit_circuits(SamplerV2(), circuits, shots=4).result()
    with pytest.raises(ValueError, match="pub results"):
        collect_samples(circuits, [result[0]])
    with pytest.raises(ValueError, match="job identifiers"):
        collect_samples(circuits, result, job_ids=["a"])
