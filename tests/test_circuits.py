import numpy as np
import pytest
from qiskit import QuantumCircuit

from lassqd import aer_sampler, cut_counts, fragment_hamiltonians, glue_circuits, lucj_circuit


def x_circuit(n, flipped):
    qc = QuantumCircuit(n)
    for q in flipped:
        qc.x(q)
    return qc


def test_cut_counts_orders_fragments_first_to_last():
    # Qiskit writes the last register first.
    counts = {"11 000 01": 3, "11 100 01": 2}
    assert cut_counts(counts) == [{"01": 5}, {"000": 3, "100": 2}, {"11": 5}]


def test_glue_and_sample_recovers_each_fragment():
    circuits = [x_circuit(2, [0]), x_circuit(3, [2]), x_circuit(2, [0, 1])]
    counts = aer_sampler(shots=64)(glue_circuits(circuits))
    # Within a fragment, qubit 0 is the rightmost bit.
    assert counts == [{"01": 64}, {"100": 64}, {"11": 64}]


def test_lucj_circuit_conserves_particle_number(h6):
    pytest.importorskip("ffsim")
    _, las, mo = h6
    h0, h1s, h2 = fragment_hamiltonians(las, mo)[0]
    norb, (neleca, nelecb) = las.ncas_sub[0], las.nelecas_sub[0]
    qc = lucj_circuit(h1s[0], h2, norb, (neleca, nelecb))
    assert qc.num_qubits == 2 * norb
    counts = aer_sampler(shots=500, method="statevector")(glue_circuits([qc]))[0]
    for key in counts:
        # Alpha on the right half, beta on the left.
        assert key[norb:].count("1") == neleca
        assert key[:norb].count("1") == nelecb
    assert len(counts) > 1
    assert np.isclose(sum(counts.values()), 500)
