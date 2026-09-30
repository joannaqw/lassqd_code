import ffsim
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

from lassqd import fragment_hamiltonians, glue_circuits, lucj_circuit


def x_circuit(n, flipped):
    qc = QuantumCircuit(n)
    for q in flipped:
        qc.x(q)
    return qc


def test_glue_and_sample_recovers_each_fragment():
    circuits = [x_circuit(2, [0]), x_circuit(3, [2]), x_circuit(2, [0, 1])]
    glued = glue_circuits(circuits)
    pass_manager = generate_preset_pass_manager(
        backend=AerSimulator(), optimization_level=3
    )
    isa = pass_manager.run(glued)
    data = SamplerV2().run([isa], shots=64).result()[0].data
    # Within a fragment, qubit 0 is the rightmost bit.
    counts = [data[creg.name].get_counts() for creg in glued.cregs]
    assert counts == [{"01": 64}, {"100": 64}, {"11": 64}]


def test_lucj_circuit_conserves_particle_number(h6):
    _, las, mo = h6
    h0, h1s, h2 = fragment_hamiltonians(las, mo)[0]
    norb, (neleca, nelecb) = las.ncas_sub[0], las.nelecas_sub[0]
    qc = lucj_circuit(h1s[0], h2, norb, (neleca, nelecb))
    assert qc.num_qubits == 2 * norb
    pass_manager = generate_preset_pass_manager(
        backend=AerSimulator(), optimization_level=3
    )
    pass_manager.pre_init = ffsim.qiskit.PRE_INIT
    isa = pass_manager.run(glue_circuits([qc]))
    counts = SamplerV2().run([isa], shots=500).result()[0].data[isa.cregs[0].name].get_counts()
    for key in counts:
        # Alpha on the right half, beta on the left.
        assert key[norb:].count("1") == neleca
        assert key[:norb].count("1") == nelecb
    assert len(counts) > 1
    assert np.isclose(sum(counts.values()), 500)
