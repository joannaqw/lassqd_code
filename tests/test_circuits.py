import ffsim
import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

from lassqd import fragment_hamiltonians, lucj_circuit


def test_lucj_circuit_conserves_particle_number(h6):
    _, las, mo = h6
    _, h1s, h2 = fragment_hamiltonians(las, mo)[0]
    norb, (neleca, nelecb) = las.ncas_sub[0], las.nelecas_sub[0]
    qc = lucj_circuit(h1s[0], h2, norb, (neleca, nelecb))
    assert qc.num_qubits == 2 * norb
    pass_manager = generate_preset_pass_manager(
        backend=AerSimulator(), optimization_level=3
    )
    pass_manager.pre_init = ffsim.qiskit.PRE_INIT
    qc.measure_all()
    isa = pass_manager.run(qc)
    data = SamplerV2().run([isa], shots=500).result()[0].data
    counts = data[isa.cregs[0].name].get_counts()
    for key in counts:
        # Alpha on the right half, beta on the left.
        assert key[norb:].count("1") == neleca
        assert key[:norb].count("1") == nelecb
    assert len(counts) > 1
    assert np.isclose(sum(counts.values()), 500)
