import numpy as np
from pyscf import gto, cc
import ffsim
from qiskit_aer import AerSimulator
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler import transpile
from ffsim.optimize import minimize_linear_method


def append_measurements(qc):
    n = qc.num_qubits
    qrg = QuantumRegister(n)
    crx = ClassicalRegister(n)
    psi = QuantumCircuit(qrg, crx)
    psi.append(qc, range(n))
    for i in range(n):
        psi.measure(qrg[i], crx[i])
    return psi


def uniform_circuit(n):
    qrg = QuantumRegister(n)
    psi = QuantumCircuit(qrg)
    for i in range(n):
        psi.h(i)
    return psi


class QASM_simulator:
    def __init__(self, shots=1024, alpha=0.0):
        self.shots = shots
        self.alpha = alpha
        self.simul = AerSimulator(method="matrix_product_state")

    def run_circuit(self, qc):
        qc_clean = transpile(qc, self.simul)
        qc_noise = uniform_circuit(qc_clean.num_qubits)
        qc_noise = transpile(append_measurements(qc_noise), self.simul)
        n_noise = sum(np.random.binomial(1, self.alpha, self.shots))
        n_noise = max(int(n_noise), 1)
        n_noise = min(n_noise, self.shots - 1)
        print("fraction of noise ", self.alpha, self.alpha * self.shots, n_noise)
        n_clean = self.shots - n_noise
        counts = []

        for n, qc in zip([n_clean, n_noise], [qc_clean, qc_noise]):
            r = self.simul.run(qc, shots=n).result().get_counts(0)
            counts.append(r)
        r = {
            k: counts[0].get(k, 0) + counts[1].get(k, 0)
            for k in set(counts[0]) | set(counts[1])
        }
        r = {k: v for k, v in sorted(r.items(), key=lambda item: item[1])}
        return r


def LUCJ_circuit(norb, nelec_a, nelec_b, spin_sub, hcore, eri, mo_counter):
    """LUCJ circuit for each fragment and sampler"""

    """convert las h1, h2 to mo basis"""
    mol = gto.M()
    mol.nelectron = nelec_a + nelec_b
    mol.spin = spin_sub
    # 2*np.abs(nelec_a-nelec_b)
    mol.nao = norb
    mf_as = mol.ROHF()
    mf_as.get_hcore = lambda *args: hcore
    mf_as.get_ovlp = lambda *args: np.eye(norb)
    mf_as._eri = eri
    mf_as.kernel()
    C = mf_as.mo_coeff
    h0e_MO = mf_as.mol.energy_nuc()
    h1e_MO = np.einsum("pi,pr,rj->ij", C, hcore, C, optimize=True)
    h2e_MO = np.einsum("pi,rj,prqs,qk,sl->ijkl", C, C, eri, C, C, optimize=True)

    mol_MO = gto.M()
    mol_MO.nelectron = nelec_a + nelec_b
    mol_MO.spin = spin_sub
    # 2*np.abs(nelec_a-nelec_b)
    mol_MO.nao = norb
    mf_as_MO = mol_MO.ROHF()
    mf_as_MO.get_hcore = lambda *args: h1e_MO
    mf_as_MO.get_ovlp = lambda *args: np.eye(norb)
    mf_as_MO._eri = h2e_MO
    mf_as_MO.kernel()
    h0e_MO = mf_as_MO.mol.energy_nuc()
    mc = cc.CCSD(mf_as_MO)
    mc.kernel()
    t1 = mc.t1
    t2 = mc.t2
    """save mf mo_coeff here"""
    if mo_counter % 2 == 0:
        np.save(f"frag0_mo_{mo_counter}", mf_as_MO.mo_coeff)
    else:
        np.save(f"frag1_mo_{mo_counter}", mf_as_MO.mo_coeff)

    # ------- UCJ operator for each fragment
    H = ffsim.MolecularHamiltonian(h1e_MO, h2e_MO, h0e_MO)
    # linop = ffsim.linear_operator(H,norb=norb,nelec=(nelec_a,nelec_b))
    n_reps = 1
    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(t2, n_reps=n_reps, t1=t1)
    nelec = (nelec_a, nelec_b)
    reference_state = ffsim.hartree_fock_state(norb, nelec)
    # ansatz_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)
    hamiltonian = ffsim.linear_operator(H, norb=norb, nelec=nelec)
    # energy = np.real(np.vdot(ansatz_state, hamiltonian @ ansatz_state))

    # ------- LUCJ and optimization
    alpha_alpha_indices = [(p, p + 1) for p in range(norb - 1)]
    alpha_beta_indices = [(p, p) for p in range(0, norb, 4)]
    beta_beta_indices = [(p, p + 1) for p in range(norb - 1)]
    interaction_pairs = (alpha_alpha_indices, alpha_beta_indices, beta_beta_indices)

    def params_to_vec(x: np.ndarray) -> np.ndarray:
        operator = ffsim.UCJOpSpinUnbalanced.from_parameters(
            x,
            norb=norb,
            n_reps=n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=True,
        )
        return ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

    result = minimize_linear_method(
        params_to_vec,
        hamiltonian,
        x0=operator.to_parameters(interaction_pairs=interaction_pairs),
    )
    # rng = np.random.default_rng()
    # x = rng.uniform(-10,10,size=(ffsim.UCJOpSpinUnbalanced.n_params(norb,n_reps,interaction_pairs=interaction_pairs,with_final_orbital_rotation=True)))
    ucj_op = ffsim.UCJOpSpinUnbalanced.from_parameters(
        result.x,
        norb=norb,
        n_reps=n_reps,
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=True,
    )

    from qiskit import QuantumCircuit, QuantumRegister

    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op), qubits)
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    # Initialize quantum device backend
    backend = GenericBackendV2(2 * norb, basis_gates=["cp", "xx_plus_yy", "p", "x"])
    # Create a pass manager for circuit transpilation
    LEVEL = 3
    pass_manager = generate_preset_pass_manager(
        optimization_level=LEVEL, backend=backend
    )
    pass_manager.pre_init = ffsim.qiskit.PRE_INIT
    # circuit = circuit.decompose(reps=1)
    # transpiled = pass_manager.run(circuit)
    # print("Optimization level ",LEVEL,transpiled.count_ops(),transpiled.depth())
    # qc = append_measurements(circuit)
    # X = QASM_simulator(shots=10_000,alpha=0)
    # r = X.run_circuit(qc)
    # counts = r
    # print(r)
    return circuit
