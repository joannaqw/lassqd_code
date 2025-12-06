from pyscf import gto, scf, lib
from mrh.my_pyscf.mcscf.lasscf_rdm2 import extremeAsynLASSCF, make_fcibox
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from LUCJ_sampler import LUCJ_circuit
from fragment_sqsd_solver import sqsd_fragment
from parallel_circuits_execution import glue_circuits, cut_results

lib.logger.TIMER_LEVEL = lib.logger.INFO
mol_name = "fefe.xyz"
basis_paper = {"Fe": "6-31g", "C": "6-31g", "H": "6-31g", "O": "6-31g", "N": "6-31g"}
mol = gto.M(atom="fefe.xyz", verbose=4, spin=0, charge=4, basis=basis_paper)
mol.build()
mf = scf.ROHF(mol)
mf.init_guess = "atom"
mf = mf.density_fit()
mf.mo_coeff = np.load("fefe_as.npy")
mf.kernel()
guess_mo_coeff = np.load("as_increase_avas.npy")
# ncas,nelecas,guess_mo_coeff=avas.kernel(mf,['Fe 3d'] ,minao=mol.basis)
mo_list = [
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
]
las = extremeAsynLASSCF(mf, (10, 10), ((4, 2), (2, 4)), spin_sub=(3, 3))
guess_mo_sorted = las.sort_mo(mo_list, guess_mo_coeff)
mo_localized = las.localize_init_guess(([0], [1]), guess_mo_sorted)
las.max_cycle_rdmjk = 0
num_orbitals = [10, 10]
num_elec_a = [4, 2]
num_elec_b = [2, 4]
open_shell = True
spin_sq = 2
max_davidson = 200
nfrag = 2
# 210 * 45 = 9450; 9450*0.8 = 7560; 7560/45 = 168
# ------- parallel with two kernels-------:
# --kernel 1: glue all fragment circuits -> make them a list  -> send to hardware for jobs -> process the strings
# --kernel 2: take in hardware results, run the same kernel logic
qcs = []
rs = []
counter = 0
mo_counter = 0


def get_kernel_fn(ifrag, jobtype="quantum"):
    print("start get_kernel_fn")

    def kernel(norb, nelec, h0, h1s, h2):
        global counter
        global mo_counter
        if jobtype == "quantum":
            print("glue fragment circuits and send to hardware")
            qc = LUCJ_circuit(
                num_orbitals[ifrag],
                num_elec_a[ifrag],
                num_elec_b[ifrag],
                spin_sq,
                h1s[0, :, :],
                h2,
                mo_counter,
            )
            qcs.append(qc)
            mo_counter += 1
            return None, None, None
        elif jobtype == "classical":
            print("processing results from hardware")
            results = rs[ifrag]
            print("results", results, flush=True)
            e, d, dm1, dm2 = sqsd_fragment(
                h1s[0, :, :],
                h2,
                num_elec_a[ifrag],
                num_elec_b[ifrag],
                num_orbitals[ifrag],
                spin_sq=spin_sq,
                iterations=6,
                n_batches=15,
                samples_per_batch=170,
                max_davidson_cycles=200,
                results=results,
                counter=counter,
            )
            counter += 1
            etot = np.min(e) + h0
            dm1s = np.zeros((2, norb, norb))
            dm1s[0, :, :] = dm1[0]
            dm1s[1, :, :] = dm1[1]
            return etot, dm1s, dm2

    return kernel


def run_hybrid_cycle(mo):
    global rs
    # ------ quantum kernel first
    qcs.clear()
    las.fciboxes = [
        make_fcibox(las.mol, kernel=get_kernel_fn(ifrag, jobtype="quantum"))
        for ifrag in range(nfrag)
    ]
    las.kernel(mo)
    # ------ get results from quantum kernel
    qc = glue_circuits(qcs)  # append measurements included
    simulator = AerSimulator(method="matrix_product_state")
    qc = transpile(qc, simulator)
    r = simulator.run(qc, shots=100_000).result().get_counts()
    rs.clear()
    rs = cut_results(r)
    # for rx in rs:
    #    print('rx',rx)
    # ----- then classical kernel
    las.fciboxes = [
        make_fcibox(las.mol, kernel=get_kernel_fn(ifrag, jobtype="classical"))
        for ifrag in range(nfrag)
    ]
    las.max_cycle_rdmjk = 0
    las.max_cycle_macro = 1
    las.kernel(mo)
    mo = las.mo_coeff
    e_tot = las.e_tot
    return mo, e_tot


n_cycles = 50
# last_mo = np.load('current_orb.npy')
mo_init = mo_localized
e_last = None
for cycle in range(n_cycles):
    print(f"Starting hybrid macro cycle {cycle}")
    new_mo, e_new = run_hybrid_cycle(mo_init)
    mo_init = new_mo
    # save mo_coeff inside the loop
    np.save("current_orb", new_mo)
    if e_last is not None:
        dE = np.abs(e_new - e_last)
        print(f"Cycle {cycle} energy: {e_new:.10f}, ΔE = {dE:.2e}")
        if dE < 10e-6:
            print("convergence reached")
            break
    e_last = e_new
