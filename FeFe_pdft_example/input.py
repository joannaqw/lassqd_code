import h5py
import os
from pyscf import gto, scf, lib, ao2mo, tools
from mrh.my_pyscf.mcscf.lasscf_rdm2 import extremeAsynLASSCF, make_fcibox
import numpy as np
from pyscf.mcscf import avas
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
lib.logger.TIMER_LEVEL=lib.logger.INFO
mol_name='fefe.xyz'
basis_paper={'Fe': '6-31g','C':'6-31g', 'H':'6-31g','O':'6-31g','N':'6-31g'}
mol=gto.M(atom='fefe.xyz',verbose=4,spin=0,charge=4,basis=basis_paper)
mol.build()
mf=scf.ROHF(mol)
mf.init_guess='atom'
mf=mf.density_fit()
mf.kernel()
ncas,nelecas,guess_mo_coeff=avas.kernel(mf,['Fe 3d'] ,minao=mol.basis)
mo_list = [100,101,102,103,104,105,106,107,108,109]
las =extremeAsynLASSCF(mf,(5,5),((4,2),(2,4)),spin_sub =(3,3))
guess_mo_sorted=las.sort_mo(mo_list,guess_mo_coeff)
mo_localized=las.localize_init_guess(([0],[1]),guess_mo_sorted)
las.max_cycle_rdmjk=0
num_orbitals = [5,5]
num_elec_a = [4,2]
num_elec_b = [2,4]
open_shell = True
spin_sq = 2 
max_davidson = 200 
nfrag = 2   
def save_rdms_to_hdf5(dm1s_list,dm2_list, filename):
    if not os.path.exists("RDMS"):
        os.makedirs("RDMS")
    with h5py.File(filename, 'w') as f:
        dm1_group = f.create_group('casdm1frs')
        for i, dm1 in enumerate(dm1s_list):
            if dm1 is not None:
                dm1 = dm1.reshape((1, 2, dm1.shape[1], dm1.shape[2]))
                dm1_group.create_dataset(str(i), data=dm1)

        dm2_group = f.create_group('casdm2frs')
        for i, dm2 in enumerate(dm2_list):
            if dm2 is not None:
                dm2 = dm2.reshape((1,) + dm2.shape)
                dm2_group.create_dataset(str(i), data=dm2)

#------- parallel with two kernels-------:
#--kernel 1: glue all fragment circuits -> make them a list  -> send to hardware for jobs -> process the strings
#--kernel 2: take in hardware results, run the same kernel logic
from LUCJ_sampler import LUCJ_circuit
from fragment_sqsd_solver import sqsd_fragment
from parallel_circuits_execution import glue_circuits, cut_results
from hardware_simulator import hardware_simulator 
from qiskit import QuantumRegister,ClassicalRegister,QuantumCircuit
qcs = []
rs =[]
dm1s_list = [None for ifrag in range (nfrag)]
dm2_list = [None for ifrag in range (nfrag)]

def get_kernel_fn (ifrag,jobtype ='quantum'):
    print("start get_kernel_fn")
    def kernel (norb, nelec,h0,h1s,h2):
        if jobtype == 'quantum':
            print('glue fragment circuits and send to hardware')
            qc=LUCJ_circuit(num_orbitals[ifrag],num_elec_a[ifrag],num_elec_b[ifrag],spin_sq,h1s[0,:,:],h2)
            qcs.append(qc)
            return None, None,None
        elif jobtype == 'classical':
            print('processing results from hardware')
            results = rs[ifrag]
            print('results',results,flush=True)
            e,d, dm1, dm2 = sqsd_fragment(h1s[0,:,:],h2,num_elec_a[ifrag],num_elec_b[ifrag],num_orbitals[ifrag],spin_sq=spin_sq,iterations=6, n_batches = 15, samples_per_batch =50,max_davidson_cycles=200,results=results)
            etot= np.min(e) + h0
            dm1s = np.zeros((2,norb,norb))
            dm1s[0,:,:] = dm1[0]
            dm1s[1,:,:] = dm1[1]
            dm1s_list[ifrag] = dm1s
            dm2_list[ifrag] = dm2
            save_rdms_to_hdf5(dm1s_list, dm2_list, "RDMS/casdm.h5")
            return etot, dm1s, dm2
    return kernel

def run_hybrid_cycle(mo):
    #------ quantum kernel first
    qcs.clear()
    las.fciboxes = [make_fcibox (las.mol, kernel=get_kernel_fn(ifrag,jobtype = 'quantum')) for ifrag in range(nfrag)]
    las.kernel(mo)
    #------ get results from quantum kernel
    qc = glue_circuits(qcs) #append measurements included
    spin_a_layout= [60,61,62,72,81,82,83,92,102,103]
    spin_b_layout= [58,71,77,78,79,91,98,99,100,101]
    LAY = spin_a_layout + spin_b_layout
    #LAY = [7,8,17,27,28,29,36,48,49,50,5,4,16,23,24,25,35,44,45,46]
    circuits=[qc]
    H = hardware_simulator()
    H.args['device_name'] = 'ibm_sherbrooke'
    H.args['initial_layout'] = LAY
    H.args['dynamical_decoupling'] = True
    H.args['twirling'] = True
    H.args['shots'] = 30000
    H.args['circuits_per_job'] = 100
    #H.submit_jobs(circuits,H.get_service('ibm-q/open/main'))#needs network
    H.submit_jobs(circuits,H.get_service('Qiksit Runtime-lassqd'))
    H.print_arguments() 
    with open("job_id.txt", "r") as f:
        job_id = f.read().strip()
    print('job_id',job_id)
    
    rs.clear()
    #service = QiskitRuntimeService()
    service = QiskitRuntimeService(channel="ibm_cloud", token='', instance='') 
    job = service.job(job_id)
    job_result = job.result()
    pub_result = job_result[0]
    keys = list(pub_result.data.keys())
    for key in keys:
        counts = pub_result.data[key].get_counts() 
        rs.append(counts)
        print(f" > Counts for {key}: {counts}") 
    for rx in rs:
        print(rx)
    #----- then classical kernel
    las.fciboxes = [make_fcibox (las.mol, kernel=get_kernel_fn(ifrag,jobtype = 'classical')) for ifrag in range(nfrag)]
    las.max_cycle_rdmjk=0
    las.max_cycle_macro=1
    las.kernel(mo)
    mo = las.mo_coeff
    return mo

n_cycles = 1
last_mo = np.load('current_orb.npy') 
mo_init = last_mo
#mo_localized
for cycle in range(n_cycles):
    print(f"Starting hybrid macro cycle {cycle}")
    new_mo = run_hybrid_cycle(mo_init)
    mo_init = new_mo
    #save mo_coeff inside the loop
    np.save('current_orb',new_mo)
    if cycle == n_cycles-1:
        from sqdpdft import sqdpdft_energy,load_rdms_from_hdf5
        casdm1frs_sqd,casdm2fr_sqd = load_rdms_from_hdf5('RDMS/casdm.h5',nfrag)
        mo_sqd = np.load('current_orb.npy')
        print('SQDPDFT energy',sqdpdft_energy(new_las,nfrag,norb_cas,ncas_sub,casdm1frs_sqd,casdm2fr_sqd,mo_sqd,'tPBE'))
