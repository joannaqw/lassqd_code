import numpy as np
from pyscf import ao2mo, tools,gto
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs, solve_sci
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left, subsample
#from LUCJ_sampler import LUCJ_Sampler
from scipy import linalg as LA
def sqsd_fragment(
        hcore: np.ndarray, eri: np.ndarray, num_elec_a: int, num_elec_b: int, num_orbitals:int,
        spin_sq: float,iterations: int,n_batches: int,samples_per_batch: int,max_davidson_cycles: int,results:dict,) -> tuple[np.ndarray, np.ndarray, np.ndarray,np.ndarray]:
    '''convert las h1, h2 to mo basis'''
    mol = gto.M()
    mol.nelectron = num_elec_a + num_elec_b
    mol.spin = spin_sq
    mol.nao = num_orbitals
    mf_as = mol.RHF()
    mf_as.get_hcore = lambda *args: hcore
    mf_as.get_ovlp = lambda *args: np.eye(num_orbitals)
    mf_as._eri = eri
    mf_as.kernel() 
    C = mf_as.mo_coeff
    h0e_MO = mf_as.mol.energy_nuc()
    h1e_MO = np.einsum('pi,pr,rj->ij',C,hcore,C,optimize=True)
    h2e_MO = np.einsum('pi,rj,prqs,qk,sl->ijkl',C,C,eri,C,C,optimize=True)

    """Perform SQSD."""
    # Self-consistent configuration recovery loop
    e_hist = np.zeros((iterations, n_batches))  # energy history
    s_hist = np.zeros((iterations, n_batches))  # spin history
    d_hist = np.zeros((iterations, n_batches))  # subspace dimension history
    occupancy_hist = np.zeros((iterations, 2 * num_orbitals))
    avg_occupancies = None  # (spin-up, spin-down) mean orbital occupancies
    rand_seed=None
    open_shell=True
    ''' set up tracker to return the lowest rdm for the global e minimum'''
    lowest_energy_global = float('inf')
    lowest_energy_global_iter = -1
    lowest_energy_global_batch = -1
    dm1_mo_lowest_global = None
    dm2_mo_lowest_global = None
    for i in range(iterations):
        print(f"Starting configuration recovery iteration {i}")
        if avg_occupancies is None:
            counts_dict = results
            bitstring_matrix_full, probs_arr_full = counts_to_arrays(counts_dict)
            bs_mat_tmp = bitstring_matrix_full
            probs_arr_tmp = probs_arr_full
        else:
            bs_mat_tmp, probs_arr_tmp = recover_configurations(bitstring_matrix_full,probs_arr_full,avg_occupancies,num_elec_a,num_elec_b,rand_seed=rand_seed)

        # Throw out samples with incorrect hamming weight and create batches of subsamples.
        bs_mat_postsel, probs_arr_postsel = postselect_by_hamming_right_and_left(bs_mat_tmp,probs_arr_tmp,hamming_right=num_elec_a,hamming_left=num_elec_b)
        batches = subsample(bs_mat_postsel,probs_arr_postsel,samples_per_batch,n_batches,rand_seed=rand_seed)
        # Run eigenstate solvers in a loop. This loop should be parallelized for larger problems.
        int_e = np.zeros(n_batches)
        int_s = np.zeros(n_batches)
        int_d = np.zeros(n_batches)
        int_occs = np.zeros((n_batches, 2 * num_orbitals))
        cs = []
        # Initialize lists to store dm1_MO and dm2_MO for each batch
        dm1_mo_list = []
        dm2_mo_list = []
        # Initialize variable to track the lowest energy and its index WITHIN ONE ITER AND X NUMBER OF BATCHES
        lowest_energy_iter = float('inf')
        lowest_energy_batch_index = -1

        for j in range(n_batches):
            addresses = bitstring_matrix_to_ci_strs(batches[j], open_shell=open_shell)
            int_d[j] = len(addresses[0]) * len(addresses[1])
            result = solve_sci(addresses,h1e_MO,h2e_MO,num_orbitals,(num_elec_a,num_elec_b),spin_sq=spin_sq,max_cycle=max_davidson_cycles,tol=1e-12)
            energy_sci = result.energy
            coeffs_sci = result.sci_state
            avg_occs = result.orbital_occupancies
            spin = result.sci_state.spin_square()
            dm1_MO = result.sci_state.rdm(rank=1, spin_summed=False)
            dm2_MO = result.rdm2
            #energy_sci += e_core 
            #nuclear_repulsion_energy
            print('spin is',spin)
            int_e[j] = energy_sci
            int_s[j] = spin
            int_occs[j, :num_orbitals] = avg_occs[0]
            int_occs[j, num_orbitals:] = avg_occs[1]
            cs.append(coeffs_sci)
            dm1_mo_list.append(dm1_MO)
            dm2_mo_list.append(dm2_MO)
            if energy_sci < lowest_energy_iter:
                lowest_energy_iter = energy_sci
                lowest_energy_batch_index = j
        # Combine batch results
        avg_occupancy = np.mean(int_occs, axis=0)
        # int_occs stores (spin-up | spin-down); recover_configurations wants (up, down).
        avg_occupancies = (avg_occupancy[:num_orbitals], avg_occupancy[num_orbitals:])

        # Track optimization history
        e_hist[i, :] = int_e
        s_hist[i, :] = int_s
        d_hist[i, :] = int_d
        occupancy_hist[i, :] = avg_occupancy
        
        # dm1/dm2 for the lowest energy *in this iteration*
        dm1_mo_lowest_iter = dm1_mo_list[lowest_energy_batch_index]
        dm2_mo_lowest_iter = dm2_mo_list[lowest_energy_batch_index]

        # Compare iteration's best with the global best
        if lowest_energy_iter < lowest_energy_global:
            lowest_energy_global = lowest_energy_iter
            lowest_energy_global_iter = i
            lowest_energy_global_batch = lowest_energy_batch_index
            dm1_mo_lowest_global = dm1_mo_lowest_iter
            dm2_mo_lowest_global = dm2_mo_lowest_iter
        
        '''convert dm1 dm2 back to LAS basis'''
        #print('mo rdm1', dm1_MO)
        D = LA.inv(mf_as.mo_coeff)
        dm1_up, dm1_dn = dm1_mo_lowest_global
        dm1_WO_up = np.einsum('pi,pr,rj->ij', D, dm1_up, D, optimize=True)
        dm1_WO_dn = np.einsum('pi,pr,rj->ij', D, dm1_dn, D, optimize=True)
        dm1_WO = (dm1_WO_up, dm1_WO_dn)
        #dm1_WO = np.einsum('pi,pr,rj->ij',D,dm1_MO,D,optimize=True)
        dm2_WO = np.einsum('pi,rj,prqs,qk,sl->ijkl',D,D,dm2_mo_lowest_global,D,D,optimize=True)
        

    return e_hist.flatten(), d_hist.flatten(), dm1_WO, dm2_WO
