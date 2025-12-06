import numpy as np
import pyscf
from pyscf import ao2mo, tools,gto
from sqsd.configuration_recovery import recover_configurations
from sqsd.qsci import solve_pyscf2
from sqsd.subsampling import postselect_and_subsample
from sqsd.utils import bitstring_matrix_to_sorted_addresses, flip_orbital_occupancies
from sqsd.utils.counts import counts_to_arrays
from sqsd.utils.counts import generate_counts_uniform
from scipy import linalg as LA

''' function defined below for SQD solving LAS fragments, assume bitstrings are already available '''
def SQD_solver(bitstrings:dict,hcore: np.ndarray, eri: np.ndarray, num_elec_a: int, num_elec_b: int, num_orbitals:int,
    spin_sq: float,iterations: int,n_batches: int,samples_per_batch: int,max_davidson_cycles: int,) -> tuple[float,np.ndarray,np.ndarray]:
    # Self-consistent configuration recovery loop
    e_hist = np.zeros((iterations, n_batches))  # energy history
    s_hist = np.zeros((iterations, n_batches))  # spin history
    d_hist = np.zeros((iterations, n_batches))  # subspace dimension history
    occupancy_hist = np.zeros((iterations, 2 * num_orbitals))
    occupancies_bitwise = None  # orbital i corresponds to column i in bitstring matrix
    rand_seed=None
    open_shell=True
    for i in range(iterations):
        if occupancies_bitwise is None:
            counts_dict = bitstrings
            bitstring_matrix_full, probs_arr_full = counts_to_arrays(counts_dict)
            bs_mat_tmp = bitstring_matrix_full
            probs_arr_tmp = probs_arr_full
        else:
            bs_mat_tmp, probs_arr_tmp = recover_configurations(bitstring_matrix_full,probs_arr_full,occupancies_bitwise,num_elec_b,num_elec_a,rand_seed=rand_seed)
        batches = postselect_and_subsample(bs_mat_tmp,probs_arr_tmp,num_elec_b,num_elec_a,samples_per_batch,n_batches,rand_seed=rand_seed)
        int_e = np.zeros(n_batches)
        int_s = np.zeros(n_batches)
        int_d = np.zeros(n_batches)
        int_occs = np.zeros((n_batches, 2 * num_orbitals))
        cs = []
        dm1_mo_list = []
        dm2_mo_list = []
        lowest_energy = float('inf')
        lowest_energy_index = -1

        for j in range(n_batches):
            addresses = bitstring_matrix_to_sorted_addresses(batches[j], open_shell=open_shell)
            int_d[j] = len(addresses[0]) * len(addresses[1])
            addresses = addresses[::-1]
            energy_sci, coeffs_sci, avg_occs, spin, dm1_MO, dm2_MO = solve_pyscf2(addresses,h1mo,h2mo,num_elec_a,num_elec_b,spin_sq=spin_sq,max_davidson=max_davidson_cycles)
            sqdvec = coeffs_sci.reshape(-1)
            sqdvec_sorted = sqdvec[(-np.abs(sqdvec)).argsort()]
            int_e[j] = energy_sci
            int_s[j] = spin
            int_occs[j, :num_orbitals] = avg_occs[0]
            int_occs[j, num_orbitals:] = avg_occs[1]
            cs.append(coeffs_sci)
            dm1_mo_list.append(dm1_MO)
            dm2_mo_list.append(dm2_MO)
            if energy_sci < lowest_energy:
                lowest_energy = energy_sci
                lowest_energy_index = j
        # Combine batch results
        avg_occupancy = np.mean(int_occs, axis=0)
        # The occupancies from the solver should be flipped to match the bits in the bitstring matrix.
        occupancies_bitwise = flip_orbital_occupancies(avg_occupancy)
        # Track optimization history
        e_hist[i, :] = int_e
        s_hist[i, :] = int_s
        d_hist[i, :] = int_d
        occupancy_hist[i, :] = avg_occupancy
        dm1_mo_lowest = dm1_mo_list[lowest_energy_index]
        dm2_mo_lowest = dm2_mo_list[lowest_energy_index]
        '''convert dm1 dm2 back to LAS basis'''
        mol = gto.M()
        mol.nelectron = num_elec_a + nun_elec_b
        mol.spin = spin_sq  
        mol.nao = num_orbitals
        mf_as= mol.ROHF()
        mf_as.get_hcore = lambda *args: hcore
        mf_as.get_ovlp = lambda *args: np.eye(norb)
        mf_as._eri = eri
        #mf_as=mf_as.newton()
        mf_as.kernel()
        C = mf_as.mo_coeff
        D = LA.inv(C)
        dm1_up, dm1_dn = dm1_mo_lowest
        dm1_WO_up = np.einsum('pi,pr,rj->ij', D, dm1_up, D, optimize=True)
        dm1_WO_dn = np.einsum('pi,pr,rj->ij', D, dm1_dn, D, optimize=True)
        dm1_WO = (dm1_WO_up, dm1_WO_dn)
        dm2_WO = np.einsum('pi,rj,prqs,qk,sl->ijkl',D,D,dm2_mo_lowest,D,D,optimize=True)
    return np.min(e_hist.flatten()), dm1_WO, dm2_WO

'''function below constructs LUCJ ansatz'''
def LUCJ_circuit(norb, nelec_a, nelec_b, spin_sub, hcore,eri):
