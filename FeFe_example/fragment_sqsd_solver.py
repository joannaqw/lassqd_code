import numpy as np
from pathlib import Path
from pyscf import gto
from sqsd.configuration_recovery import recover_configurations
from sqsd.qsci import solve_pyscf2
from sqsd.subsampling import postselect_and_subsample
from sqsd.utils import bitstring_matrix_to_sorted_addresses, flip_orbital_occupancies
from sqsd.utils.counts import counts_to_arrays

# from LUCJ_sampler import LUCJ_Sampler
from scipy import linalg as LA


def sqsd_fragment(
    hcore: np.ndarray,
    eri: np.ndarray,
    num_elec_a: int,
    num_elec_b: int,
    num_orbitals: int,
    spin_sq: float,
    iterations: int,
    n_batches: int,
    samples_per_batch: int,
    max_davidson_cycles: int,
    results: dict,
    counter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """convert las h1, h2 to mo basis"""
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
    mf_as.mol.energy_nuc()
    h1e_MO = np.einsum("pi,pr,rj->ij", C, hcore, C, optimize=True)
    h2e_MO = np.einsum("pi,rj,prqs,qk,sl->ijkl", C, C, eri, C, C, optimize=True)

    """ carry over parts"""
    # if counter is even:0,2,4..--> frag 0, if counter is odd: 1,3,5..--> frag 1
    print("counter", counter)
    if counter % 2 == 0:
        data_dir = (
            Path.cwd()
            / f"data_carryover_frag0_samples_per_batch-{samples_per_batch}_iterations-{iterations}"
        )
        data_dir.mkdir(exist_ok=True)
        if counter > 0:
            prev_mo = np.load(f"frag0_mo_{counter - 2}.npy")
    else:
        data_dir = (
            Path.cwd()
            / f"data_carryover_frag1_samples_per_batch-{samples_per_batch}_iterations-{iterations}"
        )
        data_dir.mkdir(exist_ok=True)
        if counter > 1:
            prev_mo = np.load(f"frag1_mo_{counter - 2}.npy")
    """carryover pt2"""
    carryover_threshold = 1e-3
    carryover_strings = [[], []]
    open_shell = True

    def bitstrings_from_indices(indices, norb):
        """
        Convert integer determinant indices to boolean occupation bitstrings.
        Each row is a determinant with `norb` spin orbitals.
        True = occupied (1), False = unoccupied (0).
        """
        return ((indices[:, None] >> np.arange(norb)[::-1]) & 1).astype(bool)

    if counter != 0 and counter != 1:
        # 1. calculate overlap
        mol_MO = gto.M()
        mol_MO.nelectron = num_elec_a + num_elec_b
        mol_MO.spin = np.abs(num_elec_a - num_elec_b)
        mol_MO.nao = num_orbitals
        mf_as_MO = mol_MO.ROHF()
        mf_as_MO.get_hcore = lambda *args: h1e_MO
        mf_as_MO.get_ovlp = lambda *args: np.eye(num_orbitals)
        mf_as_MO._eri = h2e_MO
        mf_as_MO.kernel()
        norb = num_orbitals
        S = mf_as_MO.get_ovlp()
        S_ovlp = mf_as_MO.mo_coeff.T @ S @ prev_mo
        # 2. load saved strings and convert to bitstring_matrix
        carryover_strings[0] = np.load(data_dir / "alpha_strings.npy")
        carryover_strings[1] = np.load(data_dir / "beta_strings.npy")
        alpha_bitstrings = bitstrings_from_indices(carryover_strings[0], norb)
        beta_bitstrings = bitstrings_from_indices(carryover_strings[1], norb)
        # 3. permute based on overlap and hstack for full bistring_matrix
        # ??? mo_coeff order v. bitstring order(right to left)
        rho = []
        for p in range(norb):
            rho.append(np.argmax(np.abs(S_ovlp[p, :])))
        np.array(rho)[::-1]
        alpha_rotated = alpha_bitstrings[:, ::-1][:, rho][:, ::-1]
        beta_rotated = beta_bitstrings[:, ::-1][:, rho][:, ::-1]
        full_spinorb_bits = np.hstack([beta_rotated, alpha_rotated])
        # 4. convert back to strings
        fullco = bitstring_matrix_to_sorted_addresses(
            full_spinorb_bits, open_shell=open_shell
        )
        carryover_strings[0] = fullco[1]
        carryover_strings[1] = fullco[0]

    """Perform SQSD."""
    # Self-consistent configuration recovery loop
    e_hist = np.zeros((iterations, n_batches))  # energy history
    s_hist = np.zeros((iterations, n_batches))  # spin history
    d_hist = np.zeros((iterations, n_batches))  # subspace dimension history
    a_hist = np.zeros((iterations, n_batches))
    b_hist = np.zeros((iterations, n_batches))
    occupancy_hist = np.zeros((iterations, 2 * num_orbitals))
    occupancies_bitwise = None  # orbital i corresponds to column i in bitstring matrix
    rand_seed = None

    for i in range(iterations):
        print(f"Starting configuration recovery iteration {i}")
        if occupancies_bitwise is None:
            counts_dict = results
            bitstring_matrix_full, probs_arr_full = counts_to_arrays(counts_dict)
            bs_mat_tmp = bitstring_matrix_full
            probs_arr_tmp = probs_arr_full
        else:
            bs_mat_tmp, probs_arr_tmp = recover_configurations(
                bitstring_matrix_full,
                probs_arr_full,
                occupancies_bitwise,
                num_elec_b,
                num_elec_a,
                rand_seed=rand_seed,
            )

        # Throw out samples with incorrect hamming weight and create batches of subsamples.
        batches = postselect_and_subsample(
            bs_mat_tmp,
            probs_arr_tmp,
            num_elec_b,
            num_elec_a,
            samples_per_batch,
            n_batches,
            rand_seed=rand_seed,
        )
        # Run eigenstate solvers in a loop. This loop should be parallelized for larger problems.
        int_e = np.zeros(n_batches)
        int_s = np.zeros(n_batches)
        int_d = np.zeros(n_batches)
        int_a = np.zeros(n_batches)
        int_b = np.zeros(n_batches)
        int_occs = np.zeros((n_batches, 2 * num_orbitals))
        cs = []
        # Initialize lists to store dm1_MO and dm2_MO for each batch
        dm1_mo_list = []
        dm2_mo_list = []

        # Initialize variable to track the lowest energy and its index
        lowest_energy = float("inf")
        lowest_energy_index = -1
        for j in range(n_batches):
            print("current batch,", j)
            addresses = bitstring_matrix_to_sorted_addresses(
                batches[j], open_shell=open_shell
            )
            addresses = addresses[::-1]
            print("addresses before", addresses)
            addresses_a, addresses_b = addresses
            addresses_a = set(addresses_a)
            addresses_b = set(addresses_b)
            addresses_a.update(carryover_strings[0])
            addresses_b.update(carryover_strings[1])
            addresses_a = sorted(addresses_a)
            addresses_b = sorted(addresses_b)
            addresses = (addresses_a, addresses_b)
            int_d[j] = len(addresses[0]) * len(addresses[1])
            int_a[j] = len(addresses[0])
            int_b[j] = len(addresses[1])
            print("after d", len(addresses[0]), len(addresses[1]))
            energy_sci, coeffs_sci, avg_occs, spin, dm1_MO, dm2_MO = solve_pyscf2(
                addresses,
                h1e_MO,
                h2e_MO,
                num_elec_a,
                num_elec_b,
                spin_sq=spin_sq,
                max_davidson=max_davidson_cycles,
                tol=1e-16,
                nroots=10,
            )
            # energy_sci += e_core
            # nuclear_repulsion_energy
            int_e[j] = energy_sci
            int_s[j] = spin
            print("SQD energy,", energy_sci)
            print("spin", spin)
            int_occs[j, :num_orbitals] = avg_occs[1]
            int_occs[j, num_orbitals:] = avg_occs[0]
            cs.append(coeffs_sci)
            dm1_mo_list.append(dm1_MO)
            dm2_mo_list.append(dm2_MO)
            if energy_sci < lowest_energy:
                lowest_energy = energy_sci
                lowest_energy_index = j
        # Combine batch results
        avg_occupancy = np.mean(int_occs, axis=0)
        # The occupancies from the solver should be flipped to match the bits in the bitstring matrix.
        print(f"\t\tIteration {i} SCI energies: {int_e}")
        lowest_index = np.argmin(int_e)
        print(f"\t\tLowest energy for this batch: {int_e[lowest_index]}.")
        sci_vec = cs[lowest_index]
        filepath = data_dir / "sci_vec"
        np.save(filepath, sci_vec)
        print(f"\t\tSCI vec saved to {filepath}.")
        flattened = np.array(sci_vec).reshape(-1)
        indices = np.argsort(np.abs(flattened))
        index = np.searchsorted(np.abs(flattened), carryover_threshold, sorter=indices)
        carryover_indices = indices[index:]
        alpha_indices, beta_indices = np.divmod(
            carryover_indices, np.array(sci_vec).shape[1]
        )
        alpha_strings = sci_vec._strs[0][alpha_indices]
        beta_strings = sci_vec._strs[1][beta_indices]
        carryover_strings[0] = alpha_strings
        carryover_strings[1] = beta_strings
        print(
            f"\t\tCarrying over alpha {len(carryover_strings[0])} beta {len(carryover_strings[1])} strings."
        )
        print(
            f"\t\tCarrying over unique alpha {len(np.unique(carryover_strings[0]))} beta {len(np.unique(carryover_strings[1]))} strings."
        )

        occupancies_bitwise = flip_orbital_occupancies(avg_occupancy)
        # Track optimization history
        e_hist[i, :] = int_e
        s_hist[i, :] = int_s
        d_hist[i, :] = int_d
        a_hist[i, :] = int_a
        b_hist[i, :] = int_b
        occupancy_hist[i, :] = avg_occupancy
    np.save(data_dir / "alpha_strings", carryover_strings[0])
    np.save(data_dir / "beta_strings", carryover_strings[1])
    dm1_mo_lowest = dm1_mo_list[lowest_energy_index]
    dm2_mo_lowest = dm2_mo_list[lowest_energy_index]

    """convert dm1 dm2 back to LAS basis"""
    # print('mo rdm1', dm1_MO)
    D = LA.inv(mf_as.mo_coeff)
    dm1_up, dm1_dn = dm1_mo_lowest
    dm1_WO_up = np.einsum("pi,pr,rj->ij", D, dm1_up, D, optimize=True)
    dm1_WO_dn = np.einsum("pi,pr,rj->ij", D, dm1_dn, D, optimize=True)
    dm1_WO = (dm1_WO_up, dm1_WO_dn)
    # dm1_WO = np.einsum('pi,pr,rj->ij',D,dm1_MO,D,optimize=True)
    dm2_WO = np.einsum(
        "pi,rj,prqs,qk,sl->ijkl", D, D, dm2_mo_lowest, D, D, optimize=True
    )
    np.save(data_dir / "d_hist", d_hist)
    np.save(data_dir / "a_hist", a_hist)
    np.save(data_dir / "b_hist", b_hist)

    return e_hist.flatten(), d_hist.flatten(), dm1_WO, dm2_WO
