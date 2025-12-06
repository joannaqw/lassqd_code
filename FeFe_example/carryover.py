import pickle
import timeit
from pathlib import Path

import numpy as np
from qiskit_addon_dice_solver import solve_fermion
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs
from qiskit_addon_sqd.subsampling import postselect_and_subsample


def concat_count_dicts(cts_list, weight_array):
    counts_dict = {}
    for i, cts in enumerate(cts_list):
        for k in cts.keys():
            if k not in counts_dict.keys():
                counts_dict[k] = cts[k] * weight_array[i]
            elif k in counts_dict.keys():
                counts_dict[k] += cts[k] * weight_array[i]
    return counts_dict


samples_per_batch = 3000
iterations = 10
data_dir = Path(
    f"/disk1/kevinsung@ibm.com/skqd/data_carryover_samples_per_batch-{samples_per_batch}_iterations-{iterations}"
)
data_dir.mkdir(exist_ok=True)

print("\tLoading data...")
# Load Hamiltonian
with open("h1e_NOS.npy", "rb") as f:
    h1e = np.load(f)
with open("h2e_NOS.npy", "rb") as f:
    h2e = np.load(f)

# Load samples
with open(
    "fez_data/28_bath_4_imp_U-10_random_bath_dispersion_backend_ibm_fez_2025-03-27.pickle",
    "rb",
) as input_file:
    data = pickle.load(input_file)

# Convert samples to bitstring array
weight_array = np.ones(len(data["counts"].keys()))
count_dict = concat_count_dicts(
    [data["counts"][key][0] for key in data["counts"].keys()],
    weight_array,
)
bitstring_matrix_full, probs_arr_full = counts_to_arrays(count_dict)
_, n_qubits = bitstring_matrix_full.shape
norb = n_qubits // 2
nelec = (norb // 2, norb // 2)

# Initialize random number generator
rng = np.random.default_rng(193447765250468271707456720300245632067)

# SQD options
carryover_threshold = 1e-7
carryover_strings = np.array([], dtype=np.int64)

# Eigenstate solver options
n_batches = 3
# max_davidson_cycles = 200

# Self-consistent configuration recovery loop
e_hist = np.zeros((iterations, n_batches))  # energy history
# s_hist = np.zeros((iterations, n_batches))  # spin history
occupancy_hist = []
avg_occupancy = None
for i in range(iterations):
    print(f"\tStarting configuration recovery iteration {i}")
    # On the first iteration, we have no orbital occupancy information from the
    # solver, so we begin with the full set of noisy configurations.
    if avg_occupancy is None:
        bs_mat_tmp = bitstring_matrix_full
        probs_arr_tmp = probs_arr_full

    # If we have average orbital occupancy information, we use it to refine the full set of noisy configurations
    else:
        print("\t\tRunning configuration recovery...")
        t0 = timeit.default_timer()
        bs_mat_tmp, probs_arr_tmp = recover_configurations(
            bitstring_matrix_full,
            probs_arr_full,
            avg_occupancy,
            nelec[0],
            nelec[1],
            rand_seed=rng,
        )
        t1 = timeit.default_timer()
        print(f"\t\tConfiguration recovery done in {t1 - t0} seconds.")

    # Create batches of subsamples. We post-select here to remove configurations
    # with incorrect hamming weight during iteration 0, since no config recovery was performed.
    print("\t\tRunning postselect and subsample...")
    t0 = timeit.default_timer()
    batches = postselect_and_subsample(
        bs_mat_tmp,
        probs_arr_tmp,
        hamming_right=nelec[0],
        hamming_left=nelec[1],
        samples_per_batch=samples_per_batch,
        num_batches=n_batches,
        rand_seed=rng,
    )
    t1 = timeit.default_timer()
    print(f"\t\tPostselect and subsample done in {t1 - t0} seconds.")

    # Run eigenstate solvers in a loop. This loop should be parallelized for larger problems.
    e_tmp = np.zeros(n_batches)
    # s_tmp = np.zeros(n_batches)
    occs_tmp = []
    coeffs = []
    print("\t\tRunning SCI...")
    for j in range(n_batches):
        strs_a, strs_b = bitstring_matrix_to_ci_strs(batches[j])
        strs_a = np.union1d(strs_a, carryover_strings)
        strs_b = np.union1d(strs_b, carryover_strings)
        print(f"\t\t\tBatch {j} subspace dimension: {len(strs_a) * len(strs_b)}")
        print(f"\t\t\tBatch {j} running SCI...")
        t0 = timeit.default_timer()
        energy_sci, coeffs_sci, avg_occs = solve_fermion(
            (strs_a, strs_b),
            h1e,
            h2e,
            mpirun_options=["-quiet", "-n", "96"],
        )
        t1 = timeit.default_timer()
        print(f"\t\t\tBatch {j} SCI finished in {t1 - t0} seconds.")
        print(f"\t\t\tBatch {j} SCI energy: {energy_sci}.")
        e_tmp[j] = energy_sci
        # s_tmp[j] = spin
        occs_tmp.append(avg_occs)
        coeffs.append(coeffs_sci)

    print(f"\t\tIteration {i} SCI energies: {e_tmp}")

    lowest_index = np.argmin(e_tmp)
    print(f"\t\tLowest energy for this batch: {e_tmp[lowest_index]}.")
    sci_state = coeffs[lowest_index]
    filepath = data_dir / f"sci_state-{i}.npz"
    sci_state.save(filepath)
    print(f"\t\tSCI state saved to {filepath}.")

    # Carry over bitstrings with large CI weight
    flattened = sci_state.amplitudes.reshape(-1)
    indices = np.argsort(np.abs(flattened))
    index = np.searchsorted(np.abs(flattened), carryover_threshold, sorter=indices)
    carryover_indices = indices[index:]
    alpha_indices, beta_indices = np.divmod(
        carryover_indices, sci_state.amplitudes.shape[1]
    )
    alpha_strings = sci_state.ci_strs_a[alpha_indices]
    beta_strings = sci_state.ci_strs_b[beta_indices]
    carryover_strings = np.union1d(alpha_strings, beta_strings)
    print(f"\t\tCarrying over {len(carryover_strings)} single-spin strings.")

    # Combine batch results
    avg_occupancy = np.mean(occs_tmp, axis=0)

    # Track optimization history
    e_hist[i, :] = e_tmp
    # s_hist[i, :] = s_tmp
    occupancy_hist.append(avg_occupancy)

filepath = data_dir / "energies.npy"
np.save(filepath, e_hist)
print(f"\tEnergies saved to {filepath}.")

filepath = data_dir / "occupancies.npy"
np.save(filepath, np.array(occupancy_hist))
print(f"\tOccupancies saved to {filepath}.")
