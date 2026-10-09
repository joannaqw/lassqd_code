# Examples

Both examples treat the FeFe complex in `fefe.xyz` (6-31G, charge +4) as two Fe
fragments with opposite spin polarization, `((4, 2), (2, 4))`, and run hybrid LASSQD
with LUCJ circuits (`lassqd.run_lassqd`). Run each script from its own directory;
see the root [installation instructions](../README.md#installation) for setup.
The [PDFT Slurm script](fefe_lassqd_pdft/run.slurm) shows one cluster's setup;
adapt its account, partition, and environment activation to your system.

Each script defines its own `prepare_fragment(h1, h2, norb, nelec)` and passes it
as the required `circuit_builder` argument. These examples initialize one LUCJ layer
by compressed factorization of CCSD amplitudes, with nearest-neighbor same-spin
interactions and opposite-spin pairs `(p, p)` for `p = 0, 4, ...`. Edit the
interaction pairs, `n_reps`, and compression settings in the script for your
calculation. The library imposes no ansatz or connectivity. The builder must
prepare occupations in the fragment ROHF basis used by `FragmentSQD`, with
alpha orbitals on the first `norb` qubits and beta on the next `norb`, and return
the circuit without measurements.

## fefe_lassqd

[lassqd_fefe.py](fefe_lassqd/lassqd_fefe.py): (6e,10o) per fragment (20 qubits each),
circuits sampled classically with `FfsimSampler` (100k shots per fragment), up to
50 hybrid cycles with determinant carryover between cycles. `glue_circuits=False`
submits the two native LUCJ circuits in one sampler job without transpilation.
Its `solve_sci_batch` function solves 10 Davidson roots per batch and retains the
first, with the fragment's minimum-spin constraint, 200 maximum cycles and `1e-16`
tolerance. It is passed to `FragmentSQD` through the `sci_solver` option.

- Inputs: only `fefe.xyz`. The script runs ROHF and an Fe 3d AVAS guess to build
  the initial orbitals.
- Outputs: `current_orb.npy` and per-fragment SQD states, histories, and carryover
  strings in `data_carryover_frag{0,1}/`, overwritten after each completed hybrid
  cycle. The callbacks in [lassqd_fefe.py](fefe_lassqd/lassqd_fefe.py) collect batch
  summaries and save the final fragment states, then clear the history buffers.
  Histories include energies, spin, subspace dimensions, alpha/beta string counts,
  and the lowest-energy batch's occupancies per round.

## fefe_lassqd_pdft

[lassqd_fefe_ibm.py](fefe_lassqd_pdft/lassqd_fefe_ibm.py): (6e,5o) per fragment from
AVAS Fe 3d orbitals, sampled on IBM Quantum hardware, then the SQD-PDFT (tPBE)
energy. The script is configured for `ibm_sherbrooke`, 30k shots, and a fixed qubit
layout; select a backend available to your account and adjust the layout before
running. Install the `ibm` extra and save an IBM Quantum account with
`QiskitRuntimeService.save_account(...)` first.

The script currently runs **one hybrid cycle** (`max_cycles=1`), restarting from
`current_orb.npy`. Increase `max_cycles` to allow convergence over multiple cycles.
To start from the localized AVAS guess, set `mo_init = mo_localized`.
Each cycle writes `current_orb.npy` and the LAS wave function (orbitals + fragment
RDMs) to `RDMS/casdm.h5`.
Its `sci_solver` is the addon's `solve_sci_batch`, configured with a triplet spin
target (`spin_sq=2`), 200 maximum Davidson cycles and `1e-12` tolerance.

[pdft_from_rdms.py](fefe_lassqd_pdft/pdft_from_rdms.py) (what `run.slurm` runs): the
SQD-PDFT energy of the wave function saved in `RDMS/casdm.h5`, without new sampling.

The committed `RDMS/casdm.h5` and `current_orb.npy` come from the earlier version of
this workflow. That version saved the RDMs *before* LASSCF's orbital step and
canonicalization, but saved the orbitals *after*, so the two are not an exactly
matched pair. Rerun `lassqd_fefe_ibm.py` to get a consistent pair; the new file
stores the orbitals next to the RDMs, and `pdft_from_rdms.py` uses them.
