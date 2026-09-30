# Examples

Both examples treat the FeFe complex in `fefe.xyz` (6-31G, charge +4) as two Fe
fragments with opposite spin polarization, `((4,2),(2,4))`, and run hybrid LASSQD
with LUCJ circuits (`lassqd.run_lassqd`). Run each script from its own directory;
`run.slurm` shows the cluster setup.

## `fefe_lassqd/`

`lassqd_fefe.py`: (6e,10o) per fragment (20 qubits each), circuits sampled classically
with `FfsimSampler` (100k shots per fragment), up to 50 hybrid cycles with determinant
carryover between cycles. `glue_circuits=False` submits the two native LUCJ
circuits in one sampler job without transpilation.
Its custom `sci_solver.py` solves 10 Davidson roots per batch and retains the first,
with the fragment's minimum-spin constraint, 200 maximum cycles and `1e-16`
tolerance. It is passed to `FragmentSQD` through the `sci_solver` option.

- Inputs: `fefe_as.npy` and `as_increase_avas.npy` (initial orbitals). **These are
  not in the repository**; copy them into this directory first.
- Outputs: `current_orb.npy` after every cycle, and per-fragment SQD histories in
  `data_carryover_frag{0,1}/`.

## `fefe_lassqd_pdft/`

`lassqd_fefe_ibm.py`: (6e,5o) per fragment from AVAS Fe 3d orbitals, sampled on IBM
Quantum hardware (`ibm_sherbrooke`, 30k shots, fixed qubit layout), then the SQD-PDFT
(tPBE) energy. It restarts from `current_orb.npy`. Save an IBM Quantum account with
`QiskitRuntimeService.save_account(...)` first. Each cycle writes `current_orb.npy`
and the LAS wave function (orbitals + fragment RDMs) to `RDMS/casdm.h5`.
Its `sci_solver` is the addon's `solve_sci_batch`, configured with a triplet spin
target (`spin_sq=2`), 200 maximum Davidson cycles and `1e-12` tolerance.

`pdft_from_rdms.py` (what `run.slurm` runs): the SQD-PDFT energy of the wave
function saved in `RDMS/casdm.h5`, without new sampling.

The committed `RDMS/casdm.h5` and `current_orb.npy` come from the earlier version of
this workflow. That version saved the RDMs *before* LASSCF's orbital step and
canonicalization, but saved the orbitals *after*, so the two are not an exactly
matched pair. Rerun `lassqd_fefe_ibm.py` to get a consistent pair; the new file
stores the orbitals next to the RDMs, and `pdft_from_rdms.py` uses them.
