# Examples

The two FeFe examples treat the complex in `fefe.xyz` (6-31G, charge +4) as two Fe
fragments with opposite spin polarization, `((4, 2), (2, 4))`, and run hybrid LASSQD
with LUCJ circuits (`lassqd.run_lassqd`) in a single process. Run each script from
its own directory;
see the [installation instructions](https://joannaqw.github.io/lassqd_code/installation.html) for setup.
The [PDFT Slurm script](fefe_lassqd_pdft/run.slurm) shows one cluster's setup;
adapt its account, partition, and environment activation to your system.

The third example, [hchain_lassqd_prefect](hchain_lassqd_prefect/README.md), takes
the same kind of calculation apart instead: it drives the library's stage
functions as a Prefect flow whose stages exchange only HDF5 records, with QCSC
launching one Slurm job per stage. It is a standalone project with its own
`pyproject.toml` and `uv.lock`, and needs Python 3.11 or newer.

Each script defines its own `prepare_fragment(fragment)` and passes it as the
required `circuit_builder` argument. These examples initialize one LUCJ layer
by compressed factorization of CCSD amplitudes, with nearest-neighbor same-spin
interactions and opposite-spin pairs `(p, p)` for `p = 0, 4, ...`. Edit the
interaction pairs, `n_reps`, and compression settings in the script for your
calculation. The library imposes no ansatz or connectivity. The builder prepares
a state in the fragment's sampling basis from `fragment.h1` and `fragment.h2`,
with alpha orbitals on the first `norb` qubits and beta on the next `norb`, and
returns the circuit without measurements.

## fefe_lassqd

[lassqd_fefe.py](fefe_lassqd/lassqd_fefe.py): (6e,10o) per fragment (20 qubits each),
circuits sampled classically with `FfsimSampler` (100k shots per fragment), up to
50 hybrid cycles with determinant carryover between cycles. `groups=[[0], [1]]`
submits the two native LUCJ circuits separately in one sampler job without
transpilation. Its `solve_sci_batch` function solves 10 Davidson roots per batch
and retains the first, with the fragment's minimum-spin constraint, 200 maximum
cycles and `1e-16` tolerance. It is passed to `FragmentSQD` as `sci_solver`.

- Inputs: only `fefe.xyz`. The script runs ROHF and an Fe 3d AVAS guess to build
  the initial orbitals.
- Outputs, overwritten after each completed hybrid cycle:
  - `state.h5`: the LAS wave function, a `lassqd.LASState`.
  - `data_carryover_frag{0,1}/sqd_state.h5`: each fragment's solver state, a
    `lassqd.SQDState` with its random generator and carryover strings.
  - `data_carryover_frag{0,1}/sci_vec`: the selected SCI state, in the fragment's
    sampling basis.
  - `data_carryover_frag{0,1}/*_hist.npy`: per-round batch summaries collected
    by the SQD callback (energies, spin, subspace dimensions, alpha/beta string
    counts, and the lowest-energy batch's occupancies).

## fefe_lassqd_pdft

[lassqd_fefe_ibm.py](fefe_lassqd_pdft/lassqd_fefe_ibm.py): (6e,5o) per fragment from
AVAS Fe 3d orbitals, sampled on IBM Quantum hardware, then the SQD-PDFT (tPBE)
energy. The script is configured for `ibm_sherbrooke`, 30k shots, and a fixed qubit
layout; select a backend available to your account and adjust the layout before
running. Install the `ibm` extra and save an IBM Quantum account with
`QiskitRuntimeService.save_account(...)` first.

The script runs **one hybrid cycle** (`max_cycles=1`) per invocation; increase
`max_cycles` to allow convergence over multiple cycles. Each cycle writes a
checkpoint, the LAS wave function as `casdm.h5` and each fragment's solver state
as `sqd_state{0,1}.h5`, into a new directory under `RDMS/`, and then switches the
symlink `RDMS/latest` to it, so an interrupted write never replaces the last
whole checkpoint. When `RDMS/latest` exists, the next run continues from it;
otherwise it starts from the localized AVAS guess.
Its `sci_solver` is the addon's `solve_sci_batch`, configured with a triplet spin
target (`spin_sq=2`), 200 maximum Davidson cycles and `1e-12` tolerance.

[pdft_from_rdms.py](fefe_lassqd_pdft/pdft_from_rdms.py) (what `run.slurm` runs): the
SQD-PDFT energy of the wave function saved in `RDMS/latest/casdm.h5`, without new
sampling.
Run `lassqd_fefe_ibm.py` first to create that file.

## hchain_lassqd_prefect

[hchain_lassqd_prefect/](hchain_lassqd_prefect/README.md): the H20 chain of the
[tutorial](../docs/lassqd.ipynb) — two H10 (10e,10o) singlets in STO-6G — run as a
distributed workflow rather than a script. Each macrocycle is five stages
(capture, prepare circuits, sample, one solve per fragment, replay and commit)
that communicate only through records, so a stage can be rerun from stored
inputs and the fragment solves run concurrently in their own Slurm jobs. Each
solve uses one node; a collective MPI solve across several nodes per fragment is
not implemented, and the README says where it would attach. Sampling runs in the
flow process through QCSC's Qiskit Runtime sampler, Aer, or `FfsimSampler`.

`serial.py` runs the same stages in one process, or as subprocesses, as the
bitwise reference for the orchestrated run, and `compare_runs.py` reports the
first divergence between two run directories.

- Inputs: none beyond the code. `settings.py` holds the geometry, active spaces,
  SQD settings and the random-stream contract; `blocks.scc.toml` holds one
  cluster's Slurm configuration; the `params.*.json` files select the sampler,
  the compilation target and the execution target.
- Outputs, under the run directory, one subdirectory per macrocycle:
  - `state.h5`, `next_state.h5`: the LAS wave function entering and leaving the
    macrocycle, each a `lassqd.LASState`.
  - `embedding.h5`, `circuits.h5`, `samples.h5`: the captured fragment problems,
    the compiled circuits and the measured counts.
  - `result_f{i}.h5`, `sqd_state_f{i}.h5`, `solve_f{i}.json`: each fragment's
    `lassqd.FragmentResult`, its `lassqd.SQDState` and what its solve observed.
  - `cycle.json`, `energies.csv`, `summary.json`: the committed energy of each
    macrocycle and the run's outcome.
  - `jobs/<stage>/`: each logical job's rendered script, its logs and its result
    manifest, which lists every record it produced with that record's size and
    SHA-256.
