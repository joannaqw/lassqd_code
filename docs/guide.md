# User guide

LASSQD is organized into modules that pass records from stage to stage:

| Module | Contents |
| --- | --- |
| {mod}`lassqd.records` | `LASState`, `Fragment`, `Embedding`, `FragmentSamples`, `SQDState`, `FragmentResult`: the data passed between stages, with HDF5 `save`/`load` |
| {mod}`lassqd.las` | `LASSCF`, `initial_state`, `capture_embedding`, `replay_orbital_step`: the only interface to mrh |
| {mod}`lassqd.circuits` | `FragmentCircuits`, `build_circuits`, `submit_circuits`, `collect_samples` |
| {mod}`lassqd.sqd` | `FragmentSQD`: configuration recovery, carryover and the fragment solve |
| {mod}`lassqd.hybrid` | `solve_fragments`, `update_orbitals`, and the serial driver `run_lassqd` |
| {mod}`lassqd.fragment` | `fragment_rohf` and `sampling_basis_rohf`: the fragment ROHF basis |
| {mod}`lassqd.pdft` | `lassqd_pdft_energy`: LAS-PDFT on a LAS state |

`lassqd.records` imports only NumPy and h5py, so stored records can be read
without mrh, PySCF or Qiskit installed. `FragmentCircuits` is saved and loaded
the same way, with the circuits in QPY; it needs Qiskit, but not mrh, PySCF or
qiskit-addon-sqd.

## Usage

The sketch below assumes that `mf` is a prepared PySCF mean-field object,
`mo_coeff` contains localized orbitals for two (6e,5o) fragments, and
`prepare_fragment` is a circuit builder as in the [tutorial](lassqd.ipynb)
or the [FeFe example](https://github.com/joannaqw/lassqd_code/blob/main/examples/fefe_lassqd/lassqd_fefe.py). Build the LAS object
with `lassqd.LASSCF`, mrh's RDM-based LASSCF factory: it uses density fitting
when `mf` does, so its active-space integrals and potentials agree. LASSQD
rejects an mrh `LASSCFNoSymm` constructed directly from a density-fitted mean
field, which would mix exact and density-fitted integrals.

```python
from functools import partial

import ffsim
from qiskit_addon_sqd.fermion import solve_sci_batch

from lassqd import LASSCF, FragmentSQD, initial_state, lassqd_pdft_energy, run_lassqd

las = LASSCF(mf, (5, 5), ((4, 2), (2, 4)), spin_sub=(3, 3))
solvers = [
    FragmentSQD(
        50,
        max_iterations=6,
        num_batches=15,
        sci_solver=partial(solve_sci_batch, spin_sq=2.0, max_cycle=200, tol=1e-12),
        seed=seed,
    )
    for seed in (1, 2)
]
result = run_lassqd(
    las,
    initial_state(las, mo_coeff),
    solvers,
    ffsim.qiskit.FfsimSampler(seed=0),
    circuit_builder=prepare_fragment,
    groups=[[0], [1]],
    shots=100_000,
)
e_pdft, e_ot, e_states = lassqd_pdft_energy(las, result.state)
```

`result.converged` means only that the absolute energy change between
consecutive cycles fell below `conv_tol` (default `1e-5`) within `max_cycles`
(default 50). `result.state` is the final LAS wave function and
`result.results` holds the last cycle's fragment results.

## Circuits

`circuit_builder(fragment)` receives a `Fragment` and returns an unmeasured
circuit with `2 * fragment.norb` qubits. It prepares a state in the fragment's
sampling basis, its ROHF orbitals, whose integrals are `fragment.h1` and
`fragment.h2` (chemists' notation). Alpha orbitals occupy qubits `0..norb-1` and
beta orbitals `norb..2*norb-1`. There is no default ansatz; the examples use
CCSD-initialized LUCJ circuits, starting CCSD from `fragment.rohf()`, the ROHF
solution in the sampling basis as a PySCF object.

`groups` lists the fragments of each circuit. A group of several fragments is
glued into one circuit with one classical register per fragment; `None`, the
default, glues all fragments. For native ffsim gates sampled with
`FfsimSampler`, use one fragment per group (`groups=[[0], [1]]`) and no pass
manager. Aer and IBM Runtime need a `pass_manager` that compiles ffsim gates:
create it with `generate_preset_pass_manager(...)`, then set
`pass_manager.pre_init = ffsim.qiskit.PRE_INIT`. `shots` is the number of
samples per fragment per cycle.

## Running the stages separately

`run_lassqd` composes stage functions that map records to records. Call them
directly to store each stage's output, re-solve saved samples with other SQD
settings, sample groups in separate jobs, or resume after a failure:

```python
from lassqd import (
    build_circuits,
    capture_embedding,
    collect_samples,
    solve_fragments,
    submit_circuits,
    update_orbitals,
)

state = initial_state(las, mo_coeff)
for cycle in range(max_cycles):
    embedding = capture_embedding(las, state)  # fragment Hamiltonians and bases
    circuits = build_circuits(embedding, prepare_fragment, groups=[[0], [1]])
    circuits.save(f"circuits_{cycle}.h5")
    job = submit_circuits(sampler, circuits, shots=100_000)
    samples = collect_samples(circuits, job.result(), job_ids=job.job_id())
    samples.save(f"samples_{cycle}.h5")

    results = solve_fragments(samples, solvers)  # SQD, outside mrh
    for i, solver in enumerate(solvers):
        solver.state.save(f"sqd_state_{cycle}_{i}.h5")
    step = update_orbitals(las, embedding, results)  # one LASSCF orbital step
    state = step.state
    state.save(f"state_{cycle}.h5")
```

- `capture_embedding` enters mrh with kernels that record each fragment's
  Hamiltonian at the state's orbitals and starting RDMs, then stops. The state's
  RDMs set each fragment's embedding in the others; `run_lassqd` continues from
  the previous cycle's RDMs, as classical LASSCF does.
- `collect_samples` accepts the pub results of any job that ran the circuits,
  including one recovered later by job ID, with the circuits from
  `FragmentCircuits.load(...)`, or results combined from several jobs with
  `submit_circuits(..., indices=...)`. Save compiled circuits: QPY loads native
  ffsim gates as opaque gates, which `FfsimSampler` cannot run.
- `update_orbitals` re-enters mrh at the same state with kernels that check each
  Hamiltonian against the captured one and return the stored fragment RDMs. mrh
  then computes the LAS energy, takes one orbital step and canonicalizes.

To resume, load the records into a new LAS object built the same way: pass
`LASState.load(...)` to `capture_embedding` or `run_lassqd`, assign
`solver.state = SQDState.load(...)`, and solve `FragmentSamples.load(...)` again
if the failure came after sampling. A resumed cycle reproduces the
uninterrupted one bit for bit.

## Fragment SQD

`FragmentSQD(samples_per_batch, *, seed, sci_solver, callback, **options)`
solves a fragment with
`qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`, whose names and
defaults the numerical `options` use. It runs up to `max_iterations` rounds of
configuration recovery, using the best batch's occupancies, and returns the
energy and RDMs of the lowest-energy batch across all rounds. Circuits and SQD
use only the alpha one-electron integrals, also when the embedding is
spin-dependent.

- **State.** `solver.state` is an `SQDState`: the random generator and the
  carryover strings with their orbital basis. `solve` replaces it only when a
  solve succeeds; assign an earlier state to repeat a solve.
- **Carryover.** Determinants with amplitude above `carryover_threshold` are
  included in every later round and solve, mapped onto the new sampling basis by
  comparing orbitals in the AO basis. Set `carryover_threshold=None` to disable
  it.
- **Batch solver.** `sci_solver(ci_strings, h1, h2, norb, nelec)` returns a
  list of `SCIResult`; it controls spin constraints and Davidson settings. The
  default is the addon's `solve_sci_batch` without a spin penalty. The
  [tutorial](lassqd.ipynb) uses a Fulqrum solver.
- **Callback.** `callback(results)` receives each round's `SCIResult` list,
  with energies excluding `h0` and states in the sampling basis, for example to
  record diagnostics as in the [FeFe example](https://github.com/joannaqw/lassqd_code/blob/main/examples/fefe_lassqd/lassqd_fefe.py).
- **Collective solves.** `prepare`, `SQDInputs.run` and `finalize` split a solve
  into rank-0 setup, the SQD call every MPI rank enters, and rank-0 finishing.

**Random streams.** Give the sampler and each solver an independent stream,
for example `sampler_seed, *solver_seeds = np.random.default_rng(seed).spawn(nfrags + 1)`.
A generator passed as `seed` spawns a child stream, so solvers given the same
generator still draw independently, and each solver's stream continues across
cycles in `solver.state`.
