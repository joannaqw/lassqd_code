# Distributed LASSQD with Prefect and QCSC on Slurm

This example runs the H₂₀ chain of [the LASSQD tutorial](../../docs/lassqd.ipynb) —
two H₁₀ fragments, ten electrons in ten spatial orbitals each, STO-6G — as a
**Prefect flow of one task per stage**, with HDF5 records as the only handoff
between stages and [QCSC Prefect](https://pypi.org/project/qcsc-prefect-executor/)
launching the Slurm jobs.

The two existing examples are single-process scripts that call `run_lassqd`.
This one takes the same calculation apart: each macrocycle is five stages that
communicate only through files, so the orchestration, restart and concurrency
story is runnable rather than described.

```
capture ──▶ prepare ──▶ sample ──┬──▶ solve f00 ──┬──▶ replay ──▶ next macrocycle
  (job)      (job)    (in-flow)  └──▶ solve f01 ──┘    (job)
```

## What this is, and what it is not

**It is** the whole task graph with one node per fragment solve (`P_f = 1`):
one Slurm job per stage, one task per fragment solve, records as the stage
boundary, a result manifest per job, block-driven resource configuration, and a
reference implementation that the orchestrated run reproduces bitwise.

**It is not** a collective MPI fragment solve across `P_f > 1` nodes. Writing
one is a numerical-equivalence project of its own: a collective `sci_solver`
must select exactly what `TrimPolicy.refine` would select using SQD's *private*
`_select_carryover`, pin the SQD revision, and abort the MPI world on any rank's
failure. With `P_f = 1` the already-implemented `FragmentSQD.prepare` /
`SQDInputs.run` / `finalize` split **is** the per-fragment worker, and
[`solve_worker.py`](solve_worker.py) is written in that three-call form from the
start so the collective adapter attaches at one line.

Everything the multi-node case would need from the orchestration is therefore
already wired, at degenerate values — `num_nodes=1`, `mpiprocs=1`,
`launcher="single"` — so raising them is a configuration change, not a rewrite.
See [Not implemented](#not-implemented-and-where-it-attaches) for every other
simplification and where each one would attach.

## Setup

This is a **standalone project** with its own `pyproject.toml` and `uv.lock`.
The repository's root project, root lock file and CI are untouched: QCSC needs
Python 3.11 or newer, while the root declares `>=3.10` and its CI tests 3.10 and
3.14.

From this directory, after the [root README](../../README.md)'s `.deps/mrh`
checkout exists:

```sh
uv sync --python 3.12 --python-preference only-managed --group fulqrum
```

Notes on that command:

- `--python-preference only-managed` is needed wherever the system Python has no
  development headers: Fulqrum builds C++ extensions and needs `Python.h`, which
  a uv-managed interpreter ships and a bare `python3.12` package often does not.
- `--group fulqrum` installs the Fulqrum batch solver, which is what produced
  the energies below. It is source-only, builds Cython and C++ extensions, and
  the repository's CI excludes it; `--sci-solver pyscf` uses the SQD addon's
  `solve_sci_batch` instead and needs none of it.
- The sync rebuilds mrh and pyscf-forge natively, which takes a few minutes.
  Those builds need a Fortran compiler and BLAS/LAPACK; see the root
  [installation guide](../../docs/installation.md). On a cluster without them in
  the usual places, extract them into a user prefix and put the driver on `PATH`
  before syncing — nothing in this example depends on where they came from.
- `[tool.uv.sources]` and `[tool.uv.extra-build-dependencies]` here repeat the
  root `pyproject.toml`'s, because uv reads `[tool.uv]` from the project being
  resolved and not from a path dependency. **They must be kept in step with the
  root's.** The nested `uv.lock` is not verified by the root CI.

One environment serves both the flow process and the workers, so the
`executable_map` Python and the flow's interpreter are the same path.

Prefect needs a server and a database. Every command below uses the explicit
`.venv/bin/` prefix rather than activating the environment, because that is the
same interpreter path the execution profiles name.

```sh
# Prefect 3 keeps its state in SQLite. On a GPFS home this fails with
# "database is locked" and the ephemeral API times out; use local disk.
export PREFECT_HOME=/var/tmp/prefect-$USER
.venv/bin/prefect server start &
export PREFECT_API_URL=http://127.0.0.1:4200/api
```

Then save the blocks and run:

```sh
.venv/bin/python create_blocks.py --config blocks.scc.toml --dry-run   # inspect first
.venv/bin/python create_blocks.py --config blocks.scc.toml
.venv/bin/python exec.py params.slurm.json
```

`blocks.scc.toml` names this cluster's partition and account. Edit them for
another site; `scontrol show partition` lists which accounts each partition
allows, and `sacctmgr show assoc user=$USER` lists the accounts you have.

## Files

| File | Contents | Heavy imports |
| --- | --- | --- |
| [`settings.py`](settings.py) | Constants and the random-stream contract | numpy |
| [`problem.py`](problem.py) | Molecule, mean field, LAS object, circuit builder, pass managers | mrh, PySCF, ffsim, Qiskit |
| [`sqd_settings.py`](sqd_settings.py) | The trim-SQD solver and its batch solvers | qiskit-addon-sqd; Fulqrum lazily, so the module imports without it |
| [`layout.py`](layout.py) | `RunLayout`: every record path and logical key | — |
| [`manifest.py`](manifest.py) | Write and verify a stage's result manifest | — |
| [`capture_worker.py`](capture_worker.py) | `LASState` → `Embedding` | mrh, PySCF |
| [`circuits_worker.py`](circuits_worker.py) | `Embedding` → `FragmentCircuits` | mrh, PySCF, ffsim, Qiskit |
| [`solve_worker.py`](solve_worker.py) | Counts + `SQDState` → `FragmentResult` + `SQDState` | qiskit-addon-sqd |
| [`replay_worker.py`](replay_worker.py) | `Embedding` + results → `LASState` and the energy | mrh, PySCF |
| [`sampling.py`](sampling.py) | The QPU gateway and its sampler paths | Qiskit |
| [`flow_params.py`](flow_params.py) | `FlowParameters`, validated once and written to `params.json` | pydantic |
| [`block_names.py`](block_names.py) | Default Prefect block names per stage and target | — |
| [`flow.py`](flow.py) | `@flow lassqd_distributed` and one `@task` per stage | Prefect, QCSC |
| [`create_blocks.py`](create_blocks.py) | TOML site configuration → saved Prefect blocks | Prefect, QCSC |
| [`exec.py`](exec.py) | Run the flow once from a parameters file | Prefect |
| [`deploy.py`](deploy.py) | `flow.serve()` deployment | Prefect |
| [`serial.py`](serial.py) | The bitwise reference: the same stages, one process or subprocesses | — |
| [`compare_runs.py`](compare_runs.py) | Walk two run directories; report the first divergence | — |
| [`blocks.scc.toml`](blocks.scc.toml) | This cluster's block configuration | — |
| `params.{runtime,local,slurm,smoke}.json` | Flow parameters for each path | — |

The three-way split of the problem modules is forced by a dependency boundary,
not by taste:

- **`settings.py`** is imported in the flow process, which needs
  `sampler_rng(cycle)`, so it may import NumPy and nothing else.
- **`problem.py`** imports mrh and PySCF and must **never** import
  `qiskit_addon_sqd`. The capture and replay workers import it, and
  [`tests/test_imports.py`](../../tests/test_imports.py) makes "no SQD import in
  capture or replay" a rule of the library: SQD initializes MPI at import time
  whenever a launcher variable such as `SLURM_NTASKS` is set and mpi4py is
  installed, so a multi-rank capture job that imported it would initialize MPI
  it never uses.
- **`sqd_settings.py`** quarantines `qiskit_addon_sqd` and Fulqrum behind
  `solve_worker.py`, the one worker that needs them.

## Stages and records

Every worker is `python <worker>.py --run-dir R --cycle C [...]`: it reads
records, writes records, and publishes `manifest.json` last. Records live in the
macrocycle directory; a job directory holds only the rendered script, the logs
and that stage's manifest, because QCSC's Slurm template hardcodes
`#SBATCH --output={{work_dir}}/output.out` and so every logical job needs its own
`work_dir`.

```
<run_dir>/params.json  problem.json  mf.chk  energies.csv  summary.json
<run_dir>/c000/ state.h5 embedding.h5 circuits.h5 samples.h5
                sqd_state_f00.h5 result_f00.h5 solve_f00.json ...
                next_state.h5 cycle.json
<run_dir>/c000/jobs/{capture,prepare,sample,solve-f00,solve-f01,replay}/
```

[`layout.RunLayout`](layout.py) is the single source of these paths, of each
job's readable logical key (`c007/solve/f01`) and of its job-script stem
(`c007-solve-f01`, which QCSC turns into `c007-solve-f01.slurm`). `jobs/sample/`
holds no script and no logs, because sampling is not a scheduler job, but it
does hold a manifest: the chosen provider job has to be recorded before anything
downstream consumes the counts.

| Stage | Reads | Writes | Library calls |
| --- | --- | --- | --- |
| capture | the previous `next_state.h5`, or nothing with `--initial` | `state.h5`, `embedding.h5`; and `problem.json`, `mf.chk` when absent | `las_from_run`, `initial_state`, `capture_embedding` |
| prepare | `embedding.h5` | `circuits.h5` | `build_circuits(embedding, problem.prepare_fragment, pass_manager=problem.pass_manager(target), groups=settings.GROUPS)` |
| sample (**in-flow**) | `circuits.h5` | `samples.h5` | `submit_circuits`, `collect_samples` |
| solve (per fragment) | `samples.h5`, the previous `sqd_state_f{F}.h5` | `result_f{F}.h5`, `sqd_state_f{F}.h5`, `solve_f{F}.json` | `FragmentSQD.prepare` → `SQDInputs.run` → `finalize` |
| replay | `embedding.h5`, every `result_f*.h5` | `next_state.h5`, `cycle.json`, `energies.csv` | `update_orbitals` |

`samples.h5` carries the whole `Embedding`, so a solve gets its `Fragment`, its
counts and `ao_ovlp` from one record. Each macrocycle's `state.h5` records the
state that macrocycle ran on, even though it duplicates the previous cycle's
`next_state.h5`: every cycle directory is then self-describing, and a rerun or a
comparison needs nothing from outside it.

**Sampling runs in the flow process, not as a Slurm job.** Both QPU paths QCSC
offers are already Prefect objects living there — `submit_sampler_job_task`,
`fetch_qiskit_job_result_task` and `QrmiRuntime` — and routing them through a
worker would mean reimplementing submit and fetch and losing QCSC's caching,
retries and artifacts. The gateway needs only Qiskit, h5py and
`lassqd.circuits`, never mrh, PySCF or SQD, which is the property
`tests/test_imports.py` guarantees; and keeping it here means no scheduler
allocation sits idle while a provider job is queued.

**Every stage task inspects `exit_status`.** `run_slurm_job` and `run_local_job`
*return* a result object carrying the process exit status and the final
scheduler state; they do not raise on failure. A task that does not look at it
treats a failed job as a successful one. The manifest is then verified too,
because a scheduler state says a process ended, not that it produced anything.

`replay_task` re-verifies every fragment's solve manifest before submitting the
replay job: the solve tasks checked them when they finished, but replay is about
to consume the files they name.

## Determinism

[`settings.py`](settings.py) holds the contract. Every stream is a pure function
of `(SEED, cycle, ifrag)`, re-derived in whichever process needs it, so no
process-local state can leak into the numbers.

```python
def _root() -> list[np.random.Generator]:
    return np.random.default_rng(SEED).spawn(NFRAGS + 1)


def sampler_rng(cycle: int) -> np.random.Generator:
    return _root()[0].spawn(cycle + 1)[-1]


def solver_root(ifrag: int) -> np.random.Generator:
    return _root()[1 + ifrag]
```

`solver_root(i)` reproduces the tutorial's
`np.random.default_rng(12345).spawn(nfrags + 1)[1 + i]` exactly, and
`FragmentSQD(seed=solver_root(i))` then spawns one child from it, so the
fragment streams of the serial reference and of the distributed run are
identical by construction. Per-fragment order independence is structural:
fragment `f`'s solve reads only `samples.h5` and `sqd_state_f{f}.h5`, and writes
only `f`-indexed files.

| Stream | Lives in | Distributed handling |
| --- | --- | --- |
| sampler, macrocycle `c` | nowhere | re-derived as `settings.sampler_rng(c)` |
| fragment `f` solver | the `SQDState` record | `sqd_state_f{f}.h5`, chained cycle to cycle |
| SQD recovery RNG | `SQDInputs.options["seed"]`, saved by `finalize` | carried inside `SQDState` |
| eigensolver start vector | `v0 = e_argmin(diagonal)` in the Fulqrum batch solver, PySCF's own guess in the addon's | unchanged; ARPACK's *default* start is random, which is why the Fulqrum adapter sets `v0` |

**`run_lassqd` cannot be the bitwise reference.** It holds one sampler object
whose stream advances across macrocycles and across the pubs of one job, and no
record persists it, so a workflow that samples afresh could never reproduce it.
[`serial.py`](serial.py) is the reference instead: the same stage loop, sharing
`settings.sampler_rng(cycle)`, `sqd_settings.make_solver(f)` and
`layout.RunLayout`. With a hardware sampler the question is moot anyway: the
counts differ between jobs, and the stored `FragmentSamples` fix them for
everything downstream.

### Run the numerics on one thread when you intend to compare runs

Measured on this cluster, with everything else held fixed:

| | RHF `mo_coeff` reproducible across processes? | compiled circuits (QPY) identical? |
| --- | --- | --- |
| `OMP_NUM_THREADS=1` | yes, bit for bit | yes |
| `OMP_NUM_THREADS=8` | no, differs by up to 1.2 × 10⁻¹³ | no |

PySCF's OpenMP reductions sum in a thread-dependent order, so RHF and CCSD
differ in their last bits between processes at more than one thread. That alone
would be far inside `lassqd.las.REPLAY_ATOL = 1e-10` and harmless — except that
the LUCJ angles come from the CCSD amplitudes, and sampling a circuit is chaotic
in them: a 10⁻¹³ change in `mo_coeff` gives a *different set of measured
bitstrings*, and from there two runs diverge completely.

Consequences, and they are the honest shape of the ladder:

- The bitwise steps (L3, L4, L5 below) run with
  `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`, and
  `create_blocks.py --ompthreads 1` renders that into the execution profiles.
  The root [installation guide](../../docs/installation.md) recommends one
  thread for this small calculation anyway.
- `problem.build_mean_field` caches the mean field in `<run_dir>/mf.chk`
  (cycle 0's capture creates it; nothing else writes it, so there is no race).
  That removes the variation *within* a run. It does not remove it between two
  run directories, which each run their own RHF — hence the thread setting.
- For anything that must compare across machines, **replay a stage** instead of
  resampling: rerun one stage on a stored request. Re-solving a stored
  `samples.h5` with `solve_worker.py` compares the solve stage exactly, whatever
  the upstream arithmetic did.

## Sampler paths

| Path | Configuration | Status here | Stored circuits |
| --- | --- | --- | --- |
| Qiskit Runtime (**default**) | `sampler="runtime"`, `runtime_block_name`, `transpile_target="backend:<name>"` | needs a saved IBM account | ISA, mapped onto the backend |
| Aer | `sampler="aer"`, `transpile_target="unmapped"` | works, no QPU needed | standard gates, no layout or routing |
| QRMI | `sampler="qrmi"` | **unavailable on this cluster** | `"unmapped"`; the node's hook maps it |
| ffsim | `sampler="ffsim"`, `transpile_target="native"` | `serial.py --inline-circuits` only | native ffsim gates |

The abstract circuit from `problem.prepare_fragment` — native ffsim gates and
all — goes straight into one pass manager: `build_circuits` measures it and runs
`problem.pass_manager(target)` on it, and the transpiled result is what
`FragmentCircuits` writes to QPY. `pm.pre_init = ffsim.qiskit.PRE_INIT` in both
pass-manager branches, as in the tutorial and `examples/fefe_lassqd_pdft`. There is no manual
`.decompose()` step: QCSC's bounded decompose loop is an artifact of serializing
between its two compilation hooks, a boundary this example's default path does
not have. `circuits_worker.py` instead **asserts** after transpilation that no
operation's module starts with `ffsim.`, because such a gate reloads from QPY as
an opaque gate and fails at submission. The `"unmapped"` branch — a preset pass
manager with basis gates and no coupling map — was checked to leave none for
`PrepareHartreeFockJW` and `UCJOpSpinUnbalancedJW`; that assertion is the check.

The branches differ only in what they target:

- **`"backend:<name>"`** → `generate_preset_pass_manager(target=...,
  optimization_level=3, initial_layout=...)`, producing the ISA circuits the
  `QCSCSamplerV2` path needs. It is kept as a module-level function
  `problem.transpile_for_target(circuit, target, config)` — the exact
  `(circuit, target, config)` signature QCSC calls a node-side `transpiler` hook
  with, and `config` holds only JSON-safe preset options, because only JSON
  crosses that boundary. Adopting QRMI's Slurm path therefore means **naming**
  that function in a `"module:function"` string rather than calling it.
- **`"unmapped"`** → `generate_preset_pass_manager(optimization_level=0,
  basis_gates=[...])`. No coupling map, so no layout and no routing: standard
  gates on virtual qubits. This is what Aer runs, and it is also the artifact to
  ship when the mapping stage moves to a compute node.
- **`"native"`** → no pass manager. `circuits_worker.py` rejects it with a
  message citing the QPY constraint; only `serial.py --inline-circuits` can use
  it, because the circuits then never leave memory.

The constraint behind all of this is `lassqd/circuits.py`'s: QPY loads native
ffsim gates as opaque gates, which `FfsimSampler` cannot run. As soon as prepare
and sample are separate processes the QPY round trip is mandatory, so
`FfsimSampler` is available only where circuits stay in memory, and Aer is the
local default.

Three facts worth stating outright:

- `QCSCSamplerV2.run` and `QCSCPrimitiveJob.result` are **coroutines**. So
  `await submit_circuits(...)` works and `job.job_id()` is synchronous, but
  `await job.result()` is required — which is why synchronous `run_lassqd`
  cannot drive that sampler and this example calls the stage functions instead.
  [`sampling.py`](sampling.py) awaits whatever the sampler returns, so one body
  serves both the simulators and QCSC.
- The `"backend:<name>"` target needs IBM Runtime credentials **on the compute
  node**, because that is where this example's prepare worker runs. A saved
  account under `~/.qiskit` on a shared filesystem is visible there, so it
  works, but it is a site assumption. The requirement is specific to the
  `QCSCSamplerV2` path: on the QRMI path the node needs no Runtime credentials,
  since SPANK supplies the resource and QRMI supplies the `Target`.
- `collect_samples(..., job_ids=...)` records the provider job identifier into
  `FragmentSamples.job_ids`, and the sample manifest records it too, before any
  downstream stage consumes the counts. Sampling is the one stage where two
  attempts would give different results, so which attempt was used has to be on
  record.

## Blocks for this cluster

`create_blocks.py --config blocks.scc.toml [--hpc-target local] [--ompthreads N]
[--dry-run]` saves, for each of the four job stages, a `CommandBlock`
(`cmd-lassqd-<stage>`, `executable_key="python"`, the worker script as
`default_args[0]`) and an `ExecutionProfileBlock` (`exec-lassqd-<stage>`,
`resource_class="cpu"`, `num_nodes=1`, `mpiprocs=1`, `launcher="single"`,
`gpus_per_node=None`, per-stage `ompthreads` and `walltime`), plus one
`HPCProfileBlock` per target and the `QiskitRuntimeConfig` block for the Runtime
path. Saving a block document registers its block type, so the QCSC block types
need no separate registration step.

Two decisions matter:

- **`executable_map={"python": "<example venv>/bin/python"}`, with the worker
  script as `default_args[0]`.** QCSC's Slurm template preflights the executable
  with `[ ! -x ]`, which a venv Python passes, so no `pre_commands` are needed —
  and that is exactly what lets **one set of execution profiles serve both
  `hpc_target="slurm"` and `hpc_target="local"`**, because local execution
  *rejects* `modules`, `pre_commands` and `gpus_per_node` while honouring
  `environments`. `hpc_target="local"` is therefore a faithful validation mode
  here, not a different configuration. Python puts a script's own directory on
  `sys.path[0]`, so `import problem` resolves in a worker even though the job
  runs with `cwd` set to its work directory.
- **`ompthreads` renders only `#SBATCH --cpus-per-task`; nothing exports
  `OMP_NUM_THREADS`.** `create_blocks.py` derives `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` from `ompthreads` in
  `environments` so the two cannot drift apart — equal thread counts are what
  make the ladder's comparisons meaningful. It also sets `PYTHONPATH=""` and
  `PYSCF_EXT_PATH=""` so no other copy of PySCF or mrh on the node can shadow
  the venv's (as `examples/fefe_lassqd_pdft/run.slurm` does),
  `PYTHONUNBUFFERED=1`, and `HDF5_USE_FILE_LOCKING=FALSE` for GPFS.

`blocks.scc.toml` carries `hpc_target`, `partition`, `account`, `python`,
`worker_dir`, `memory`, `block_prefix`, an `environments` table and a per-stage
table of `walltime` and `ompthreads`. `python` and `worker_dir` default to this
directory and its `.venv`, so an unmodified checkout needs no edit. The
cluster's confirmed facts: CPU partitions `cpu-{debug,low,mid,high}` of 5 nodes
with 192 CPUs and 772 GB, `sacct` through slurmdbd, `srun --mpi=list` offering
`pmi2`/`pmix`/`pmix_v3`, **no `module` command** and no QRMI SPANK resource. It
*does* require an `--account`, and `cpu-debug` is restricted to the `admin`
account, so the configuration uses `cpu-low` with the `general` account; an
empty account renders no `#SBATCH --account` at all and `sbatch` then fails with
"Invalid account or account/partition combination specified".
`HPCProfileBlock.slurm_qpu` and `slurm_ntasks` stay unset — the latter is
profile-wide, so the task count is left to `mpiprocs`, which is per-stage.

Node budgeting is unnecessary at `P_f = 1` and `F = 2`, but the solve task is
tagged `@task(tags=["lassqd-solve"])`, so a Prefect concurrency limit — or a
global limit whose slots stand for nodes, with each solve holding `P_f` of
them — can be attached without a code change.

## Verification

| Rung | What changes | Expected | Measured |
| --- | --- | --- | --- |
| L0 | — | ruff clean | clean |
| L1 | — | no SQD in capture/replay; no mrh or PySCF in the gateway | holds |
| **L2** | the serial reference, all stages in one process | baseline | −10.6948904321 Ha over 3 macrocycles |
| **L3** | the stages as separate processes | bitwise | 128 comparisons agree bitwise |
| **L4** | Prefect tasks and QCSC's local runtime | bitwise | 128 comparisons agree bitwise |
| **L5** | 15 real Slurm jobs | bitwise where the arithmetic is identical | two Slurm runs agree bitwise (128 comparisons); against the local run, capture agrees bitwise and the counts are the first divergence; solve agrees bitwise under stage replay |
| L6 | the Qiskit Runtime path | — | **not run**: no saved IBM account here |
| L7 | restart | resumes at the first uncommitted macrocycle, reusing stored counts | holds; a deleted result fails the manifest check |

Each rung changes one thing and is compared only with the one below it.
`compare_runs.py` walks both run directories in execution order and reports the
**first** divergence. All of the numerics below ran with

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

for the reason given under [Determinism](#determinism).

### L0 — lint (seconds)

From the repository root. The example is clean without the flow dependencies
installed, because ruff does not import anything.

```sh
uvx ruff@0.16.9 check .
uvx ruff@0.16.9 format --check .
```

### L1 — import hygiene (seconds)

These mirror [`tests/test_imports.py`](../../tests/test_imports.py) for the
example's own modules. They are **not** in the repository's `tests/`: the root
project's CI tests Python 3.10 and 3.14, and this example requires 3.11 or newer
with its own lock file, so importing its modules from the root test suite would
run them on an unsupported interpreter. The library's own boundary is already
covered there; this is the example's.

```sh
.venv/bin/python -c "import sys, capture_worker, replay_worker, problem, settings, layout; \
  assert 'qiskit_addon_sqd' not in sys.modules, sorted(m for m in sys.modules if 'sqd' in m)"
.venv/bin/python -c "import sys, layout, settings; assert not {'mrh','pyscf'} & set(sys.modules)"
.venv/bin/python -c "import sys, sampling, manifest, block_names; \
  assert not {'mrh','pyscf','qiskit_addon_sqd'} & set(sys.modules)"
```

### L2 — the serial reference (about 75 s)

```sh
.venv/bin/python serial.py --run-dir ~/runs/serial --max-cycles 3 --sampler aer
.venv/bin/python serial.py --classical-reference
```

Measured, Fulqrum batch solver, Aer on the `"unmapped"` circuits, 10,000 shots
per fragment per macrocycle, the tutorial's SQD settings:

| | Hartree |
| --- | --- |
| RHF | −10.4948302677 |
| LASSQD after 3 macrocycles (not converged) | −10.6948904321 |
| Classical LASSCF with exact fragment solvers | −10.8286357812 |

The classical reference agrees with the tutorial's −10.8286357812 exactly. The
LASSQD number does **not** approach the tutorial's −10.8268, and the reason is
worth stating: **the tutorial samples with 10% global depolarizing noise and Aer
here samples noiselessly.** A noiseless LUCJ circuit conserves electron number
exactly, so every measured bitstring is already in the right sector and SQD's
configuration recovery has nothing to repair; 10,000 noiseless shots gave only
28–47 *distinct* bitstrings per fragment, and the solved subspaces held 14–17
strings per spin sector against the `max_dim=150` allowed. The tutorial's noise
instead scatters a few per cent of the shots over the whole space, which is what
fills the subspace.

That is an accuracy limitation of the simulator configuration, not of the
workflow, and the same workflow reproduces the tutorial when it samples the way
the tutorial does. `serial.py` can, because `--inline-circuits` keeps the
circuits in memory and so can use native ffsim gates:

```sh
.venv/bin/python serial.py --run-dir ~/runs/ffsim --sampler ffsim \
  --transpile-target native --inline-circuits --global-depolarizing 0.1 --max-cycles 10
```

Measured: converged in 3 macrocycles to **−10.8265100755 Hartree** in 215 s,
with subspaces of 135–141 strings per spin sector — **2.1 mHartree** above the
classical reference, matching the 1.9 mHartree the tutorial reports. Noiseless
Aer remains the default for the rungs below because they compare *execution*,
and for that a cheap sampler that goes through the QPY record is what is wanted.

The choice of batch solver is not what decides these numbers.
`--sci-solver pyscf` (the SQD addon's `solve_sci_batch`) and the default
`--sci-solver fulqrum` were compared on the same stored counts: the subspaces
and CI strings came out identical, the fragment energies agreed to 1 × 10⁻¹⁴
Hartree, and the RDMs and amplitudes to about 1 × 10⁻⁸, which is the difference
between the two eigensolvers' tolerances. Fulqrum produced the energies quoted
here.

### L3 — the stages as separate processes (about 85 s)

`--subprocess` invokes the four workers with the exact argument list the flow
uses, with no Prefect involved. The only change from L2 is the process boundary.

```sh
.venv/bin/python serial.py --run-dir ~/runs/subprocess --max-cycles 3 --sampler aer --subprocess
.venv/bin/python compare_runs.py ~/runs/serial ~/runs/subprocess --bitwise
```

Measured: **`128 comparisons agree (bitwise) over 3 macrocycles`** — the
counts and their key order, every `SQDState` generator state, the carryover
strings and orbitals, the CI strings, the fragment digests, the RDMs, the
amplitudes, the orbitals and `energies.csv`.

### L4 — Prefect on `hpc_target="local"` (about 2 min)

The only change from L3 is who spawns the processes: Prefect tasks and QCSC's
local runtime instead of `subprocess.run`. Run the smoke parameters first.

```sh
export PREFECT_HOME=/var/tmp/prefect-$USER      # not a GPFS home; see below
.venv/bin/prefect server start &
export PREFECT_API_URL=http://127.0.0.1:4200/api
.venv/bin/python create_blocks.py --config blocks.scc.toml --hpc-target local --ompthreads 1 --dry-run
.venv/bin/python create_blocks.py --config blocks.scc.toml --hpc-target local --ompthreads 1
.venv/bin/python exec.py params.smoke.json
.venv/bin/python exec.py params.local.json
.venv/bin/python compare_runs.py ~/runs/subprocess ~/runs/lassqd-local --bitwise
```

Measured: **`128 comparisons agree (bitwise) over 3 macrocycles`**. The two
solve tasks ran concurrently, as their task-run logs show submitting within
milliseconds of each other.

`PREFECT_HOME` must be on local disk. With it on this cluster's GPFS home,
saving a block fails with `sqlite3.OperationalError: database is locked` and
`RuntimeError: Timed out while attempting to connect to ephemeral Prefect API
server`. `/var/tmp` here is a local XFS filesystem and works; a Postgres backend
would too. Only the flow process needs it — the workers never talk to Prefect.

### L5 — real Slurm (about 8 min; 15 jobs)

```sh
.venv/bin/python create_blocks.py --config blocks.scc.toml --hpc-target slurm --ompthreads 1
# The mean field is an input to the pipeline, not a product of it. Share it, so
# a comparison isolates the scheduler instead of two separate RHF runs.
mkdir -p ~/runs/lassqd-slurm && cp ~/runs/lassqd-local/mf.chk ~/runs/lassqd-slurm/
.venv/bin/python exec.py params.slurm.json
sacct -X --format=JobID,JobName%22,State,Elapsed,AllocCPUS,NodeList
cat ~/runs/lassqd-slurm/c000/jobs/solve-f00/*.slurm
```

Measured: 15 Slurm jobs — three macrocycles of capture, prepare, two solves and
replay, with sampling in the flow process — converging in 3 macrocycles to
−10.7249260952 Hartree, with the two solves of each macrocycle submitted
concurrently. `sacct` names each job after its script, so the logical keys
appear directly in the accounting records (`c002-solve-f01.slurm`), because
QCSC's Slurm template sets no `--job-name` of its own. Note
that `sacct` reports `AllocCPUS=2` for `--cpus-per-task=1`, because this cluster
allocates whole hyperthread pairs — what pins the arithmetic is the thread
variables in `environments`, not the allocation.

**Two Slurm runs of the same parameters agree bitwise.** Repeating the run into
a second directory, with the same shared mean field:

```sh
mkdir -p ~/runs/lassqd-slurm-b && cp ~/runs/lassqd-local/mf.chk ~/runs/lassqd-slurm-b/
.venv/bin/python exec.py params.slurm.json --run-dir ~/runs/lassqd-slurm-b
.venv/bin/python compare_runs.py ~/runs/lassqd-slurm ~/runs/lassqd-slurm-b --bitwise
```

```
128 comparisons agree (bitwise) over 3 macrocycles
```

So the scheduler path is as reproducible as the local one. Comparing it with the
**local** run is where a difference appears, and it appears in one place:

```sh
.venv/bin/python compare_runs.py ~/runs/lassqd-local ~/runs/lassqd-slurm --bitwise
```

```
first divergence after 18 comparisons:
  c000/samples.h5.counts[0]: 37 and 43 bitstrings
```

The 17 comparisons before it agree bitwise — `params.json`, `problem.json`, the
first macrocycle's orbitals and fragment RDMs, and both fragments' Hamiltonian
digests — so **capture is bitwise identical across machines**. Two further
checks pin down what is not:

| Check | Result |
| --- | --- |
| Re-run **prepare** on the login node from the Slurm run's `embedding.h5`, twice | the two agree bitwise with each other; neither matches the circuits the compute node produced |
| Re-run **solve** on the login node from the Slurm run's `samples.h5` — *stage replay* | both fragments' `e_tot`, CI strings, RDMs, amplitudes, generator state and carryover strings are **bitwise** identical to the compute node's |

So the chain is: on a different CPU the prepare stage's CCSD amplitudes differ in
their last bits, the LUCJ angles inherit that, and sampling a circuit is chaotic
in its angles. Everything on either side of that one step is bitwise
reproducible across machines, the fragment solve included — which is why
**stage replay**, not resampling, is the tool for comparing across machines.
Rerunning one stage on a stored request takes four commands:

```sh
mkdir -p ~/runs/replay/c000
cp ~/runs/lassqd-slurm/params.json ~/runs/lassqd-slurm/problem.json ~/runs/replay/
cp ~/runs/lassqd-slurm/c000/samples.h5 ~/runs/replay/c000/
.venv/bin/python solve_worker.py --run-dir ~/runs/replay --cycle 0 --fragment 0
```

### L6 — the Qiskit Runtime path

**Not exercised**: this cluster has no saved IBM Quantum account. The path is
implemented and configured in [`params.runtime.json`](params.runtime.json) —
`sampler="runtime"`, the `runtime-lassqd` block, and
`transpile_target="backend:ibm_torino"` — and `create_blocks.py` saves the
`QiskitRuntimeConfig` block it needs. With an account, the rung is one
macrocycle followed by the stage replay above: re-solving the stored
`samples.h5` reproduces the fragment results from fixed counts, which is the
only form of reproducibility a hardware sampler admits.

### L7 — restart

A rerun with the same `run_dir` resumes at the first macrocycle that has no
committed `cycle.json`, from the `LASState` the previous replay committed. The
unit of restart is a whole macrocycle: SQD's API exposes no mid-iteration
continuation state, so a killed solve is redone from its input checkpoint and
its stored counts. Kill the flow during a solve and run `exec.py` again.

Measured, by deleting macrocycle 0's `cycle.json` and `next_state.h5` from a
finished run and rerunning it:

```
Task run 'sample_task-ac3' - reusing the stored counts of c000/sample/all from job 6bbffaac-...
Task run 'replay_task-ed6' - committed c000/replay/all: E = -10.6050994440 Ha
```

— the same energy the first run committed. Rerunning a *complete* run instead
logs `resuming after 1 committed macrocycles` and does no work.

The classical stages are rerun, because each is a deterministic function of its
inputs. **Sampling is not**: two sampler jobs give different counts, so
`sample_task` reuses this macrocycle's stored counts when a valid sample
manifest names them *and* their fragments' digests match the embedding that was
just captured. It is the one stage whose attempts are not interchangeable.

The manifest check is what makes a missing result loud rather than silent:

```sh
rm ~/runs/lassqd-local/c002/result_f01.h5
.venv/bin/python -c "import layout, manifest; run = layout.RunLayout('$HOME/runs/lassqd-local'); \
  manifest.verify(run.manifest(2, 'solve', 1), key=run.logical_key(2, 'solve', 1))"
```

```
FileNotFoundError: .../c002/jobs/solve-f01/manifest.json: references a missing
file .../c002/jobs/solve-f01/../../result_f01.h5
```

`replay_task` runs exactly that check for every fragment before it submits the
replay job, so a deleted or altered result cannot be replayed stale.

The module-digest check has been seen to fire for real. Comparing a run made
before an edit to `problem.py` with one made after it, `compare_runs.py`
stopped at

```
first divergence after 10 comparisons:
  problem.json['module_digests']['problem.py']: 'e1951645...' and 'c06b2cf4...'
```

which is what `verify_problem_task` raises on inside a flow run: it means the
worker that captured was built from different code than the flow process
expects. (In that case the edit was import-only and the energies were
unchanged, which the rest of the comparison then confirmed.)

## Not implemented, and where it attaches

| What | Where it attaches here |
| --- | --- |
| The collective MPI `sci_solver` with local trim | the `rank = 0` line of [`solve_worker.py`](solve_worker.py) and `sqd_settings.sci_solver`, plus one `execution_profile_overrides={"num_nodes": P_f, "launcher": "srun", "mpi_options": ["--mpi=pmix_v3"]}` in `flow.solve_task` |
| Claim records, so that a job whose submit response was lost can be found again, and reconciliation of an ambiguous submission | `flow.run_stage`, *before* submission |
| Semantic manifest validation: RDM traces, the 2-RDM partial trace, a recomputed energy | `manifest.verify` |
| A fuller logical job key: calculation identifier, parent-snapshot digest, request digest, attempt | `layout.RunLayout.logical_key`. The run directory stands in for the calculation and the snapshot, and the manifest carries the attempt |
| `run_jobs_from_blocks_bulk` | would replace the `F` solve tasks with one bulk task. It needs an application-supplied Slurm `QueueProbe` — QCSC ships one only for Fugaku, and a probe that raises reports zero free slots, so submission stalls without an error. Per-fragment tasks are preferable anyway: they express each solve's dependency on that fragment's counts and report its progress on their own |
| Per-macrocycle subflows, which would group a macrocycle's tasks in the Prefect UI | the `lassqd_distributed` body |
| The QRMI access path | `sampling.make_sampler`. Unavailable here: no `qrmi` binary, no `qrmi_config.json`, and `GresTypes=gpu` only, so no QRMI SPANK resource. `HPCProfileBlock.slurm_qpu` stays unset either way |

The collective solve is the one that drove several decisions above, so it is
worth being precise about why it is not here. It must reproduce what
`TrimPolicy.refine` selects using SQD's **private** `_select_carryover`, pin an
SQD revision and test the equality, and abort the MPI world on any rank's
failure, so that a failure on one rank cannot leave its peers blocked in a
collective — a numerical-equivalence project of its own. This cluster also
cannot validate it well: no system MPI, no `module` command, and `mpi4py` only
from a PyPI wheel with bundled MPICH under `srun --mpi=pmix_v3`.

What this example does contribute towards it: `solve_worker.py` is already in
the three-call form, which is bitwise identical to `FragmentSQD.step`
(`lassqd/sqd.py`); the pre-flight batch check already runs before SQD is
entered, rather than failing inside the batch solver on whichever rank owns an
empty batch; and the execution profiles already carry `num_nodes`, `mpiprocs`,
`launcher` and `mpi_options` at degenerate values.

## Known limitations

- `HPCProfileBlock.slurm_memory` and `slurm_extra_directives` are
  **profile-wide**, so per-stage memory would need one `HPCProfileBlock` per
  stage. This example uses one profile and one memory figure.
- QCSC's Slurm template sets no `--job-name`, so the scheduler names a job after
  its script. Job-name-based recovery would need
  `run_job_from_blocks(submit_env={"SBATCH_JOB_NAME": key})`, which is accepted
  in QCSC 0.3.1 but is not used or verified here.
- `hpc_target="local"` rejects `pre_commands`, so anything like
  `ulimit -s unlimited` must be set in the shell that launches the flow.
- A manifest is published with `os.link`, which refuses to replace one, so at
  most one attempt's result is accepted. Concurrent attempts are **not**
  reconciled: a deliberate rerun clears the previous manifest in
  `flow.run_stage`, not in the worker.
- The Qiskit Runtime path (L6 above: one macrocycle, then re-solving the
  stored `samples.h5` to confirm stage replay) is implemented and configured in
  `params.runtime.json`, but **was not exercised**, because this cluster has no
  saved IBM Quantum account. The Aer and ffsim paths were.
- The example is fixed to this H₂₀ fragmentation. `settings.py` is where the
  geometry, active spaces and SQD settings live; `problem.py` is where the
  ansatz and compilation targets live.

## Relation to the other examples

[`fefe_lassqd`](../fefe_lassqd) and [`fefe_lassqd_pdft`](../fefe_lassqd_pdft)
are single-process scripts that call `run_lassqd`, which is the right shape when
one process can hold the whole calculation. This example is for when it cannot:
when the fragment solves want their own allocations, when a stage has to be
rerun from stored records, or when a QPU and an HPC scheduler have to be
coordinated by something that survives a crash.
