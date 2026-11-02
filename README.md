# LASSQD

Sample-based quantum diagonalization (SQD) as a fragment solver for the localized
active space self-consistent field (LASSCF) method.

LASSQD combines [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) with
[mrh](https://github.com/MatthewRHermes/mrh)'s RDM-based LASSCF
(`mrh.my_pyscf.mcscf.lasscf_rdm`). Both are installed as dependencies.

Start with the [Run LASSQD notebook](docs/lassqd.ipynb) to build an H₁₂ system
from scratch, run two H₆ fragments with local ffsim sampling and a Fulqrum batch
solver, and compare the result with classical LASSCF.

## Repository contents

| Path | Contents |
| --- | --- |
| [lassqd/](lassqd/) | LASSQD package |
| [docs/lassqd.ipynb](docs/lassqd.ipynb) | H₁₂ tutorial with local sampling and Fulqrum |
| [examples/](examples/README.md) | FeFe calculations with classical sampling, IBM hardware, and SQD-PDFT |
| [tests/](tests/) | pytest tests |
| [pyproject.toml](pyproject.toml) | Package metadata, dependencies, and development tools |
| [uv.lock](uv.lock) | Locked dependency versions for uv |

## Installation

Use Python 3.10+ with pip or uv in a virtual environment; conda is not required.
The installation was verified on Linux with Python 3.14.
The native dependencies need Git, Make, C/C++ and Fortran compilers, OpenMP,
and BLAS/LAPACK development libraries. On Debian/Ubuntu, install these once with:

```bash
sudo apt-get install python3-venv git build-essential gfortran libblas-dev liblapack-dev
```

### With pip

From the root of this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate

# Build prerequisites: PySCF must be installed before the native extensions build.
python -m pip install --upgrade \
  pip 'setuptools>=77' wheel setuptools-scm 'cmake>=3.19,<4' 'pyscf==2.14.0'
python -m pip install --no-build-isolation \
  -e 'git+https://github.com/MatthewRHermes/mrh.git@f007765830b62c83aef5839119732a63f5e463e0#egg=mrh' \
  -e '.[sim]'
```

Pip builds pyscf-forge and mrh automatically. `--no-build-isolation` lets their
builds use the installed PySCF and build tools. `pyproject.toml` pins PySCF 2.14.0,
pyscf-forge `f55abdb`, and the mrh version from commit `f0077658`.
LASSQD also requires SciPy <1.18 because mrh's `LinearOperator` subclasses are
incompatible with SciPy 1.18.

mrh uses an editable Git installation: pip clones it into `.venv/src/mrh`, which
satisfies its directory-name requirement and keeps its compiled libraries
available. Keep that checkout for as long as you use the environment. A regular
mrh wheel currently omits those libraries, so `pip install .` alone is not yet
sufficient to set up LASSQD.
The explicit mrh Git URL in the command above supplies the declared dependency
as an editable installation, which dependency metadata cannot request.

The `sim` extra installs ffsim and Qiskit Aer for local circuit simulation, as
used by the tutorial and the examples; LASSQD itself does not need them.

On macOS, see [mrh's README](https://github.com/MatthewRHermes/mrh#readme) for
compiler and OpenMP setup. On Windows, use Linux through WSL.

For the IBM hardware example, also run `python -m pip install -e '.[ibm]'`.

### With uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and the
native prerequisites listed above, then run from the root of this repository:

```bash
# uv requires a local directory for an editable installation.
git clone https://github.com/MatthewRHermes/mrh.git .deps/mrh
git -C .deps/mrh checkout f007765830b62c83aef5839119732a63f5e463e0
uv sync --python 3.14
```

`uv sync` creates `.venv`, installs LASSQD and its development dependencies
(including ffsim and Qiskit Aer), and builds the native extensions automatically. `pyproject.toml` supplies the extra
build requirements, including CMake and the pinned PySCF version, in isolated
build environments. No separate Python build-tool installation or environment
activation is needed.

Keep `.deps/mrh`: the configured editable source uses its Python code and
compiled libraries. The checkout lives outside `.venv` so it survives recreating
the environment. Subsequent setup only needs `uv sync`.

Run scripts with `uv run python path/to/script.py`. For the IBM hardware example,
install with `uv sync --extra ibm` and run with
`uv run --extra ibm python path/to/script.py`.

## Running the notebook

The [H₁₂ tutorial](docs/lassqd.ipynb) uses Fulqrum for the SQD batch solver.
Fulqrum and ipykernel are included in the `dev` dependency group, which uv installs
by default. With pip, install that group explicitly as shown below.

For the small tutorial calculation, use one thread for OpenMP, OpenBLAS, and MKL
to avoid the overhead of coordinating many CPU threads. Set these variables in
your shell before starting Jupyter so the kernel inherits them before importing
the numerical libraries:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

After installing LASSQD, run from the repository root. With the pip environment
activated:

```bash
python -m pip install --group dev jupyterlab
python -m jupyterlab docs/lassqd.ipynb
```

With uv:

```bash
uv run --with jupyterlab python -m jupyterlab docs/lassqd.ipynb
```

Select the Python kernel from that environment, then run the cells in order.
Restart Jupyter and its kernel if you change the thread settings after launch.
For larger calculations, adjust the thread counts to the CPU resources available.

## The `lassqd` package

Each hybrid cycle starts from a LAS state: orbitals and fragment RDMs. LASSQD
captures each fragment's embedded Hamiltonian from mrh, samples one circuit per
fragment, solves each fragment with SQD on its samples, and then replays the
fragment RDMs into mrh, which takes one LASSCF orbital step. The fragments are
solved outside mrh and independently of each other.

| Module | Contents |
| --- | --- |
| [lassqd.records](lassqd/records.py) | `LASState`, `Fragment`, `Embedding`, `FragmentSamples`, `SQDState`, `FragmentResult`: the data passed between stages, with HDF5 `save`/`load` |
| [lassqd.las](lassqd/las.py) | `LASSCF`, `initial_state`, `capture_embedding`, `replay_orbital_step`: the only interface to mrh |
| [lassqd.circuits](lassqd/circuits.py) | `build_circuits`, `submit_circuits`, `collect_samples` |
| [lassqd.sqd](lassqd/sqd.py) | `FragmentSQD`: configuration recovery, carryover and the fragment solve |
| [lassqd.hybrid](lassqd/hybrid.py) | `solve_fragments`, `update_orbitals`, and the serial driver `run_lassqd` |
| [lassqd.fragment](lassqd/fragment.py) | `fragment_rohf` and `sampling_basis_rohf`: the fragment ROHF basis |
| [lassqd.pdft](lassqd/pdft.py) | `lassqd_pdft_energy`: LAS-PDFT on a LAS state |

`lassqd.records` imports only NumPy and h5py, so stored records can be read
without mrh, PySCF or Qiskit installed.

### Usage

The sketch below assumes that `mf` is a prepared PySCF mean-field object,
`mo_coeff` contains localized orbitals for two (6e,5o) fragments, and
`prepare_fragment` is a circuit builder as in the [tutorial](docs/lassqd.ipynb)
or the [FeFe example](examples/fefe_lassqd/lassqd_fefe.py). Build the LAS object
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

### Circuits

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

### Running the stages separately

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
  including one recovered later by job ID, or results combined from several jobs
  with `submit_circuits(..., indices=...)`.
- `update_orbitals` re-enters mrh at the same state with kernels that check each
  Hamiltonian against the captured one and return the stored fragment RDMs. mrh
  then computes the LAS energy, takes one orbital step and canonicalizes.

To resume, load the records into a new LAS object built the same way: pass
`LASState.load(...)` to `capture_embedding` or `run_lassqd`, assign
`solver.state = SQDState.load(...)`, and solve `FragmentSamples.load(...)` again
if the failure came after sampling. A resumed cycle reproduces the
uninterrupted one bit for bit.

### Fragment SQD

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
  [tutorial](docs/lassqd.ipynb) uses a Fulqrum solver.
- **Callback.** `callback(results)` receives each round's `SCIResult` list,
  with energies excluding `h0` and states in the sampling basis, for example to
  record diagnostics as in the [FeFe example](examples/fefe_lassqd/lassqd_fefe.py).
- **Collective solves.** `prepare`, `SQDInputs.run` and `finalize` split a solve
  into rank-0 setup, the SQD call every MPI rank enters, and rank-0 finishing.

**Random streams.** Give the sampler and each solver an independent stream,
for example `sampler_seed, *solver_seeds = np.random.default_rng(seed).spawn(nfrags + 1)`.
A generator passed as `seed` spawns a child stream, so solvers given the same
generator still draw independently, and each solver's stream continues across
cycles in `solver.state`.

## Tests

With pip:

```bash
python -m pip install --group dev
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest
```

With uv, pytest is included in the default development group:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m pytest
```
