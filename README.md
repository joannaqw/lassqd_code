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
  -e .
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

On macOS, see [mrh's README](https://github.com/MatthewRHermes/mrh#readme) for
compiler and OpenMP setup. On Windows, use Linux through WSL.
The old `environment.yml` is retained as a historical environment snapshot.

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

`uv sync` creates `.venv`, installs LASSQD and its development dependencies, and
builds the native extensions automatically. `pyproject.toml` supplies the extra
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

Each hybrid cycle builds fragment Hamiltonians at fixed orbitals, prepares and
samples all fragment circuits in one job, and solves each fragment with SQD.
LASSCF then uses the resulting reduced density matrices (RDMs) for one orbital
step before the next cycle.

| Module | Interface |
| --- | --- |
| [lassqd.las](lassqd/las.py) | `LASSCFNoSymm`, `fragment_hamiltonians`, and `set_fragment_kernels`: the mrh interface |
| [lassqd.basis](lassqd/basis.py) | `fragment_mo_basis` and `fragment_rohf`: fragment orbital preparation |
| [lassqd.sqd](lassqd/sqd.py) | `FragmentSQD`: configuration recovery and determinant carryover |
| [lassqd.hybrid](lassqd/hybrid.py) | `run_lassqd` and `HybridResult`: the hybrid loop and its result |
| [lassqd.pdft](lassqd/pdft.py) | `lassqd_pdft_energy`, `save_rdms`, and `load_rdms`: LAS-PDFT and wave-function storage |

### Usage

`run_lassqd` requires a `circuit_builder(h1, h2, norb, nelec)` callable. Define
`prepare_fragment` as in the [tutorial](docs/lassqd.ipynb) or the
[FeFe example](examples/fefe_lassqd/lassqd_fefe.py), then pass it explicitly.
The sketch below assumes that `mf` is a prepared PySCF mean-field object and
`mo_coeff` contains localized orbitals for the two (6e,5o) fragments:

```python
from functools import partial

import ffsim
from qiskit_addon_sqd.fermion import solve_sci_batch

from lassqd import FragmentSQD, LASSCFNoSymm, lassqd_pdft_energy, run_lassqd

las = LASSCFNoSymm(mf, (5, 5), ((4, 2), (2, 4)), spin_sub=(3, 3))
solvers = [
    FragmentSQD(
        samples_per_batch=50,
        max_iterations=6,
        num_batches=15,
        sci_solver=partial(solve_sci_batch, spin_sq=2.0, max_cycle=200, tol=1e-12),
    )
    for _ in range(las.nfrags)
]
sampler = ffsim.qiskit.FfsimSampler()
result = run_lassqd(
    las,
    mo_coeff,
    solvers,
    sampler,
    circuit_builder=prepare_fragment,
    glue_circuits=False,
    shots=100_000,
)
e_pdft = lassqd_pdft_energy(las, result.casdm1frs, result.casdm2fr, result.mo_coeff)
```

`max_cycles=50` and `conv_tol=1e-5` are the hybrid loop defaults. Check
`result.converged` and `result.e_hist` before using the final energy: convergence
means that the absolute energy change between consecutive cycles is below
`conv_tol`.

### Circuit builder

The builder receives integrals in the LAS basis (`h2` in chemists' notation).
It must return an unmeasured circuit with `2 * norb` qubits whose occupations
refer to the fragment ROHF basis returned by `lassqd.basis.fragment_mo_basis`,
which is also used by `FragmentSQD`. Alpha orbitals occupy qubits `0..norb-1`
and beta orbitals `norb..2*norb-1`. Orbital and electron counts may be NumPy
integers; convert them to Python integers for Qiskit.

There is no default ansatz. The example builders choose CCSD-initialized local
unitary cluster Jastrow (LUCJ) circuits and specify their interaction pairs,
layer count, and compression settings locally. Code using the former
`lassqd.lucj_circuit` default must now supply its own builder through
`circuit_builder`.

### Sampling

For classical LUCJ sampling, use `FfsimSampler` with `glue_circuits=False` and
no pass manager. Each fragment is a separate measured circuit in the same job,
so ffsim works in each fragment's fixed electron-number sector. The example above
has two (6e,5o) fragments: 10 qubits and 50 state amplitudes each. The larger
classical FeFe example has two (6e,10o) fragments: 20 qubits and 9,450 amplitudes each.

The default `glue_circuits=True` glues the fragments into one circuit with
one classical register per fragment, as used by the IBM hardware example. Aer
and IBM Runtime require a `pass_manager` to compile circuits containing native
ffsim gates. Create it with `generate_preset_pass_manager(...)`, then set
`pass_manager.pre_init = ffsim.qiskit.PRE_INIT`.
`shots` is the number of samples per fragment per cycle in either mode;
`shots=None` uses the sampler's default.

### Fragment SQD

`FragmentSQD` uses `qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`
for configuration recovery and diagonalization. It runs up to `max_iterations`
rounds, using the best batch's occupancies for recovery, and returns the
energy and RDMs of the lowest-energy batch across all iterations. Optional
determinant carryover is handled by the addon within a call and by `FragmentSQD`
between calls. Circuit construction and SQD use only the alpha one-electron
Hamiltonian `h1s[0]`, including when the embedding is spin-dependent.

`FragmentSQD(samples_per_batch, **sqd_options)`
passes SQD options to `diagonalize_fermionic_hamiltonian` using the addon's names
and defaults. Only `samples_per_batch` is required. The Hamiltonian, orbital and
electron counts come from the fragment kernel arguments; the sampled `BitArray`
is built from `solver.counts`.

The defaults in the locked qiskit-addon-sqd version are:

| Option | Default |
| --- | --- |
| `num_batches` | `1` |
| `max_iterations` | `100` |
| `energy_tol` | `1e-8` |
| `occupancies_tol` | `1e-5` |
| `carryover_threshold` | `1e-4` |

Carryover also persists between fragment calls.
Set `carryover_threshold=None` (or `np.inf`) to disable it. Set either convergence
tolerance to `0.0` to run all requested iterations.

For example, use five batches and limit each spin sector to 100 strings:

```python
solver = FragmentSQD(
    samples_per_batch=50,
    max_iterations=100,
    num_batches=5,
    energy_tol=1e-8,
    occupancies_tol=1e-5,
    max_dim=100,
    seed=0,
)
```

Both convergence criteria must be satisfied to stop early. Options are stored in
`solver.sqd_options` and can be updated between calls, for example
`solver.sqd_options["max_dim"] = 200`. The
`seed` option becomes a persistent NumPy generator on use, so repeated calls
advance the same random stream. A custom `callback(results)` is passed directly
to the addon and receives each iteration's list of `SCIResult` objects, with energies
excluding `h0` and states in the fragment ROHF basis. A custom `sci_solver` accepts
`(ci_strings, h1, h2, norb, nelec)` and returns that list; it controls its own spin
constraint and Davidson settings. Explicit configurations are merged with
carryover from the preceding call before the addon applies `max_dim`.
Explicit configurations and `initial_occupancies` use the current fragment ROHF basis.

Code using the previous constructor should replace `iterations` with
`max_iterations` and `n_batches` with `num_batches`, and pass these as keyword
arguments. Early stopping and carryover are enabled by default.

With `sci_solver=None`, the addon uses `solve_sci_batch` with its default Davidson
settings and no total-spin penalty. To set a spin target or change Davidson
settings, pass a configured solver such as
`partial(solve_sci_batch, spin_sq=2.0, max_cycle=200, tol=1e-12)`, as above.
These settings belong to the batch solver; `energy_tol` and `occupancies_tol`
control convergence of configuration recovery.

The [tutorial](docs/lassqd.ipynb) demonstrates a custom `sci_solver` built with
Fulqrum and SciPy's `eigsh`, without a total-spin penalty.

#### Recording and saving SQD results

`FragmentSQD` keeps the final `e_tot`, `dm1s`, `dm2`, `sci_state`, and carryover
strings. Use a callback to record iteration summaries or log progress:

```python
import numpy as np

history = []


def record_iteration(results):
    best = min(results, key=lambda result: result.energy)
    history.append(
        {
            "energy": [r.energy for r in results],
            "spin_square": [r.sci_state.spin_square() for r in results],
            "dimension": [r.sci_state.amplitudes.size for r in results],
            "alpha_count": [len(r.sci_state.ci_strs_a) for r in results],
            "beta_count": [len(r.sci_state.ci_strs_b) for r in results],
            "occupancies": np.concatenate(best.orbital_occupancies),
        }
    )
    print(f"SQD round {len(history)}: lowest energy = {best.energy:.12g}")


solver = FragmentSQD(50, callback=record_iteration)
```

The caller owns `history`: clear it before a new solve to record that solve alone.
These summaries do not retain all the batch wave functions. Callback energies
exclude `h0`; callback states and occupancies use the current fragment ROHF basis.
The solver's final attributes are updated **after** the solve returns, so save them
after `solver(...)` or in `run_lassqd`'s cycle callback:

```python
from pathlib import Path

directory = Path("fragment_results")
directory.mkdir(parents=True, exist_ok=True)
solver.sci_state.save(directory / "sci_vec")
for name in history[0]:
    np.save(directory / name, np.asarray([row[name] for row in history]))
if solver.carryover_strings is not None:
    np.save(directory / "alpha_strings", solver.carryover_strings[0])
    np.save(directory / "beta_strings", solver.carryover_strings[1])
```

For a complete cycle callback preserving the FeFe example's checkpoint filenames,
see [lassqd_fefe.py](examples/fefe_lassqd/lassqd_fefe.py). It saves after each hybrid
cycle completes and clears the per-fragment histories. The SQD state remains in
its fragment ROHF basis; LAS orbitals saved after the cycle include the orbital update.

Migration: remove `output_dir` and `verbose` from `FragmentSQD` construction and
replace reads of `solver.e_hist`, `d_hist`, `a_hist`, `b_hist`, `s_hist`, and
`occupancy_hist` with callback-owned summaries. Saving and diagnostic logging are
now caller responsibilities. `HybridResult.e_hist` still records molecular total
energies after each hybrid cycle.

### LAS-PDFT and saved wave functions

`result.mo_coeff`, `result.casdm1frs`, and `result.casdm2fr` describe one consistent
LAS wave function: the RDMs use the active space of the returned orbitals. Pass
them together to `lassqd_pdft_energy`, which uses the `tPBE` on-top functional by
default.

To save that wave function, pass `mo_coeff=result.mo_coeff` to `save_rdms` along
with the fragment RDMs. `load_rdms` returns `(casdm1frs, casdm2fr, mo_coeff)`;
`mo_coeff` is `None` for older files that contain only RDMs. See the
[FeFe example notes](examples/README.md#fefe_lassqd_pdft) before using the
historical data committed with that example.

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
