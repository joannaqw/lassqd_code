LASSQD: a code to run Sample-based quantum diagonalization as parallel fragment solver for the localized active space self-consistent field method

SQD is provided by the open-source [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd)
package; this repo no longer vendors its own copy.

LASSCF is provided by [mrh](https://github.com/MatthewRHermes/mrh), installed as a package;
this repo no longer vendors a copy of it either. `lassqd/las.py` is the interface to mrh's
RDM-based LASSCF (`mrh.my_pyscf.mcscf.lasscf_rdm`).

contains:
- `lassqd/`: the LASSQD package
- `examples/`: FeFe LASSQD calculations on a classical simulator and on IBM hardware, and SQD-PDFT (see `examples/README.md`)
- `tests/`: pytest tests
- `pyproject.toml`: package metadata and dependencies

## The `lassqd` package

Each hybrid cycle runs at fixed orbitals: the fragment Hamiltonians are turned into
circuits, all fragment circuits are glued into one job and sampled, and then SQD on
each fragment's counts is the fragment solver for one LASSCF orbital step.

- `lassqd.las`: `LASSCFNoSymm`, `fragment_hamiltonians`, `set_fragment_kernels` (mrh interface)
- `lassqd.circuits`: `lucj_circuit` (CCSD-initialized, linear-method-optimized LUCJ), `glue_circuits` (one classical register per fragment), `preset_pass_manager` (qiskit's preset pass manager with ffsim's `PRE_INIT` stage)
- `lassqd.sqd`: `FragmentSQD`, the SQD fragment kernel (configuration recovery, optional determinant carryover between cycles), and `solve_sci_nroots`
- `lassqd.hybrid`: `run_lassqd`, the hybrid loop; it samples with any qiskit SamplerV2 primitive (Aer, IBM Runtime, ...)
- `lassqd.pdft`: `lassqd_pdft_energy` (LAS-PDFT on the SQD RDMs), `save_rdms`, `load_rdms`

```python
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

from lassqd import FragmentSQD, LASSCFNoSymm, lassqd_pdft_energy, preset_pass_manager, run_lassqd

las = LASSCFNoSymm(mf, (5, 5), ((4, 2), (2, 4)), spin_sub=(3, 3))
solvers = [FragmentSQD(iterations=6, n_batches=15, samples_per_batch=50) for _ in range(las.nfrags)]
sampler = SamplerV2(options={"backend_options": {"method": "matrix_product_state"}})
pass_manager = preset_pass_manager(AerSimulator(method="matrix_product_state"))
result = run_lassqd(las, mo_coeff, solvers, sampler, pass_manager=pass_manager, shots=100_000)
e_pdft = lassqd_pdft_energy(las, result.casdm1frs, result.casdm2fr, result.mo_coeff)
```

`FragmentSQD` returns the energy and RDMs of the lowest-energy batch of the last
configuration-recovery iteration, and SQD sees only the alpha one-electron
Hamiltonian `h1s[0]`. `result.mo_coeff` with `result.casdm1frs`/`result.casdm2fr` is
one consistent LAS wave function (RDMs in those orbitals' active space), which is
what LAS-PDFT needs.

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
python -m pip install --upgrade pip 'setuptools>=77' wheel setuptools-scm 'cmake>=3.19,<4' 'pyscf==2.14.0'
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
build environments. No separate prerequisite installation or activation is
needed.

Keep `.deps/mrh`: the configured editable source uses its Python code and
compiled libraries. The checkout lives outside `.venv` so it survives recreating
the environment. Subsequent setup only needs `uv sync`.

Run scripts with `uv run python path/to/script.py`. For the IBM hardware example,
install with `uv sync --extra ibm` and run with `uv run --extra ibm python
path/to/script.py`.

## Tests

With pip:

```bash
python -m pip install --group dev
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest
```

With uv, pytest is included in the default development group:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python -m pytest
```
