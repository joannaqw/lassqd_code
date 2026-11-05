# Installation

Use Python 3.10+ with pip or uv in a virtual environment; conda is not required.
The installation was verified on Linux with Python 3.14.
The native dependencies need Git, Make, C/C++ and Fortran compilers, OpenMP,
and BLAS/LAPACK development libraries. On Debian/Ubuntu, install these once with:

```bash
sudo apt-get install python3-venv git build-essential gfortran libblas-dev liblapack-dev
```

## With pip

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

## With uv

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

The [H₂₄ tutorial](lassqd.ipynb) uses Fulqrum for the SQD batch solver.
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

## Building the documentation

```bash
uv run --group docs sphinx-build docs docs/_build/html
```

With pip, `python -m pip install --group docs` installs the documentation tools.
mrh is mocked, so the documentation builds without the native dependencies.
