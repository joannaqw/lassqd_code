# LASSQD

[![Tests](https://github.com/joannaqw/lassqd_code/actions/workflows/tests.yml/badge.svg)](https://github.com/joannaqw/lassqd_code/actions/workflows/tests.yml)

Sample-based quantum diagonalization (SQD) as a fragment solver for the localized
active space self-consistent field (LASSCF) method.

LASSQD combines [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) with
[mrh](https://github.com/MatthewRHermes/mrh)'s RDM-based LASSCF
(`mrh.my_pyscf.mcscf.lasscf_rdm`). Both are installed as dependencies.

**Documentation:** <https://joannaqw.github.io/lassqd_code/>. It covers
installation, the H₂₀ tutorial, the user guide and the API reference.

## Quick start

LASSQD needs Python 3.10+, Git, Make, C/C++ and Fortran compilers, OpenMP, and
BLAS/LAPACK development libraries. On Debian/Ubuntu, with
[uv](https://docs.astral.sh/uv/getting-started/installation/), run from the root
of this repository:

```bash
sudo apt-get install git build-essential gfortran libblas-dev liblapack-dev
git clone https://github.com/MatthewRHermes/mrh.git .deps/mrh
git -C .deps/mrh checkout f007765830b62c83aef5839119732a63f5e463e0
uv sync --python 3.14
```

Keep `.deps/mrh`: the environment uses it as an editable installation. See the
[installation guide](https://joannaqw.github.io/lassqd_code/installation.html)
for pip, macOS, the IBM extra, and running the notebook.

## Repository contents

| Path | Contents |
| --- | --- |
| [lassqd/](lassqd/) | LASSQD package |
| [docs/](docs/) | Sphinx documentation, including the [H₂₀ tutorial](docs/lassqd.ipynb) |
| [examples/](examples/README.md) | FeFe calculations with classical sampling, IBM hardware, and SQD-PDFT |
| [tests/](tests/) | pytest tests |
| [pyproject.toml](pyproject.toml) | Package metadata, dependencies, and development tools |
| [uv.lock](uv.lock) | Locked dependency versions for uv |

## Development

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m pytest
uv run --group docs sphinx-build docs docs/_build/html
```

## License

LASSQD is licensed under the [Apache License 2.0](LICENSE). It depends on
[mrh](https://github.com/MatthewRHermes/mrh), which is licensed under the
[GNU General Public License v2.0 or later](https://github.com/MatthewRHermes/mrh/blob/HEAD/LICENSE).
