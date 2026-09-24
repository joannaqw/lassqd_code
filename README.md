LASSQD: a code to run Sample-based quantum diagonalization as parallel fragment solver for the localized active space self-consistent field method

SQD is provided by the open-source [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd)
package (see `environment.yml`); this repo no longer vendors its own copy.

LASSCF is provided by [mrh](https://github.com/MatthewRHermes/mrh), installed as a package;
this repo no longer vendors a copy of it either. `lassqd/las.py` is the interface to mrh's
RDM-based LASSCF (`mrh.my_pyscf.mcscf.lasscf_rdm`).

contains:
- `lassqd/`: the LASSQD package
- `examples/`: FeFe LASSQD calculations on a classical simulator and on IBM hardware, and SQD-PDFT (see `examples/README.md`)
- `tests/`: pytest tests
- environment file

## The `lassqd` package

Each hybrid cycle runs at fixed orbitals: the fragment Hamiltonians are turned into
circuits, all fragment circuits are glued into one job and sampled, and then SQD on
each fragment's counts is the fragment solver for one LASSCF orbital step.

- `lassqd.las`: `LASSCFNoSymm`, `fragment_hamiltonians`, `set_fragment_kernels` (mrh interface)
- `lassqd.circuits`: `lucj_circuit` (CCSD-initialized, linear-method-optimized LUCJ), `glue_circuits`, `cut_counts`
- `lassqd.samplers`: `aer_sampler`, `ibm_runtime_sampler`; a sampler maps a glued circuit to one counts dict per fragment
- `lassqd.sqd`: `FragmentSQD`, the SQD fragment kernel (configuration recovery, optional determinant carryover between cycles), and `solve_sci_nroots`
- `lassqd.hybrid`: `run_lassqd`, the hybrid loop
- `lassqd.pdft`: `lassqd_pdft_energy` (LAS-PDFT on the SQD RDMs), `save_rdms`, `load_rdms`

```python
from lassqd import FragmentSQD, LASSCFNoSymm, aer_sampler, lassqd_pdft_energy, run_lassqd

las = LASSCFNoSymm(mf, (5, 5), ((4, 2), (2, 4)), spin_sub=(3, 3))
solvers = [FragmentSQD(iterations=6, n_batches=15, samples_per_batch=50) for _ in range(las.nfrags)]
result = run_lassqd(las, mo_coeff, solvers, aer_sampler(shots=100_000))
e_pdft = lassqd_pdft_energy(las, result.casdm1frs, result.casdm2fr, result.mo_coeff)
```

`FragmentSQD` returns the energy and RDMs of the lowest-energy batch of the last
configuration-recovery iteration, and SQD sees only the alpha one-electron
Hamiltonian `h1s[0]`. `result.mo_coeff` with `result.casdm1frs`/`result.casdm2fr` is
one consistent LAS wave function (RDMs in those orbitals' active space), which is
what LAS-PDFT needs.

## Installation

pyscf-forge and mrh both compile against an installed pyscf, so install in this order.
Tested with pyscf 2.14.0, pyscf-forge `f55abdb`, and mrh `f0077658`.

```bash
conda env create -f environment.yml
conda activate lassqd

# pyscf-forge (mrh's CSF solver now lives here, so LASSCF needs it)
pip install --no-build-isolation \
  "pyscf-forge @ git+https://github.com/pyscf/pyscf-forge.git@f55abdb2806697c4132c416a43ddebadf93e3b1a"

# mrh: the clone must be in a directory named `mrh`
git clone https://github.com/MatthewRHermes/mrh.git
cd mrh && git checkout f0077658
pip install --no-deps -e .
cd lib && mkdir build && cd build
cmake -DPYSCFLIB="$(python -c 'import os, pyscf.lib; print(os.path.dirname(pyscf.lib.__file__))')" ..
make -j8
cd ../../..

# lassqd, from the root of this repo
pip install --no-deps -e .
```

mrh cannot currently be installed with `pip install git+https://...`, and `pip install -e`
does not compile its C libraries, which is why they are built with CMake above. On macOS,
see mrh's README for compiler and OpenMP setup.

## Tests

```bash
pip install pytest
pytest
```
