# LASSQD

Sample-based quantum diagonalization (SQD) as a fragment solver for the localized
active space self-consistent field (LASSCF) method.

LASSQD combines [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) with
[mrh](https://github.com/MatthewRHermes/mrh)'s RDM-based LASSCF
(`mrh.my_pyscf.mcscf.lasscf_rdm`). Both are installed as dependencies.

Each hybrid cycle starts from a LAS state: orbitals and fragment RDMs. LASSQD
captures each fragment's embedded Hamiltonian from mrh, samples one circuit per
fragment, solves each fragment with SQD on its samples, and then replays the
fragment RDMs into mrh, which takes one LASSCF orbital step. The fragments are
solved outside mrh and independently of each other.

Start with the [tutorial](lassqd.ipynb) to build an H₁₂ system from scratch, run
two H₆ fragments with local ffsim sampling and a Fulqrum batch solver, and
compare the result with classical LASSCF.

```{toctree}
:maxdepth: 2

installation
lassqd
guide
examples
api
```
