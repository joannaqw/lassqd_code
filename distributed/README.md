# Distributed LASSQD report

[report.tex](report.tex) describes an HPC architecture for LASSQD using QCSC
Prefect. It starts from the library's stages and records, which already solve
fragments outside mrh and store every stage boundary in HDF5. It then proposes
concurrent fragment workflows, with one MPI allocation per fragment and one
screening batch per node. The fragment solve uses trim SQD
(`qiskit_addon_sqd.trim.TrimPolicy`), as in `docs/lassqd.ipynb`, through SQD's
existing collective interface and an application batch-solver adapter. Each
recovery iteration screens disjoint batches across the ranks, and each rank
trims its own results so that only CI strings reach rank 0. Rank 0 then
diagonalizes the merged subspace. The allocation is retained across recovery
iterations. The report covers rank responsibilities, global orbital stages,
QPU coordination through `QCSCSamplerV2` or QCSC's QRMI runtime, and durable
restarts. It maps the workflow to Prefect tasks and
estimates task counts; defines logical job identity, result acceptance, and
recovery from ambiguous submissions; quantifies memory and `sci_solver`
input/output sizes, with mitigations; and sets out a step-by-step numerical
validation strategy and the process after validation. It outlines an optional
`lassqd.distributed` subpackage, which would contain the collective adapter and
the distributed driver, and an optional mrh optimization. It records the
inspected upstream releases: qiskit-addon-sqd 0.14.1, Fulqrum 0.5.0 and QCSC
Prefect 0.3.1. The subpackage is proposed work; the report
does not implement it or claim measured MPI performance.

Build [report.pdf](report.pdf) with a TeX installation containing `latexmk`,
pdfLaTeX, TikZ, and the standard packages used in the preamble:

```sh
make -C distributed
```

Remove auxiliary build files while keeping the PDF:

```sh
make -C distributed clean
```
