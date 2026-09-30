# Distributed LASSQD report

[report.tex](report.tex) describes an HPC architecture for LASSQD using QCSC
Prefect. It starts from the library's stages and records, which already solve
fragments outside mrh and store every stage boundary in HDF5. It then proposes
concurrent fragment workflows, with one MPI allocation per fragment and one
diagonalization batch per node, using SQD's existing collective interface and an
application batch-solver adapter, retaining the allocation across recovery
rounds. It covers rank responsibilities, global orbital stages, QPU
coordination, and durable restarts. It maps the workflow to Prefect tasks and
estimates task counts; defines logical job identity, result acceptance, and
recovery from ambiguous submissions; quantifies memory and `sci_solver`
input/output sizes, with mitigations; and sets out a step-by-step numerical
validation strategy and the process after validation. It outlines an optional
`lassqd.distributed` subpackage, which would contain the collective adapter and
the distributed driver, and an optional mrh optimization, and records the
inspected upstream source revisions. The subpackage is proposed work; the report
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
