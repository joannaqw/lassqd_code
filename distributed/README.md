# Distributed LASSQD report

[report.tex](report.tex) describes an HPC architecture for LASSQD using QCSC
Prefect: concurrent fragment workflows, with one MPI allocation per fragment
and one diagonalization batch per node. It uses SQD's existing collective
interface and an application batch-solver adapter, retaining the allocation
across recovery rounds. It covers rank responsibilities, gathered CI-state
memory limits, global orbital stages, QPU coordination, and durable restarts.
It outlines implementation within the existing LASSQD package and an
optional mrh optimization, and records the inspected source revisions. The collective
adapter and distributed driver are proposed application work; the report does
not implement them or claim measured MPI performance.

Build [report.pdf](report.pdf) with a TeX installation containing `latexmk`,
pdfLaTeX, TikZ, and the standard packages used in the preamble:

```sh
make -C distributed
```

Remove auxiliary build files while keeping the PDF:

```sh
make -C distributed clean
```
