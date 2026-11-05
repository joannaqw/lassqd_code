# Examples

The [examples directory](https://github.com/joannaqw/lassqd_code/tree/main/examples)
treats the FeFe complex (6-31G, charge +4) as two Fe fragments with opposite spin
polarization and runs hybrid LASSQD with CCSD-initialized LUCJ circuits. Its
[README](https://github.com/joannaqw/lassqd_code/blob/main/examples/README.md)
describes the inputs, outputs and settings of each script.

- [fefe_lassqd](https://github.com/joannaqw/lassqd_code/tree/main/examples/fefe_lassqd):
  (6e,10o) per fragment, circuits sampled classically with `FfsimSampler`, trim
  SQD, up to 50 hybrid cycles with determinant carryover between cycles.
- [fefe_lassqd_pdft](https://github.com/joannaqw/lassqd_code/tree/main/examples/fefe_lassqd_pdft):
  (6e,5o) per fragment, sampled on IBM Quantum hardware, trim SQD, followed by the
  SQD-PDFT (tPBE) energy.

See [Installation](installation.md) for setup; the IBM example also needs the
`ibm` extra.
