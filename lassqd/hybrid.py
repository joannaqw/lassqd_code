"""The hybrid LASSQD loop."""

from dataclasses import dataclass

import numpy as np
from pyscf.lib import logger

from lassqd.circuits import glue_circuits, lucj_circuit
from lassqd.las import fragment_hamiltonians, set_fragment_kernels


@dataclass
class HybridResult:
    """Outcome of :func:`run_lassqd`.

    ``mo_coeff``, ``casdm1frs`` and ``casdm2fr`` are one consistent LAS wave
    function (the RDMs are expressed in ``mo_coeff``'s active orbitals), as needed
    for LAS-PDFT; see :func:`lassqd.pdft.lassqd_pdft_energy`.
    """

    converged: bool
    e_tot: float
    mo_coeff: np.ndarray
    casdm1frs: list
    casdm2fr: list
    e_hist: list


def run_lassqd(
    las,
    mo_coeff,
    solvers,
    sampler,
    *,
    circuit_fn=lucj_circuit,
    max_cycles=50,
    conv_tol=1e-5,
    callback=None,
):
    """Run hybrid LASSQD cycles from ``mo_coeff`` until ``|dE| < conv_tol``.

    Each cycle, at fixed orbitals:

    1. quantum: build ``circuit_fn(h1, h2, norb, nelec)`` for every fragment
       Hamiltonian, glue them into one circuit and ``sampler`` it;
    2. classical: give each fragment's counts to its solver (e.g.
       :class:`lassqd.sqd.FragmentSQD`) and let ``las.kernel`` solve the fragments
       and take one orbital step.

    ``callback(cycle, las)`` is called after every cycle, e.g. to checkpoint
    ``las.mo_coeff``.
    """
    log = logger.new_logger(las)
    saved = las.max_cycle_macro, las.max_cycle_rdmjk
    las.max_cycle_macro, las.max_cycle_rdmjk = 1, 0
    e_hist = []
    converged = False
    try:
        for cycle in range(max_cycles):
            circuits = [
                circuit_fn(h1s[0], h2, norb, nelec)
                for (h0, h1s, h2), norb, nelec in zip(
                    fragment_hamiltonians(las, mo_coeff), las.ncas_sub, las.nelecas_sub
                )
            ]
            for solver, counts in zip(solvers, sampler(glue_circuits(circuits))):
                solver.counts = counts
            set_fragment_kernels(las, solvers)
            las.kernel(mo_coeff)
            mo_coeff = las.mo_coeff
            e_hist.append(las.e_tot)
            if len(e_hist) > 1:
                de = e_hist[-1] - e_hist[-2]
                log.note("LASSQD cycle %d: E = %.10f, dE = %.2e", cycle, las.e_tot, de)
                converged = bool(abs(de) < conv_tol)
            else:
                log.note("LASSQD cycle %d: E = %.10f", cycle, las.e_tot)
            if callback is not None:
                callback(cycle, las)
            if converged:
                break
    finally:
        las.max_cycle_macro, las.max_cycle_rdmjk = saved
    log.note("LASSQD %s after %d cycles", ("not converged", "converged")[converged], len(e_hist))
    return HybridResult(
        converged=converged,
        e_tot=las.e_tot,
        mo_coeff=mo_coeff,
        casdm1frs=[np.array(dm) for dm in las.casdm1frs],
        casdm2fr=[np.array(dm) for dm in las.casdm2fr],
        e_hist=e_hist,
    )
