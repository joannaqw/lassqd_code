"""The hybrid LASSQD loop."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from pyscf.lib import logger
from qiskit.primitives import BaseSamplerV2
from qiskit.transpiler import PassManager

from lassqd.circuits import CircuitFn, glue_circuits, lucj_circuit
from lassqd.las import LASSCFNoSymm, fragment_hamiltonians, set_fragment_kernels
from lassqd.sqd import FragmentSQD


@dataclass
class HybridResult:
    """Outcome of :func:`run_lassqd`.

    ``mo_coeff``, ``casdm1frs`` and ``casdm2fr`` are one consistent LAS wave
    function (the RDMs are expressed in ``mo_coeff``'s active orbitals), as needed
    for LAS-PDFT; see :func:`lassqd.pdft.lassqd_pdft_energy`.

    Attributes:
        converged: Whether ``|dE| < conv_tol`` was reached within ``max_cycles``.
        e_tot: Total energy after the last cycle.
        mo_coeff: Orbitals after the last cycle.
        casdm1frs: Per-fragment spin-separated 1-RDMs, shape ``(nroots, 2, n_i, n_i)``.
        casdm2fr: Per-fragment spin-summed 2-RDMs, shape ``(nroots, n_i, n_i, n_i, n_i)``.
        e_hist: Total energy after each cycle.
    """

    converged: bool
    e_tot: float
    mo_coeff: np.ndarray
    casdm1frs: list[np.ndarray]
    casdm2fr: list[np.ndarray]
    e_hist: list[float]


def run_lassqd(
    las: LASSCFNoSymm,
    mo_coeff: np.ndarray,
    solvers: Sequence[FragmentSQD],
    sampler: BaseSamplerV2,
    *,
    pass_manager: PassManager | None = None,
    shots: int | None = None,
    circuit_fn: CircuitFn = lucj_circuit,
    max_cycles: int = 50,
    conv_tol: float = 1e-5,
    callback: Callable[[int, LASSCFNoSymm], None] | None = None,
) -> HybridResult:
    """Run hybrid LASSQD cycles from ``mo_coeff`` until ``|dE| < conv_tol``.

    Each cycle, at fixed orbitals:

    1. quantum: build ``circuit_fn(h1, h2, norb, nelec)`` for every fragment
       Hamiltonian, glue them into one circuit, transpile it with ``pass_manager``
       and sample it with ``sampler``;
    2. classical: give each fragment's counts to its solver (e.g.
       :class:`lassqd.sqd.FragmentSQD`) and let ``las.kernel`` solve the fragments
       and take one orbital step.

    ``las.max_cycle_macro`` and ``las.max_cycle_rdmjk`` are overridden during the
    run and restored afterwards.

    Args:
        las: RDM-based LASSCF object. Its fragment solvers are replaced by
            ``solvers``.
        mo_coeff: Initial molecular orbital coefficients.
        solvers: One solver per fragment, in fragment order.
        sampler: SamplerV2 primitive, e.g. ``qiskit_aer.primitives.SamplerV2`` or
            ``qiskit_ibm_runtime.SamplerV2``. Construct an IBM Runtime sampler with
            ``mode=session`` to run every cycle in one session. Each job id is
            logged so a lost result can be retrieved.
        pass_manager: Transpiles the glued circuit before sampling, e.g. from
            :func:`lassqd.circuits.preset_pass_manager`. Aer and IBM Runtime
            samplers need one, since they do not accept ffsim's gates; None samples
            the glued circuit as is.
        shots: Shots per cycle; None uses the sampler's default.
        circuit_fn: Builds the state-preparation circuit for one fragment.
        max_cycles: Maximum number of hybrid cycles.
        conv_tol: Energy convergence threshold between consecutive cycles.
        callback: Called as ``callback(cycle, las)`` after every cycle, e.g. to
            checkpoint ``las.mo_coeff``.

    Returns:
        The final energy, orbitals and fragment RDMs, and the energy history.
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
            circuit = glue_circuits(circuits)
            if pass_manager is not None:
                circuit = pass_manager.run(circuit)
            job = sampler.run([circuit], shots=shots)
            log.note("LASSQD cycle %d: sampler job %s", cycle, job.job_id())
            data = job.result()[0].data
            for solver, creg in zip(solvers, circuit.cregs):
                solver.counts = data[creg.name].get_counts()
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
