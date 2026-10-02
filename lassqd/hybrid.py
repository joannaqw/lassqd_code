"""The hybrid LASSQD loop."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from pyscf.lib import logger
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.primitives import BaseSamplerV2
from qiskit.transpiler import PassManager

from lassqd.las import LASSCFNoSymm, fragment_hamiltonians, set_fragment_kernels
from lassqd.sqd import FragmentSQD

# ``circuit_builder(h1, h2, norb, nelec) -> circuit``.
# See run_lassqd for the basis contract.
CircuitBuilder = Callable[
    [np.ndarray, np.ndarray, int, tuple[int, int]], QuantumCircuit
]


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
    circuit_builder: CircuitBuilder,
    pass_manager: PassManager | None = None,
    glue_circuits: bool = True,
    shots: int | None = None,
    max_cycles: int = 50,
    conv_tol: float = 1e-5,
    callback: Callable[[int, LASSCFNoSymm], None] | None = None,
) -> HybridResult:
    """Run hybrid LASSQD cycles from ``mo_coeff`` until ``|dE| < conv_tol``.

    Each cycle, at fixed orbitals:

    1. quantum: build ``circuit_builder(h1, h2, norb, nelec)`` for every fragment
       Hamiltonian, optionally glue them into one circuit, transpile with
       ``pass_manager`` and sample them in one job with ``sampler``;
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
        sampler: SamplerV2 primitive used to sample the circuits.
        circuit_builder: Required callable ``(h1, h2, norb, nelec) -> QuantumCircuit``.
            Receives the fragment integrals in the LAS basis: ``h1`` has shape
            ``(norb, norb)`` and ``h2`` has shape ``(norb,) * 4`` in chemists'
            notation. ``nelec`` is ``(neleca, nelecb)``; orbital and electron
            counts may be NumPy integers. Return an unmeasured ``2 * norb``-qubit
            state-preparation circuit whose occupations refer to the fragment
            ROHF basis from :func:`lassqd.basis.fragment_mo_basis`, matching
            :class:`lassqd.sqd.FragmentSQD`. Alpha orbitals occupy qubits
            ``0..norb-1`` and beta orbitals ``norb..2*norb-1``. The caller chooses
            the ansatz, interaction pairs and initialization settings.
        pass_manager: Optional pass manager applied before sampling.
        glue_circuits: Glue the fragments into one circuit with one classical
            register per fragment (default). False submits each fragment as a
            separate measured circuit in the same sampler job.
        shots: Shots per fragment per cycle; None uses the sampler's default.
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
                circuit_builder(h1s[0], h2, norb, nelec)
                for (h0, h1s, h2), norb, nelec in zip(
                    fragment_hamiltonians(las, mo_coeff), las.ncas_sub, las.nelecas_sub
                )
            ]
            if glue_circuits:
                circuits = [_glue_circuits(circuits)]
                output_registers = [(0, creg.name) for creg in circuits[0].cregs]
            else:
                circuits = [circuit.measure_all(inplace=False) for circuit in circuits]
                output_registers = [
                    (i, circuit.cregs[-1].name) for i, circuit in enumerate(circuits)
                ]
            if pass_manager is not None:
                circuits = [pass_manager.run(circuit) for circuit in circuits]
            job = sampler.run(circuits, shots=shots)
            log.note("LASSQD cycle %d: sampler job %s", cycle, job.job_id())
            results = job.result()
            for solver, (pub_index, register) in zip(solvers, output_registers):
                solver.counts = results[pub_index].data[register].get_counts()
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
    log.note(
        "LASSQD %s after %d cycles",
        ("not converged", "converged")[converged],
        len(e_hist),
    )
    return HybridResult(
        converged=converged,
        e_tot=las.e_tot,
        mo_coeff=mo_coeff,
        casdm1frs=[np.array(dm) for dm in las.casdm1frs],
        casdm2fr=[np.array(dm) for dm in las.casdm2fr],
        e_hist=e_hist,
    )


def _glue_circuits(circuits: Sequence[QuantumCircuit]) -> QuantumCircuit:
    """Place circuits side by side on one register and measure each separately.

    Args:
        circuits: Fragment circuits, in fragment order.

    Returns:
        One circuit whose qubits are the fragment circuits' qubits concatenated in
        order, with one classical register per fragment; ``circuits[i]`` is
        measured into ``cregs[i]``.
    """
    widths = [qc.num_qubits for qc in circuits]
    cregs = [ClassicalRegister(n) for n in widths]
    glued = QuantumCircuit(QuantumRegister(sum(widths)), *cregs)
    start = 0
    for qc, creg, n in zip(circuits, cregs, widths):
        qubits = range(start, start + n)
        glued.append(qc, qubits)
        glued.measure(qubits, creg)
        start += n
    return glued
