"""The hybrid LASSQD loop and its stages.

One cycle at fixed orbitals ``mo_coeff``:

1. :func:`build_circuits` collects the fragment Hamiltonians at ``mo_coeff``,
   embedded with the previous cycle's fragment RDMs, and builds and compiles the
   fragment circuits, glued into one or kept separate.
2. :func:`submit_circuits` samples them in one job, and :func:`collect_samples`
   splits the job's result into per-fragment counts.
3. :func:`solve_cycle` solves each fragment with SQD on its counts and takes one
   LASSCF orbital step. mrh solves the fragments inside its orbital step, so both
   happen in one call.

:func:`run_lassqd` repeats these until the energy converges. Calling the stages
directly lets you save samples, re-solve them with other SQD settings, or resume
after a classical failure without submitting another job.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from pyscf.lib import logger
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.primitives import BasePrimitiveJob, BaseSamplerV2, PrimitiveResult
from qiskit.transpiler import PassManager

from lassqd.las import LASSCFNoSymm, fragment_hamiltonians, set_fragment_kernels
from lassqd.sqd import FragmentSQD

# ``circuit_builder(h1, h2, norb, nelec) -> circuit``.
# See run_lassqd for the basis contract.
CircuitBuilder = Callable[
    [np.ndarray, np.ndarray, int, tuple[int, int]], QuantumCircuit
]


@dataclass
class FragmentCircuits:
    """One cycle's circuits, from :func:`build_circuits`.

    Attributes:
        mo_coeff: Orbitals the fragment Hamiltonians were built at. Each circuit
            uses the fragment ROHF basis of its Hamiltonian at these orbitals.
        hamiltonians: ``[(h0, h1s, h2), ...]`` in fragment order, from
            :func:`lassqd.las.fragment_hamiltonians`.
        circuits: Measured, compiled circuits to submit in one job: one glued
            circuit, or one per fragment.
        output_registers: ``(pub_index, register_name)`` holding each fragment's
            measurements, in fragment order.
        casdm1frs: Fragment 1-RDMs the Hamiltonians were built from (normally the
            previous cycle's), or None for mrh's initial guess.
        casdm2fr: Fragment 2-RDMs matching ``casdm1frs``.
    """

    mo_coeff: np.ndarray
    hamiltonians: list[tuple[float, np.ndarray, np.ndarray]]
    circuits: list[QuantumCircuit]
    output_registers: list[tuple[int, str]]
    casdm1frs: list[np.ndarray] | None = None
    casdm2fr: list[np.ndarray] | None = None


@dataclass
class FragmentSamples:
    """One cycle's per-fragment measurements, from :func:`collect_samples`.

    Picklable, so it can be saved and solved later with :func:`solve_cycle`.

    Attributes:
        mo_coeff: Orbitals the circuits were built at. :func:`solve_cycle` solves
            at these orbitals and starting RDMs, so the counts and the fragment
            bases match.
        counts: Each fragment's counts, in fragment order. The right half of each
            key holds the alpha occupations.
        job_id: Identifier of the sampler job, if known.
        casdm1frs: Fragment 1-RDMs the circuits' Hamiltonians were built from, or
            None for mrh's initial guess.
        casdm2fr: Fragment 2-RDMs matching ``casdm1frs``.
    """

    mo_coeff: np.ndarray
    counts: list[dict[str, int]]
    job_id: str | None = None
    casdm1frs: list[np.ndarray] | None = None
    casdm2fr: list[np.ndarray] | None = None


@dataclass
class CycleResult:
    """One LASSCF orbital step, from :func:`solve_cycle`.

    ``mo_coeff``, ``casdm1frs`` and ``casdm2fr`` are one consistent LAS wave
    function, as in :class:`HybridResult`.

    Attributes:
        e_tot: Total energy after the orbital step.
        mo_coeff: Orbitals after the orbital step.
        casdm1frs: Per-fragment spin-separated 1-RDMs, shape ``(nroots, 2, n_i, n_i)``.
        casdm2fr: Per-fragment spin-summed 2-RDMs, shape ``(nroots, n_i, n_i, n_i, n_i)``.
        job_id: Identifier of the sampler job the fragments were solved on.
    """

    e_tot: float
    mo_coeff: np.ndarray
    casdm1frs: list[np.ndarray]
    casdm2fr: list[np.ndarray]
    job_id: str | None


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
        job_ids: Sampler job identifier of each cycle.
    """

    converged: bool
    e_tot: float
    mo_coeff: np.ndarray
    casdm1frs: list[np.ndarray]
    casdm2fr: list[np.ndarray]
    e_hist: list[float]
    job_ids: list[str]


def build_circuits(
    las: LASSCFNoSymm,
    mo_coeff: np.ndarray,
    circuit_builder: CircuitBuilder,
    *,
    casdm1frs: Sequence[np.ndarray] | None = None,
    casdm2fr: Sequence[np.ndarray] | None = None,
    pass_manager: PassManager | None = None,
    glue_circuits: bool = True,
) -> FragmentCircuits:
    """Build and compile the fragment circuits at ``mo_coeff``.

    Args:
        las: RDM-based LASSCF object.
        mo_coeff: Orbitals to build the fragment Hamiltonians at.
        circuit_builder: Callable ``(h1, h2, norb, nelec) -> QuantumCircuit``; see
            :func:`run_lassqd`.
        casdm1frs: Fragment 1-RDMs to continue from, normally the previous cycle's
            ``CycleResult.casdm1frs``. They set each fragment's embedding in the
            others. None uses mrh's initial guess, as for a first cycle.
        casdm2fr: Fragment 2-RDMs matching ``casdm1frs``.
        pass_manager: Optional pass manager applied to the measured circuits.
        glue_circuits: Glue the fragments into one circuit with one classical
            register per fragment (default). False keeps one measured circuit per
            fragment.

    Returns:
        The circuits, the Hamiltonians they were built from, and where each
        fragment's measurements will be.
    """
    hamiltonians = fragment_hamiltonians(las, mo_coeff, casdm1frs, casdm2fr)
    circuits = [
        circuit_builder(h1s[0], h2, norb, nelec)
        for (h0, h1s, h2), norb, nelec in zip(
            hamiltonians, las.ncas_sub, las.nelecas_sub
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
    return FragmentCircuits(
        mo_coeff, hamiltonians, circuits, output_registers, casdm1frs, casdm2fr
    )


def submit_circuits(
    sampler: BaseSamplerV2, circuits: FragmentCircuits, *, shots: int | None = None
) -> BasePrimitiveJob:
    """Submit one cycle's circuits to ``sampler`` in one job.

    Keep ``job.job_id()`` to recover a remote result later, and pass
    ``job.result()`` to :func:`collect_samples`.

    Args:
        sampler: SamplerV2 primitive.
        circuits: Circuits from :func:`build_circuits`.
        shots: Shots per fragment; None uses the sampler's default.

    Returns:
        The submitted job.
    """
    return sampler.run(circuits.circuits, shots=shots)


def collect_samples(
    circuits: FragmentCircuits,
    result: PrimitiveResult,
    *,
    job_id: str | None = None,
) -> FragmentSamples:
    """Split a sampler job's result into per-fragment counts.

    Args:
        circuits: The circuits that were submitted.
        result: The job's result, from ``job.result()`` or recovered from a remote
            service by job ID.
        job_id: Identifier of the job, recorded with the samples.

    Returns:
        Each fragment's counts and the orbitals they were sampled at.
    """
    counts = [
        result[pub_index].data[register].get_counts()
        for pub_index, register in circuits.output_registers
    ]
    return FragmentSamples(
        circuits.mo_coeff, counts, job_id, circuits.casdm1frs, circuits.casdm2fr
    )


def solve_cycle(
    las: LASSCFNoSymm,
    samples: FragmentSamples,
    solvers: Sequence[FragmentSQD],
) -> CycleResult:
    """Solve each fragment on its samples and take one LASSCF orbital step.

    Each solver gets its fragment's counts and LAS orbitals, then ``las.kernel``
    solves the fragments at ``samples.mo_coeff``, starting from the same RDMs the
    samples' circuits were built from, and updates the orbitals. ``las``
    is updated in place and the solvers keep their state, such as
    :class:`lassqd.sqd.FragmentSQD`'s carryover and random stream. To re-solve the
    same samples from the same starting point, pass ``copy.deepcopy(solvers)``.

    ``las.max_cycle_macro`` and ``las.max_cycle_rdmjk`` are set to 1 and 0 for the
    step and restored afterwards, including on failure.

    Args:
        las: RDM-based LASSCF object. Its fragment solvers are replaced by
            ``solvers``.
        samples: Samples from :func:`collect_samples`.
        solvers: One solver per fragment, in fragment order.

    Returns:
        The energy, orbitals and fragment RDMs after the step.
    """
    saved = las.max_cycle_macro, las.max_cycle_rdmjk
    try:
        las.max_cycle_macro, las.max_cycle_rdmjk = 1, 0
        ao_ovlp = las._scf.get_ovlp()
        bounds = las.ncore + np.cumsum([0, *las.ncas_sub])
        for solver, counts, start, stop in zip(
            solvers, samples.counts, bounds[:-1], bounds[1:]
        ):
            solver.counts = counts
            solver.las_orbitals = samples.mo_coeff[:, start:stop]
            solver.ao_ovlp = ao_ovlp
        set_fragment_kernels(las, solvers)
        las.kernel(
            samples.mo_coeff, casdm1frs=samples.casdm1frs, casdm2fr=samples.casdm2fr
        )
    finally:
        las.max_cycle_macro, las.max_cycle_rdmjk = saved
    return CycleResult(
        e_tot=las.e_tot,
        mo_coeff=las.mo_coeff,
        casdm1frs=[np.array(dm) for dm in las.casdm1frs],
        casdm2fr=[np.array(dm) for dm in las.casdm2fr],
        job_id=samples.job_id,
    )


def run_lassqd(
    las: LASSCFNoSymm,
    mo_coeff: np.ndarray,
    solvers: Sequence[FragmentSQD],
    sampler: BaseSamplerV2,
    *,
    circuit_builder: CircuitBuilder,
    casdm1frs: Sequence[np.ndarray] | None = None,
    casdm2fr: Sequence[np.ndarray] | None = None,
    pass_manager: PassManager | None = None,
    glue_circuits: bool = True,
    shots: int | None = None,
    max_cycles: int = 50,
    conv_tol: float = 1e-5,
    callback: Callable[[int, LASSCFNoSymm], None] | None = None,
) -> HybridResult:
    """Run hybrid LASSQD cycles from ``mo_coeff`` until ``|dE| < conv_tol``.

    Each cycle, at fixed orbitals:

    1. quantum: :func:`build_circuits` builds ``circuit_builder(h1, h2, norb,
       nelec)`` for every fragment Hamiltonian, optionally glues them into one
       circuit and transpiles with ``pass_manager``; :func:`submit_circuits`
       samples them in one job with ``sampler``;
    2. classical: :func:`solve_cycle` gives each fragment's counts to its solver
       (e.g. :class:`lassqd.sqd.FragmentSQD`) and lets ``las.kernel`` solve the
       fragments and take one orbital step.

    Each cycle continues from the previous cycle's fragment RDMs, which set each
    fragment's embedding in the others, as classical LASSCF does between macro
    iterations. ``las.max_cycle_macro`` and ``las.max_cycle_rdmjk`` are
    overridden during each orbital step and restored afterwards.

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
        casdm1frs: Fragment 1-RDMs to start the first cycle from, e.g. from
            :func:`lassqd.pdft.load_rdms` when restarting at the matching
            ``mo_coeff``. None uses mrh's initial guess.
        casdm2fr: Fragment 2-RDMs matching ``casdm1frs``.
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
        The final energy, orbitals and fragment RDMs, the energy history and the
        sampler job identifiers.
    """
    log = logger.new_logger(las)
    e_hist = []
    job_ids = []
    converged = False
    for cycle in range(max_cycles):
        circuits = build_circuits(
            las,
            mo_coeff,
            circuit_builder,
            casdm1frs=casdm1frs,
            casdm2fr=casdm2fr,
            pass_manager=pass_manager,
            glue_circuits=glue_circuits,
        )
        job = submit_circuits(sampler, circuits, shots=shots)
        job_id = job.job_id()
        job_ids.append(job_id)
        log.note("LASSQD cycle %d: sampler job %s", cycle, job_id)
        samples = collect_samples(circuits, job.result(), job_id=job_id)
        step = solve_cycle(las, samples, solvers)
        mo_coeff, casdm1frs, casdm2fr = step.mo_coeff, step.casdm1frs, step.casdm2fr
        e_hist.append(step.e_tot)
        if len(e_hist) > 1:
            de = e_hist[-1] - e_hist[-2]
            log.note("LASSQD cycle %d: E = %.10f, dE = %.2e", cycle, step.e_tot, de)
            converged = bool(abs(de) < conv_tol)
        else:
            log.note("LASSQD cycle %d: E = %.10f", cycle, step.e_tot)
        if callback is not None:
            callback(cycle, las)
        if converged:
            break
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
        job_ids=job_ids,
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
