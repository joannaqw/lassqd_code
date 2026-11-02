"""The hybrid LASSQD loop and its stages.

One cycle at fixed orbitals:

1. :func:`build_circuits` collects the fragment Hamiltonians of a
   :class:`lassqd.records.LASState`, the previous cycle's orbitals and fragment
   RDMs, and builds and compiles the fragment circuits, glued into one or kept
   separate.
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
from lassqd.records import LASState
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
        state: Orbitals and starting RDMs the fragment Hamiltonians were built
            from. Each circuit uses the fragment ROHF basis of its Hamiltonian.
        hamiltonians: ``[(h0, h1s, h2), ...]`` in fragment order, from
            :func:`lassqd.las.fragment_hamiltonians`.
        circuits: Measured, compiled circuits to submit in one job: one glued
            circuit, or one per fragment.
        output_registers: ``(pub_index, register_name)`` holding each fragment's
            measurements, in fragment order.
    """

    state: LASState
    hamiltonians: list[tuple[float, np.ndarray, np.ndarray]]
    circuits: list[QuantumCircuit]
    output_registers: list[tuple[int, str]]


@dataclass
class FragmentSamples:
    """One cycle's per-fragment measurements, from :func:`collect_samples`.

    Picklable, so it can be saved and solved later with :func:`solve_cycle`.

    Attributes:
        state: Orbitals and starting RDMs the circuits were built from.
            :func:`solve_cycle` solves from this state, so the counts and the
            fragment bases match.
        counts: Each fragment's counts, in fragment order. The right half of each
            key holds the alpha occupations.
        job_id: Identifier of the sampler job, if known.
    """

    state: LASState
    counts: list[dict[str, int]]
    job_id: str | None = None


@dataclass
class CycleResult:
    """One LASSCF orbital step, from :func:`solve_cycle`.

    Attributes:
        e_tot: Total energy after the orbital step.
        state: Orbitals after the orbital step and the fragment RDMs in their
            active space.
        job_id: Identifier of the sampler job the fragments were solved on.
    """

    e_tot: float
    state: LASState
    job_id: str | None


@dataclass
class HybridResult:
    """Outcome of :func:`run_lassqd`.

    Attributes:
        converged: Whether ``|dE| < conv_tol`` was reached within ``max_cycles``.
        e_tot: Total energy after the last cycle.
        state: The LAS wave function after the last cycle, as needed for LAS-PDFT;
            see :func:`lassqd.pdft.lassqd_pdft_energy`.
        e_hist: Total energy after each cycle.
        job_ids: Sampler job identifier of each cycle.
    """

    converged: bool
    e_tot: float
    state: LASState
    e_hist: list[float]
    job_ids: list[str]


def build_circuits(
    las: LASSCFNoSymm,
    state: LASState,
    circuit_builder: CircuitBuilder,
    *,
    pass_manager: PassManager | None = None,
    glue_circuits: bool = True,
) -> FragmentCircuits:
    """Build and compile the fragment circuits for ``state``.

    Args:
        las: RDM-based LASSCF object.
        state: Orbitals to build the fragment Hamiltonians at, and the fragment
            RDMs that set each fragment's embedding in the others: normally the
            previous cycle's ``CycleResult.state``, or
            :func:`lassqd.las.initial_state` for a first cycle.
        circuit_builder: Callable ``(h1, h2, norb, nelec) -> QuantumCircuit``; see
            :func:`run_lassqd`.
        pass_manager: Optional pass manager applied to the measured circuits.
        glue_circuits: Glue the fragments into one circuit with one classical
            register per fragment (default). False keeps one measured circuit per
            fragment.

    Returns:
        The circuits, the Hamiltonians they were built from, and where each
        fragment's measurements will be.
    """
    hamiltonians = fragment_hamiltonians(las, state)
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
    return FragmentCircuits(state, hamiltonians, circuits, output_registers)


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
        Each fragment's counts and the state they were sampled from.
    """
    counts = [
        result[pub_index].data[register].get_counts()
        for pub_index, register in circuits.output_registers
    ]
    return FragmentSamples(circuits.state, counts, job_id)


def solve_cycle(
    las: LASSCFNoSymm,
    samples: FragmentSamples,
    solvers: Sequence[FragmentSQD],
) -> CycleResult:
    """Solve each fragment on its samples and take one LASSCF orbital step.

    Each solver gets its fragment's counts and LAS orbitals, then ``las.kernel``
    solves the fragments from ``samples.state``, the orbitals and starting RDMs
    the samples' circuits were built from, and updates the orbitals. ``las``
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
        The energy and LAS state after the step.
    """
    state = samples.state
    saved = las.max_cycle_macro, las.max_cycle_rdmjk
    try:
        las.max_cycle_macro, las.max_cycle_rdmjk = 1, 0
        ao_ovlp = las._scf.get_ovlp()
        bounds = las.ncore + np.cumsum([0, *las.ncas_sub])
        for solver, counts, start, stop in zip(
            solvers, samples.counts, bounds[:-1], bounds[1:]
        ):
            solver.counts = counts
            solver.las_orbitals = state.mo_coeff[:, start:stop]
            solver.ao_ovlp = ao_ovlp
        set_fragment_kernels(las, solvers)
        las.kernel(
            state.mo_coeff,
            casdm1frs=list(state.casdm1frs),
            casdm2fr=list(state.casdm2fr),
        )
    finally:
        las.max_cycle_macro, las.max_cycle_rdmjk = saved
    return CycleResult(
        e_tot=las.e_tot,
        state=LASState(las.mo_coeff, tuple(las.casdm1frs), tuple(las.casdm2fr)),
        job_id=samples.job_id,
    )


def run_lassqd(
    las: LASSCFNoSymm,
    state: LASState,
    solvers: Sequence[FragmentSQD],
    sampler: BaseSamplerV2,
    *,
    circuit_builder: CircuitBuilder,
    pass_manager: PassManager | None = None,
    glue_circuits: bool = True,
    shots: int | None = None,
    max_cycles: int = 50,
    conv_tol: float = 1e-5,
    callback: Callable[[int, CycleResult], None] | None = None,
) -> HybridResult:
    """Run hybrid LASSQD cycles from ``state`` until ``|dE| < conv_tol``.

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
        state: Initial orbitals and fragment RDMs: :func:`lassqd.las.initial_state`
            for a new calculation, or a saved :class:`lassqd.records.LASState` to
            restart from.
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
        max_cycles: Maximum number of hybrid cycles, at least 1.
        conv_tol: Energy convergence threshold between consecutive cycles.
        callback: Called as ``callback(cycle, step)`` with each cycle's
            :class:`CycleResult`, e.g. to checkpoint ``step.state``.

    Returns:
        The final energy and LAS state, the energy history and the sampler job
        identifiers.
    """
    if max_cycles < 1:
        raise ValueError(f"max_cycles must be at least 1, got {max_cycles}")
    log = logger.new_logger(las)
    e_hist = []
    job_ids = []
    converged = False
    for cycle in range(max_cycles):
        circuits = build_circuits(
            las,
            state,
            circuit_builder,
            pass_manager=pass_manager,
            glue_circuits=glue_circuits,
        )
        job = submit_circuits(sampler, circuits, shots=shots)
        job_id = job.job_id()
        job_ids.append(job_id)
        log.note("LASSQD cycle %d: sampler job %s", cycle, job_id)
        samples = collect_samples(circuits, job.result(), job_id=job_id)
        step = solve_cycle(las, samples, solvers)
        state = step.state
        e_hist.append(step.e_tot)
        if len(e_hist) > 1:
            de = e_hist[-1] - e_hist[-2]
            log.note("LASSQD cycle %d: E = %.10f, dE = %.2e", cycle, step.e_tot, de)
            converged = bool(abs(de) < conv_tol)
        else:
            log.note("LASSQD cycle %d: E = %.10f", cycle, step.e_tot)
        if callback is not None:
            callback(cycle, step)
        if converged:
            break
    log.note(
        "LASSQD %s after %d cycles",
        ("not converged", "converged")[converged],
        len(e_hist),
    )
    return HybridResult(
        converged=converged,
        e_tot=step.e_tot,
        state=state,
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
