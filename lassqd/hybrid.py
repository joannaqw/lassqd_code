"""The hybrid LASSQD loop and its stages.

One cycle at fixed orbitals:

1. :func:`lassqd.las.capture_embedding` collects the fragment Hamiltonians of a
   :class:`lassqd.records.LASState`, the previous cycle's orbitals and fragment
   RDMs, and finds each fragment's sampling basis.
2. :func:`lassqd.circuits.build_circuits` builds and compiles the fragment
   circuits; :func:`lassqd.circuits.submit_circuits` samples them in one job, and
   :func:`lassqd.circuits.collect_samples` splits the pub results into
   per-fragment counts.
3. :func:`solve_cycle` solves each fragment with SQD on its counts and takes one
   LASSCF orbital step. mrh solves the fragments inside its orbital step, so both
   happen in one call.

:func:`run_lassqd` repeats these until the energy converges. Calling the stages
directly lets you save samples, re-solve them with other SQD settings, or resume
after a classical failure without submitting another job.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pyscf.lib import logger
from qiskit.primitives import BaseSamplerV2
from qiskit.transpiler import PassManager

from lassqd.circuits import (
    CircuitBuilder,
    build_circuits,
    collect_samples,
    submit_circuits,
)
from lassqd.las import LASSCFNoSymm, capture_embedding, set_fragment_kernels
from lassqd.records import FragmentSamples, LASState
from lassqd.sqd import FragmentSQD


@dataclass
class CycleResult:
    """One LASSCF orbital step, from :func:`solve_cycle`.

    Attributes:
        e_tot: Total energy after the orbital step.
        state: Orbitals after the orbital step and the fragment RDMs in their
            active space.
        job_ids: Identifiers of the sampler jobs the fragments were solved on,
            one per circuit.
    """

    e_tot: float
    state: LASState
    job_ids: tuple[str | None, ...]


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


def solve_cycle(
    las: LASSCFNoSymm,
    samples: FragmentSamples,
    solvers: Sequence[FragmentSQD],
) -> CycleResult:
    """Solve each fragment on its samples and take one LASSCF orbital step.

    Each solver gets its fragment's counts and LAS orbitals, then ``las.kernel``
    solves the fragments from ``samples.embedding.state``, the orbitals and
    starting RDMs the samples' circuits were built from, and updates the orbitals. ``las``
    is updated in place and the solvers keep their state, such as
    :class:`lassqd.sqd.FragmentSQD`'s carryover and random stream. To re-solve the
    same samples from the same starting point, pass ``copy.deepcopy(solvers)``.

    ``las.max_cycle_macro`` and ``las.max_cycle_rdmjk`` are set to 1 and 0 for the
    step and restored afterwards, including on failure.

    Args:
        las: RDM-based LASSCF object. Its fragment solvers are replaced by
            ``solvers``.
        samples: Samples from :func:`lassqd.circuits.collect_samples`.
        solvers: One solver per fragment, in fragment order.

    Returns:
        The energy and LAS state after the step.
    """
    embedding = samples.embedding
    state = embedding.state
    saved = las.max_cycle_macro, las.max_cycle_rdmjk
    try:
        las.max_cycle_macro, las.max_cycle_rdmjk = 1, 0
        for solver, counts, fragment in zip(
            solvers, samples.counts, embedding.fragments
        ):
            solver.counts = counts
            solver.las_orbitals = fragment.las_orbitals
            solver.ao_ovlp = embedding.ao_ovlp
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
        job_ids=samples.job_ids,
    )


def run_lassqd(
    las: LASSCFNoSymm,
    state: LASState,
    solvers: Sequence[FragmentSQD],
    sampler: BaseSamplerV2,
    *,
    circuit_builder: CircuitBuilder,
    pass_manager: PassManager | None = None,
    groups: Sequence[Sequence[int]] | None = None,
    shots: int | None = None,
    max_cycles: int = 50,
    conv_tol: float = 1e-5,
    callback: Callable[[int, CycleResult], None] | None = None,
) -> HybridResult:
    """Run hybrid LASSQD cycles from ``state`` until ``|dE| < conv_tol``.

    Each cycle, at fixed orbitals:

    1. quantum: :func:`lassqd.las.capture_embedding` collects the fragment
       problems; :func:`lassqd.circuits.build_circuits` builds
       ``circuit_builder(fragment)`` for each, groups them into circuits and
       transpiles with ``pass_manager``; :func:`lassqd.circuits.submit_circuits`
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
        circuit_builder: Required callable ``(fragment) -> QuantumCircuit``
            preparing a state in the fragment's sampling basis; see
            :func:`lassqd.circuits.build_circuits`.
        pass_manager: Optional pass manager applied before sampling.
        groups: Fragment indices of each circuit, all submitted in one job;
            see :func:`lassqd.circuits.build_circuits`. None glues all
            fragments into one circuit; ``[[0], [1], ...]`` samples each fragment
            as a separate circuit.
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
            capture_embedding(las, state),
            circuit_builder,
            pass_manager=pass_manager,
            groups=groups,
        )
        job = submit_circuits(sampler, circuits, shots=shots)
        job_id = job.job_id()
        job_ids.append(job_id)
        log.note("LASSQD cycle %d: sampler job %s", cycle, job_id)
        samples = collect_samples(circuits, job.result(), job_ids=job_id)
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
