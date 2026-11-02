"""The hybrid LASSQD loop and its stages.

One cycle from a :class:`lassqd.records.LASState`, the previous cycle's orbitals
and fragment RDMs:

1. :func:`lassqd.las.capture_embedding` collects the fragment Hamiltonians and
   finds each fragment's sampling basis.
2. :func:`lassqd.circuits.build_circuits` builds and compiles the fragment
   circuits; :func:`lassqd.circuits.submit_circuits` samples them, and
   :func:`lassqd.circuits.collect_samples` splits the pub results into
   per-fragment counts.
3. :func:`solve_fragments` solves each fragment with SQD on its counts, outside
   mrh and independently of the other fragments.
4. :func:`update_orbitals` replays the fragment results into mrh, which takes one
   LASSCF orbital step.

:func:`run_lassqd` repeats these until the energy converges. Calling the stages
directly lets you save samples, re-solve them with other SQD settings, or resume
after a classical failure without submitting another job.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyscf.lib import logger
from qiskit.primitives import BaseSamplerV2
from qiskit.transpiler import PassManager

from lassqd.circuits import (
    CircuitBuilder,
    build_circuits,
    collect_samples,
    submit_circuits,
)
from lassqd.las import LASSCFNoSymm, capture_embedding, replay_orbital_step
from lassqd.records import Embedding, FragmentResult, FragmentSamples, LASState

if TYPE_CHECKING:
    # Only for annotations: replay workers import this module, and SQD sets up
    # MPI when it is imported.
    from lassqd.sqd import FragmentSQD


@dataclass
class CycleResult:
    """One LASSCF orbital step, from :func:`update_orbitals`.

    Attributes:
        e_tot: Total energy after the orbital step.
        state: Orbitals after the orbital step and the fragment RDMs in their
            active space.
        results: The fragment results the step was taken with, in fragment order.
    """

    e_tot: float
    state: LASState
    results: tuple[FragmentResult, ...]


@dataclass
class HybridResult:
    """Outcome of :func:`run_lassqd`.

    Attributes:
        converged: Whether ``|dE| < conv_tol`` was reached within ``max_cycles``.
        e_tot: Total energy after the last cycle.
        state: The LAS wave function after the last cycle, as needed for LAS-PDFT;
            see :func:`lassqd.pdft.lassqd_pdft_energy`.
        results: The last cycle's fragment results.
        e_hist: Total energy after each cycle.
        job_ids: Sampler job identifier of each cycle.
    """

    converged: bool
    e_tot: float
    state: LASState
    results: tuple[FragmentResult, ...]
    e_hist: list[float]
    job_ids: list[str]


def solve_fragments(
    samples: FragmentSamples, solvers: Sequence[FragmentSQD]
) -> list[FragmentResult]:
    """Solve each fragment of ``samples.embedding`` with SQD on its counts.

    Each solver's ``state`` advances; see :meth:`lassqd.sqd.FragmentSQD.solve`.

    Args:
        samples: Samples from :func:`lassqd.circuits.collect_samples`.
        solvers: One solver per fragment, in fragment order.

    Returns:
        One result per fragment, in fragment order.

    Raises:
        ValueError: If the number of solvers differs from the number of fragments.
    """
    embedding = samples.embedding
    if len(solvers) != len(embedding.fragments):
        raise ValueError(
            f"{len(solvers)} solvers for {len(embedding.fragments)} fragments"
        )
    return [
        solver.solve(fragment, counts, ao_ovlp=embedding.ao_ovlp)
        for solver, fragment, counts in zip(
            solvers, embedding.fragments, samples.counts, strict=True
        )
    ]


def update_orbitals(
    las: LASSCFNoSymm, embedding: Embedding, results: Sequence[FragmentResult]
) -> CycleResult:
    """Take one LASSCF orbital step with the fragments of ``embedding`` solved.

    See :func:`lassqd.las.replay_orbital_step`. ``las`` is updated in place.

    Args:
        las: RDM-based LASSCF object, set up like the one the embedding was
            captured with.
        embedding: The fragment problems, from :func:`lassqd.las.capture_embedding`.
        results: One result per fragment, in fragment order.

    Returns:
        The energy and LAS state after the step.

    Raises:
        ValueError: If a result was solved for a different fragment problem.
    """
    if len(results) != len(embedding.fragments):
        raise ValueError(
            f"{len(results)} results for {len(embedding.fragments)} fragments"
        )
    for fragment, result in zip(embedding.fragments, results, strict=True):
        if result.fragment_digest != fragment.digest():
            raise ValueError(
                f"result {fragment.index} was solved for a different fragment problem"
            )
    e_tot, state = replay_orbital_step(las, embedding, results)
    return CycleResult(e_tot=e_tot, state=state, results=tuple(results))


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
    2. classical: :func:`solve_fragments` solves each fragment with its solver,
       and :func:`update_orbitals` lets mrh take one orbital step with the
       results.

    Each cycle continues from the previous cycle's fragment RDMs, which set each
    fragment's embedding in the others, as classical LASSCF does between macro
    iterations.

    Args:
        las: RDM-based LASSCF object. Its settings and fragment solvers are
            restored after each step.
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
        results = solve_fragments(samples, solvers)
        step = update_orbitals(las, samples.embedding, results)
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
        results=step.results,
        e_hist=e_hist,
        job_ids=job_ids,
    )
