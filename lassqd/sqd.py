"""Sample-based quantum diagonalization of LAS fragments.

:class:`FragmentSQD` solves a :class:`lassqd.records.Fragment` on its measured
counts in three parts, so that an MPI job can run them on the right ranks:

1. :meth:`FragmentSQD.prepare` (rank 0) maps carryover onto the fragment's
   sampling basis and resolves the SQD arguments into :class:`SQDInputs`;
2. :meth:`SQDInputs.run` calls
   :func:`qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`, which every
   rank of a collective solve enters with equal inputs;
3. :meth:`FragmentSQD.finalize` (rank 0) rotates the RDMs to the LAS basis and
   records the next :class:`lassqd.records.SQDState`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import signature
from typing import Any

import numpy as np
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import (
    SCIResult,
    SCIState,
    diagonalize_fermionic_hamiltonian,
)

from lassqd.records import Fragment, FragmentResult, SQDState

_DEFAULT_CARRYOVER_THRESHOLD = (
    signature(diagonalize_fermionic_hamiltonian)
    .parameters["carryover_threshold"]
    .default
)


def permute_carryover(
    strings_a: np.ndarray, strings_b: np.ndarray, overlap: np.ndarray, norb: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map carried-over determinant strings onto a new orbital basis.

    Each new orbital takes the occupation of the old orbital it overlaps most.

    Args:
        strings_a: Alpha determinant strings in the old basis, as integers.
        strings_b: Beta determinant strings in the old basis, as integers.
        overlap: ``overlap[p, q]`` is the overlap of new orbital ``p`` with old
            orbital ``q``, shape ``(norb, norb)``.
        norb: Number of orbitals.

    Returns:
        Unique ``(strings_a, strings_b)`` in the new basis.
    """
    rho = np.argmax(np.abs(overlap), axis=1)
    # Bit p of a new string is bit rho[p] of the old string.
    weights = 1 << np.arange(norb)
    return tuple(
        np.unique(((strings[:, None] >> rho) & 1) @ weights)
        for strings in (strings_a, strings_b)
    )


@dataclass
class SQDInputs:
    """The arguments of one SQD call, from :meth:`FragmentSQD.prepare`.

    Attributes:
        one_body_tensor: Alpha one-electron integrals in the sampling basis.
        two_body_tensor: Two-electron integrals in the sampling basis.
        bit_array: The fragment's measurements.
        samples_per_batch: Samples drawn for each batch.
        norb: Number of fragment orbitals.
        nelec: ``(neleca, nelecb)``.
        options: Keyword arguments of
            :func:`qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`,
            with carryover merged into ``include_configurations`` and ``seed`` set
            to the generator for this solve.
        state: The solver state the inputs were prepared from.
        carryover_orbitals: AO coefficients of the sampling basis, recorded with
            the new carryover strings; None if carryover is disabled.
    """

    one_body_tensor: np.ndarray
    two_body_tensor: np.ndarray
    bit_array: BitArray
    samples_per_batch: int
    norb: int
    nelec: tuple[int, int]
    options: dict[str, Any]
    state: SQDState
    carryover_orbitals: np.ndarray | None

    @property
    def rng(self) -> np.random.Generator:
        """The generator SQD draws from; it advances during :meth:`run`."""
        return self.options["seed"]

    def run(self) -> SCIResult:
        """Run configuration recovery and diagonalization; returns the best batch."""
        return diagonalize_fermionic_hamiltonian(
            self.one_body_tensor,
            self.two_body_tensor,
            self.bit_array,
            self.samples_per_batch,
            self.norb,
            self.nelec,
            **self.options,
        )


class FragmentSQD:
    """SQD solver for one LAS fragment.

    Each solve runs up to ``max_iterations`` rounds of self-consistent
    configuration recovery in the fragment's sampling basis, with
    ``num_batches`` subsampled batches per round, using the addon's
    :func:`qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`. Recovery
    uses the lowest-energy batch's occupancies from the preceding round. The
    returned energy, RDMs and carryover strings all come from the lowest-energy
    batch across all rounds. Only the alpha one-electron Hamiltonian is used.

    Carryover (enabled by default): determinants whose amplitude exceeds the
    threshold are added to every batch of the next round, and of the next solve,
    after being mapped onto that solve's sampling basis with
    :func:`permute_carryover`. The bases are compared in the AO basis, using the
    fragment's LAS orbitals and the AO overlap. Set ``carryover_threshold=None``
    or ``np.inf`` to disable carryover.

    The solver's continuation state, the random generator and the carryover
    strings with their basis, is ``solver.state``, a
    :class:`lassqd.records.SQDState`. :meth:`solve` replaces it only when a solve
    succeeds; assign an earlier state to repeat a solve.

    Args:
        samples_per_batch: Samples drawn for each batch.
        seed: Seed of the random generator, or a generator whose current state is
            copied. Each solve continues the stream where the previous one ended.
        sci_solver: Batch solver passed to the addon, accepting
            ``(ci_strings, h1, h2, norb, nelec)`` and returning a list of
            ``SCIResult``; None uses the addon's ``solve_sci_batch``. Configure spin
            constraints and Davidson settings here, for example
            ``functools.partial(solve_sci_batch, spin_sq=2, tol=1e-12)``.
        callback: ``callback(results)`` passed to the addon, receiving each
            iteration's list of ``SCIResult`` objects, with energies excluding
            ``h0`` and states in the sampling basis.
        **options: Numerical options of
            :func:`qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`,
            using the addon's names and defaults, including early stopping and
            carryover. Stored in ``solver.options`` and editable between solves.
            ``include_configurations`` and ``initial_occupancies`` use the
            current sampling basis. Included configurations are merged with
            carryover before the addon's ``max_dim`` truncation.
    """

    def __init__(
        self,
        samples_per_batch: int,
        *,
        seed: int | np.random.Generator | None = None,
        sci_solver: Callable[..., list[SCIResult]] | None = None,
        callback: Callable[[list[SCIResult]], None] | None = None,
        **options: Any,
    ) -> None:
        self.samples_per_batch = samples_per_batch
        self.sci_solver = sci_solver
        self.callback = callback
        self.options = options
        self.state = SQDState.from_seed(seed)

    def solve(
        self,
        fragment: Fragment,
        counts: Mapping[str, int],
        *,
        ao_ovlp: np.ndarray | None = None,
    ) -> FragmentResult:
        """Solve ``fragment`` on ``counts`` and advance ``self.state``.

        Args:
            fragment: The fragment problem.
            counts: The fragment's measured counts. The right half of each key
                holds the alpha occupations.
            ao_ovlp: AO overlap matrix, used to map carryover between bases;
                None means an orthonormal AO basis.

        Returns:
            The fragment energy and LAS-basis RDMs of the lowest-energy batch.
        """
        result, self.state = self.step(fragment, counts, self.state, ao_ovlp=ao_ovlp)
        return result

    def step(
        self,
        fragment: Fragment,
        counts: Mapping[str, int],
        state: SQDState,
        *,
        ao_ovlp: np.ndarray | None = None,
    ) -> tuple[FragmentResult, SQDState]:
        """Solve ``fragment`` from ``state`` without changing ``self.state``.

        Returns:
            The result and the state for the next solve.
        """
        inputs = self.prepare(fragment, counts, state, ao_ovlp=ao_ovlp)
        return self.finalize(fragment, inputs, inputs.run())

    def prepare(
        self,
        fragment: Fragment,
        counts: Mapping[str, int],
        state: SQDState,
        *,
        ao_ovlp: np.ndarray | None = None,
    ) -> SQDInputs:
        """Resolve the SQD arguments for ``fragment``: the rank-0 setup of a solve.

        Maps the carryover strings of ``state`` onto the fragment's sampling
        basis, merges them with ``include_configurations``, and creates the
        generator from ``state``.
        """
        norb = fragment.norb
        options = dict(self.options)
        carryover_threshold = options.get(
            "carryover_threshold", _DEFAULT_CARRYOVER_THRESHOLD
        )
        if carryover_threshold is None:
            carryover_threshold = np.inf
        options["carryover_threshold"] = carryover_threshold
        carryover_orbitals = None
        if carryover_threshold != np.inf:
            carryover_orbitals = fragment.sampling_orbitals
            options["include_configurations"] = _include_carryover(
                state,
                carryover_orbitals,
                np.eye(len(carryover_orbitals)) if ao_ovlp is None else ao_ovlp,
                norb,
                options.get("include_configurations"),
            )
        options["sci_solver"] = self.sci_solver
        options["callback"] = self.callback
        options["seed"] = state.rng()
        return SQDInputs(
            one_body_tensor=fragment.h1,
            two_body_tensor=fragment.h2,
            bit_array=BitArray.from_counts(counts, num_bits=2 * norb),
            samples_per_batch=self.samples_per_batch,
            norb=norb,
            nelec=fragment.nelec,
            options=options,
            state=state,
            carryover_orbitals=carryover_orbitals,
        )

    def finalize(
        self, fragment: Fragment, inputs: SQDInputs, best: SCIResult
    ) -> tuple[FragmentResult, SQDState]:
        """Build the result and next state from SQD's best batch: rank-0 work.

        Returns:
            The fragment energy including ``h0`` with LAS-basis RDMs, and the state
            for the next solve.
        """
        carryover_threshold = inputs.options["carryover_threshold"]
        if carryover_threshold != np.inf:
            carryover_strings = _carryover(best.sci_state, carryover_threshold)
            carryover_orbitals = inputs.carryover_orbitals
        else:
            carryover_strings = inputs.state.carryover_strings
            carryover_orbitals = inputs.state.carryover_orbitals
        rdm2 = best.rdm2
        if rdm2 is None:
            rdm2 = best.sci_state.rdm(rank=2, spin_summed=True)
        dm1s, dm2 = fragment.rdms_to_las(best.sci_state.rdm(rank=1), rdm2)
        sci_state = best.sci_state
        result = FragmentResult(
            e_tot=best.energy + fragment.h0,
            dm1s=dm1s,
            dm2=dm2,
            amplitudes=sci_state.amplitudes,
            ci_strs_a=sci_state.ci_strs_a,
            ci_strs_b=sci_state.ci_strs_b,
            norb=fragment.norb,
            nelec=fragment.nelec,
            fragment_digest=fragment.digest(),
        )
        state = SQDState(
            inputs.rng.bit_generator.state, carryover_strings, carryover_orbitals
        )
        return result, state


def _include_carryover(
    state: SQDState,
    orbitals: np.ndarray,
    ao_ovlp: np.ndarray,
    norb: int,
    include: list[int]
    | np.ndarray
    | tuple[list[int] | np.ndarray, list[int] | np.ndarray]
    | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map the carried strings onto ``orbitals`` and merge explicit strings."""
    # Compare in the AO basis: LASSCF re-canonicalizes the LAS orbitals between
    # solves.
    if state.carryover_strings is None:
        empty = np.array([], dtype=np.int64)
        strings = (empty, empty)
    else:
        strings = permute_carryover(
            *state.carryover_strings,
            orbitals.T @ ao_ovlp @ state.carryover_orbitals,
            norb,
        )
    if include is None:
        return strings
    include_a, include_b = include if isinstance(include, tuple) else (include, include)
    return np.union1d(include_a, strings[0]), np.union1d(include_b, strings[1])


def _carryover(
    sci_state: SCIState, carryover_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Alpha and beta strings of the determinants with ``|amplitude| >= carryover_threshold``."""
    amplitudes = np.abs(sci_state.amplitudes)
    alpha_idx, beta_idx = np.nonzero(amplitudes >= carryover_threshold)
    return sci_state.ci_strs_a[alpha_idx], sci_state.ci_strs_b[beta_idx]
