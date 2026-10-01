"""Sample-based quantum diagonalization as a LASSCF fragment kernel."""

from inspect import signature
from typing import Any

import numpy as np
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import (
    SCIState,
    diagonalize_fermionic_hamiltonian,
)

from lassqd.basis import fragment_mo_basis, fragment_rohf, from_mo

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


class FragmentSQD:
    """SQD solver for one LAS fragment, callable as a LASSCF fragment kernel.

    ``solver(norb, nelec, h0, h1s, h2) -> (e, dm1s, dm2)``. Set ``solver.counts``
    to the fragment's measured counts before each call (:func:`lassqd.run_lassqd`
    does this).

    Each call rotates the Hamiltonian to the fragment's ROHF orbitals and runs
    up to ``max_iterations`` rounds of self-consistent configuration recovery,
    with ``num_batches`` subsampled batches per round, using the addon's
    :func:`qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`. Recovery
    uses the lowest-energy batch's occupancies from the preceding round. The
    returned energy, RDMs and carryover strings all come from the lowest-energy
    batch across all rounds.

    Only the alpha one-electron Hamiltonian ``h1s[0]`` is used.

    Carryover (enabled by default): determinants whose amplitude
    exceeds the threshold are added to every batch of the next round, and of the
    next call, after being mapped onto that call's orbitals with
    :func:`permute_carryover`. Set ``carryover_threshold=None`` or ``np.inf`` to
    disable carryover.

    After each call, ``e_tot`` includes ``h0``, ``dm1s`` and ``dm2`` use the LAS
    basis, and ``sci_state`` uses the fragment ROHF basis. Record diagnostics with
    the addon's ``callback`` option and save these attributes after the call.

    Args:
        samples_per_batch: Samples drawn for each batch.
        **sqd_options: Options for
            :func:`qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`,
            using the addon's names and defaults, including early stopping and
            carryover. Stored in ``solver.sqd_options`` and editable between calls.
            ``seed`` is converted to a persistent NumPy generator on use, so the
            random stream advances across calls. ``callback(results)`` is passed
            directly to the addon and receives each iteration's list of
            ``SCIResult`` objects, with energies excluding ``h0``. Final solver
            attributes are updated after the addon returns. States,
            ``include_configurations`` and ``initial_occupancies`` use the current
            fragment ROHF basis. Included configurations are merged with carryover
            from previous calls before the addon's ``max_dim`` truncation.
            Configure spin constraints and Davidson settings through ``sci_solver``,
            for example ``functools.partial(solve_sci_batch, spin_sq=2, tol=1e-12)``.
            Missing RDMs are computed from the returned state.
    """

    def __init__(self, samples_per_batch: int, **sqd_options: Any) -> None:
        self.samples_per_batch = samples_per_batch
        self.sqd_options = sqd_options
        self.counts: dict[str, int] | None = None
        self.carryover_strings: tuple[np.ndarray, np.ndarray] | None = None
        self.prev_mo: np.ndarray | None = None

    def __call__(
        self,
        norb: int,
        nelec: tuple[int, int],
        h0: float,
        h1s: np.ndarray,
        h2: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Solve the fragment with SQD on ``self.counts``.

        Args:
            norb: Number of fragment orbitals.
            nelec: ``(neleca, nelecb)``.
            h0: Constant energy term.
            h1s: One-electron integrals in the LAS basis, spin-separated with shape
                ``(2, norb, norb)`` or spin-free with shape ``(norb, norb)``.
            h2: Two-electron integrals in the LAS basis, in chemists' notation,
                shape ``(norb,) * 4``.

        Returns:
            ``(e_tot, dm1s, dm2)``: the fragment energy including ``h0``, the
            spin-separated 1-RDMs of shape ``(2, norb, norb)`` and the spin-summed
            2-RDM of shape ``(norb,) * 4``, both in the LAS basis.

        Raises:
            RuntimeError: If ``self.counts`` has not been set.
        """
        if self.counts is None:
            raise RuntimeError(
                "FragmentSQD.counts must be set before the kernel is called"
            )
        h1 = h1s[0] if np.ndim(h1s) == 3 else h1s
        mo_coeff, h1_mo, h2_mo = fragment_mo_basis(h1, h2, norb, nelec)

        self.sqd_options["seed"] = np.random.default_rng(self.sqd_options.get("seed"))
        options = self.sqd_options.copy()
        carryover_threshold = options.get(
            "carryover_threshold", _DEFAULT_CARRYOVER_THRESHOLD
        )
        if carryover_threshold is None:
            carryover_threshold = np.inf
        options["carryover_threshold"] = carryover_threshold
        use_carryover = carryover_threshold != np.inf
        if use_carryover:
            options["include_configurations"] = self._include_carryover(
                h1_mo, h2_mo, norb, nelec, options.get("include_configurations")
            )

        best = diagonalize_fermionic_hamiltonian(
            h1_mo,
            h2_mo,
            BitArray.from_counts(self.counts, num_bits=2 * norb),
            self.samples_per_batch,
            norb,
            nelec,
            **options,
        )
        if use_carryover:
            self.carryover_strings = self._carryover(
                best.sci_state, carryover_threshold
            )

        self.sci_state = best.sci_state
        self.e_tot = best.energy + h0
        rdm2 = best.rdm2
        if rdm2 is None:
            rdm2 = best.sci_state.rdm(rank=2, spin_summed=True)
        self.dm1s, self.dm2 = from_mo(mo_coeff, best.sci_state.rdm(rank=1), rdm2)
        return self.e_tot, self.dm1s, self.dm2

    def _include_carryover(
        self,
        h1_mo: np.ndarray,
        h2_mo: np.ndarray,
        norb: int,
        nelec: tuple[int, int],
        include: list[int]
        | np.ndarray
        | tuple[list[int] | np.ndarray, list[int] | np.ndarray]
        | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map previous strings to this call's orbitals and merge explicit strings."""
        mo_ref = fragment_rohf(h1_mo, h2_mo, norb, nelec).mo_coeff
        if self.carryover_strings is None:
            empty = np.array([], dtype=np.int64)
            self.carryover_strings = (empty, empty)
        else:
            self.carryover_strings = permute_carryover(
                *self.carryover_strings, mo_ref.T @ self.prev_mo, norb
            )
        self.prev_mo = mo_ref
        if include is None:
            return self.carryover_strings
        include_a, include_b = (
            include if isinstance(include, tuple) else (include, include)
        )
        return (
            np.union1d(include_a, self.carryover_strings[0]),
            np.union1d(include_b, self.carryover_strings[1]),
        )

    def _carryover(
        self, sci_state: SCIState, carryover_threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Alpha and beta strings of the determinants with ``|amplitude| >= carryover_threshold``."""
        amplitudes = np.abs(sci_state.amplitudes)
        alpha_idx, beta_idx = np.nonzero(amplitudes >= carryover_threshold)
        return sci_state.ci_strs_a[alpha_idx], sci_state.ci_strs_b[beta_idx]
