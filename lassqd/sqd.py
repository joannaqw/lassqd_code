"""Sample-based quantum diagonalization as a LASSCF fragment kernel."""

import os
import sys
from inspect import signature
from pathlib import Path
from typing import Any

import numpy as np
from pyscf.lib import logger
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import (
    bitstring_matrix_to_ci_strs,
    diagonalize_fermionic_hamiltonian,
)

from lassqd.basis import fragment_mo_basis, fragment_rohf, from_mo


def _strings_to_bitstrings(strings, norb):
    """Integer determinant strings -> boolean rows, most significant orbital first."""
    return ((strings[:, None] >> np.arange(norb)[::-1]) & 1).astype(bool)


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
    # Bitstring columns run from orbital norb-1 down to 0; flip to index by orbital.
    alpha = _strings_to_bitstrings(strings_a, norb)[:, ::-1][:, rho][:, ::-1]
    beta = _strings_to_bitstrings(strings_b, norb)[:, ::-1][:, rho][:, ::-1]
    # bitstring_matrix_to_ci_strs returns (right half, left half) == (alpha, beta)
    return bitstring_matrix_to_ci_strs(np.hstack([beta, alpha]), open_shell=True)


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

    Attributes set by each call: ``e_hist``, ``d_hist`` (subspace dimension),
    ``a_hist``/``b_hist`` (alpha/beta string counts), ``s_hist`` (<S^2>), all of
    shape ``(completed_iterations, num_batches)``; ``occupancy_hist`` (the lowest-energy
    batch's occupancies per round, alpha then beta, shape
    ``(completed_iterations, 2 * norb)``); ``e_tot``; ``dm1s`` and ``dm2`` in the
    LAS basis; ``sci_state`` in the fragment ROHF basis.

    Args:
        samples_per_batch: Samples drawn for each batch.
        output_dir: If given, the final state, histories and carryover strings are
            saved here after every call.
        verbose: pyscf logger verbosity.
        **sqd_options: Options for
            :func:`qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`,
            using the addon's names and defaults, including early stopping and
            carryover. Stored in ``solver.sqd_options`` and editable between calls.
            ``seed`` is converted to a persistent NumPy generator on use, so the
            random stream advances across calls. ``callback`` runs after internal
            histories are recorded; its energies exclude ``h0``. States,
            ``include_configurations`` and ``initial_occupancies`` use the current
            fragment ROHF basis. Included configurations are merged with carryover
            from previous calls before the addon's ``max_dim`` truncation.
            Configure spin constraints and Davidson settings through ``sci_solver``,
            for example ``functools.partial(solve_sci_batch, spin_sq=2, tol=1e-12)``.
            Missing RDMs are computed from the returned state.
    """

    def __init__(
        self,
        samples_per_batch: int,
        *,
        output_dir: str | os.PathLike | None = None,
        verbose: int = logger.INFO,
        **sqd_options: Any,
    ) -> None:
        self.samples_per_batch = samples_per_batch
        self.sqd_options = sqd_options
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.verbose = verbose
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
        log = logger.Logger(sys.stdout, self.verbose)
        if self.counts is None:
            raise RuntimeError(
                "FragmentSQD.counts must be set before the kernel is called"
            )
        h1 = h1s[0] if np.ndim(h1s) == 3 else h1s
        mo_coeff, h1_mo, h2_mo = fragment_mo_basis(h1, h2, norb, nelec)

        self.sqd_options["seed"] = np.random.default_rng(self.sqd_options.get("seed"))
        options = self.sqd_options.copy()
        carryover_threshold = options.get(
            "carryover_threshold",
            signature(diagonalize_fermionic_hamiltonian)
            .parameters["carryover_threshold"]
            .default,
        )
        if carryover_threshold is None:
            carryover_threshold = np.inf
        options["carryover_threshold"] = carryover_threshold
        use_carryover = carryover_threshold != np.inf
        if use_carryover:
            # The orbitals the carried-over strings are expressed in, for mapping
            # them from the previous call.
            mo_ref = fragment_rohf(h1_mo, h2_mo, norb, nelec).mo_coeff
            if self.carryover_strings is not None:
                self.carryover_strings = permute_carryover(
                    *self.carryover_strings, mo_ref.T @ self.prev_mo, norb
                )
            else:
                empty = np.array([], dtype=np.int64)
                self.carryover_strings = (empty, empty)
            self.prev_mo = mo_ref

        include_configurations = options.get("include_configurations")
        if use_carryover:
            if include_configurations is None:
                include_configurations = self.carryover_strings
            else:
                if isinstance(include_configurations, tuple):
                    include_a, include_b = include_configurations
                else:
                    include_a = include_b = include_configurations
                include_configurations = (
                    np.union1d(include_a, self.carryover_strings[0]),
                    np.union1d(include_b, self.carryover_strings[1]),
                )
        options["include_configurations"] = include_configurations

        histories = {
            name: []
            for name in (
                "e_hist",
                "s_hist",
                "d_hist",
                "a_hist",
                "b_hist",
                "occupancy_hist",
            )
        }
        for name in histories:
            setattr(
                self, name, np.empty((0, 2 * norb if name == "occupancy_hist" else 0))
            )
        user_callback = options.get("callback")

        def callback(results):
            best = min(results, key=lambda result: result.energy)
            rows = {
                "e_hist": [r.energy for r in results],
                "s_hist": [r.sci_state.spin_square() for r in results],
                "a_hist": [len(r.sci_state.ci_strs_a) for r in results],
                "b_hist": [len(r.sci_state.ci_strs_b) for r in results],
                "d_hist": [r.sci_state.amplitudes.size for r in results],
                "occupancy_hist": np.concatenate(best.orbital_occupancies),
            }
            for name, row in rows.items():
                histories[name].append(row)
                setattr(self, name, np.asarray(histories[name]))
            log.info(
                "SQD iteration %d: lowest E = %.12g, <S^2> = %.6f, dims = %s",
                len(self.e_hist) - 1,
                best.energy,
                best.sci_state.spin_square(),
                self.d_hist[-1],
            )
            if user_callback is not None:
                user_callback(results)

        options["callback"] = callback
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
            log.info(
                "SQD carrying over %d alpha, %d beta strings",
                len(np.unique(self.carryover_strings[0])),
                len(np.unique(self.carryover_strings[1])),
            )

        self.sci_state = best.sci_state
        self.e_tot = best.energy + h0
        rdm2 = best.rdm2
        if rdm2 is None:
            rdm2 = best.sci_state.rdm(rank=2, spin_summed=True)
        self.dm1s, self.dm2 = from_mo(mo_coeff, best.sci_state.rdm(rank=1), rdm2)
        if self.output_dir is not None:
            self._save()
        return self.e_tot, self.dm1s, self.dm2

    def _carryover(self, sci_state, carryover_threshold):
        """Alpha and beta strings of the determinants with ``|amplitude| >= carryover_threshold``."""
        amplitudes = np.abs(sci_state.amplitudes)
        alpha_idx, beta_idx = np.nonzero(amplitudes >= carryover_threshold)
        return sci_state.ci_strs_a[alpha_idx], sci_state.ci_strs_b[beta_idx]

    def _save(self):
        d = self.output_dir
        d.mkdir(parents=True, exist_ok=True)
        self.sci_state.save(d / "sci_vec")
        for name in ("e_hist", "d_hist", "a_hist", "b_hist"):
            np.save(d / name, getattr(self, name))
        if self.carryover_strings is not None:
            np.save(d / "alpha_strings", self.carryover_strings[0])
            np.save(d / "beta_strings", self.carryover_strings[1])
