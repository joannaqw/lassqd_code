"""Sample-based quantum diagonalization as a LASSCF fragment kernel."""

import sys
from pathlib import Path

import numpy as np
from pyscf import fci
from pyscf.lib import logger
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.fermion import (
    SCIResult,
    SCIState,
    bitstring_matrix_to_ci_strs,
    solve_sci,
)
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left, subsample

from lassqd.basis import fragment_mo_basis, fragment_rohf, from_mo


def solve_sci_nroots(
    ci_strings,
    hcore: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    spin_sq: float | None = None,
    nroots: int | None = None,
    **kwargs,
) -> SCIResult:
    """``qiskit_addon_sqd.fermion.solve_sci`` with an optional ``nroots``.

    The addon's own solvers only handle a single Davidson root -- their RDM
    post-processing assumes one SCIvector, not a list. When ``nroots`` is given
    we therefore call pyscf directly and repackage the lowest root into the
    addon's :class:`SCIResult` / :class:`SCIState` types, so the rest of the
    loop is unchanged. ``nroots=None`` delegates straight to the addon.
    """
    if nroots is None:
        return solve_sci(ci_strings, hcore, eri, norb, nelec, spin_sq=spin_sq, **kwargs)

    myci = fci.selected_ci.SelectedCI()
    if spin_sq is not None:
        myci = fci.addons.fix_spin_(myci, ss=spin_sq)
    _, sci_vecs = fci.selected_ci.kernel_fixed_space(
        myci, hcore, eri, norb, nelec, ci_strs=ci_strings, nroots=nroots, **kwargs
    )
    sci_vec = sci_vecs[0]

    # Energy from the RDMs, matching what solve_sci does.
    dm1s = myci.make_rdm1s(sci_vec, norb, nelec)
    dm1 = myci.make_rdm1(sci_vec, norb, nelec)
    dm2 = myci.make_rdm2(sci_vec, norb, nelec)
    energy = np.einsum("pr,pr->", dm1, hcore) + 0.5 * np.einsum("prqs,prqs->", dm2, eri)

    sci_state = SCIState(
        amplitudes=np.array(sci_vec),
        ci_strs_a=sci_vec._strs[0],
        ci_strs_b=sci_vec._strs[1],
        norb=norb,
        nelec=nelec,
    )
    return SCIResult(
        energy,
        sci_state,
        orbital_occupancies=(np.diagonal(dm1s[0]), np.diagonal(dm1s[1])),
        rdm1=dm1,
        rdm2=dm2,
    )


def _strings_to_bitstrings(strings, norb):
    """Integer determinant strings -> boolean rows, most significant orbital first."""
    return ((strings[:, None] >> np.arange(norb)[::-1]) & 1).astype(bool)


def permute_carryover(strings_a, strings_b, overlap, norb):
    """Map carried-over determinant strings onto a new orbital basis.

    ``overlap[p, q]`` is the overlap of new orbital ``p`` with old orbital ``q``.
    Each new orbital takes the occupation of the old orbital it overlaps most.
    Returns unique ``(strings_a, strings_b)``.
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
    ``iterations`` rounds of self-consistent configuration recovery, with
    ``n_batches`` subsampled batches per round. The energy, RDMs and carryover
    strings all come from the lowest-energy batch of the last round.

    Only the alpha one-electron Hamiltonian ``h1s[0]`` is used.

    Carryover (``carryover_threshold`` not None): determinants whose amplitude
    exceeds the threshold are added to every batch of the next round, and of the
    next call, after being mapped onto that call's orbitals with
    :func:`permute_carryover`.

    Attributes set by each call: ``e_hist``, ``d_hist`` (subspace dimension),
    ``a_hist``/``b_hist`` (alpha/beta string counts), ``s_hist`` (<S^2>), all of
    shape ``(iterations, n_batches)``; ``occupancy_hist``; ``e_tot``; ``dm1s`` and
    ``dm2`` in the LAS basis; ``sci_state`` in the fragment ROHF basis.
    """

    def __init__(
        self,
        iterations,
        n_batches,
        samples_per_batch,
        *,
        max_davidson_cycles=200,
        tol=1e-12,
        nroots=None,
        spin_sq=None,
        carryover_threshold=None,
        seed=None,
        output_dir=None,
        verbose=logger.INFO,
    ):
        self.iterations = iterations
        self.n_batches = n_batches
        self.samples_per_batch = samples_per_batch
        self.max_davidson_cycles = max_davidson_cycles
        self.tol = tol
        self.nroots = nroots
        self.spin_sq = spin_sq
        self.carryover_threshold = carryover_threshold
        self.rng = np.random.default_rng(seed)
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.verbose = verbose
        self.counts = None
        self.carryover_strings = None
        self.prev_mo = None

    def __call__(self, norb, nelec, h0, h1s, h2):
        log = logger.Logger(sys.stdout, self.verbose)
        if self.counts is None:
            raise RuntimeError("FragmentSQD.counts must be set before the kernel is called")
        neleca, nelecb = nelec
        spin_sq = self.spin_sq
        if spin_sq is None:
            s = abs(neleca - nelecb) / 2
            spin_sq = s * (s + 1)
        h1 = h1s[0] if np.ndim(h1s) == 3 else h1s
        mo_coeff, h1_mo, h2_mo = fragment_mo_basis(h1, h2, norb, nelec)

        use_carryover = self.carryover_threshold is not None
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

        shape = (self.iterations, self.n_batches)
        self.e_hist = np.zeros(shape)
        self.s_hist = np.zeros(shape)
        self.d_hist = np.zeros(shape)
        self.a_hist = np.zeros(shape)
        self.b_hist = np.zeros(shape)
        self.occupancy_hist = np.zeros((self.iterations, 2 * norb))

        bitstring_matrix_full, probs_full = counts_to_arrays(self.counts)
        avg_occupancies = None
        for i in range(self.iterations):
            if avg_occupancies is None:
                bs_mat, probs = bitstring_matrix_full, probs_full
            else:
                bs_mat, probs = recover_configurations(
                    bitstring_matrix_full,
                    probs_full,
                    avg_occupancies,
                    neleca,
                    nelecb,
                    rand_seed=self.rng,
                )
            # Throw out samples with incorrect hamming weight and create batches of subsamples.
            bs_mat, probs = postselect_by_hamming_right_and_left(
                bs_mat, probs, hamming_right=neleca, hamming_left=nelecb
            )
            batches = subsample(
                bs_mat, probs, self.samples_per_batch, self.n_batches, rand_seed=self.rng
            )

            results = []
            occs = np.zeros((self.n_batches, 2 * norb))
            for j, batch in enumerate(batches):
                strings_a, strings_b = bitstring_matrix_to_ci_strs(batch, open_shell=True)
                if use_carryover:
                    strings_a = np.union1d(strings_a, self.carryover_strings[0])
                    strings_b = np.union1d(strings_b, self.carryover_strings[1])
                result = solve_sci_nroots(
                    (strings_a, strings_b),
                    h1_mo,
                    h2_mo,
                    norb,
                    (neleca, nelecb),
                    spin_sq=spin_sq,
                    nroots=self.nroots,
                    max_cycle=self.max_davidson_cycles,
                    tol=self.tol,
                )
                results.append(result)
                self.e_hist[i, j] = result.energy
                self.s_hist[i, j] = result.sci_state.spin_square()
                self.a_hist[i, j] = len(strings_a)
                self.b_hist[i, j] = len(strings_b)
                self.d_hist[i, j] = len(strings_a) * len(strings_b)
                # occs stores (spin-up | spin-down); recover_configurations wants (up, down).
                occs[j, :norb], occs[j, norb:] = result.orbital_occupancies
            self.occupancy_hist[i] = occs.mean(axis=0)
            avg_occupancies = (self.occupancy_hist[i, :norb], self.occupancy_hist[i, norb:])

            best = results[int(np.argmin(self.e_hist[i]))]
            log.info(
                "SQD iteration %d: lowest E = %.12g, <S^2> = %.6f, dims = %s",
                i,
                best.energy,
                best.sci_state.spin_square(),
                self.d_hist[i].astype(int),
            )
            if use_carryover:
                self.carryover_strings = self._carryover(best.sci_state)
                log.info(
                    "SQD carrying over %d alpha, %d beta strings",
                    len(np.unique(self.carryover_strings[0])),
                    len(np.unique(self.carryover_strings[1])),
                )

        self.sci_state = best.sci_state
        self.e_tot = best.energy + h0
        self.dm1s, self.dm2 = from_mo(mo_coeff, best.sci_state.rdm(rank=1), best.rdm2)
        if self.output_dir is not None:
            self._save()
        return self.e_tot, self.dm1s, self.dm2

    def _carryover(self, sci_state):
        """Alpha and beta strings of the determinants with ``|amplitude| >= carryover_threshold``."""
        amplitudes = np.abs(sci_state.amplitudes)
        alpha_idx, beta_idx = np.nonzero(amplitudes >= self.carryover_threshold)
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
