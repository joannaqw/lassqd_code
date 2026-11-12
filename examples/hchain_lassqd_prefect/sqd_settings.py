"""The SQD solver of this example, and the batch solvers it can use.

This module quarantines ``qiskit_addon_sqd`` and Fulqrum behind the one worker
that needs them, ``solve_worker.py``. Nothing else in the example may import it:
SQD initializes MPI at import time whenever a launcher variable such as
``SLURM_NTASKS`` is set and mpi4py is installed, which would break the capture
and replay workers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial

import numpy as np
from lassqd import FragmentSQD
from qiskit_addon_sqd.fermion import SCIResult, SCIState, solve_sci_batch
from qiskit_addon_sqd.trim import TrimPolicy

import settings

# How :data:`settings.SQD_DEFAULTS` is split between the three objects that
# consume it. Every default must appear in exactly one group.
POLICY_KEYS = ("trim_ratio", "carryover_ratio")
BATCH_SOLVER_KEYS = ("tol",)
SOLVER_KEYS = ("max_iterations", "num_batches", "max_dim", "symmetrize_spin")
SCI_SOLVERS = ("fulqrum", "pyscf")


def postselected_bitstrings(
    counts: Mapping[str, int], norb: int, nelec: tuple[int, int]
) -> set[str]:
    """The distinct measured bitstrings with the requested electron counts.

    The right half of each key holds the alpha occupations, as
    ``lassqd.records.FragmentSamples`` documents.
    """
    neleca, nelecb = nelec
    keys = (key.replace(" ", "") for key in counts)
    return {
        key
        for key in keys
        if key[norb:].count("1") == neleca and key[:norb].count("1") == nelecb
    }


def check_batches(
    counts: Mapping[str, int], norb: int, nelec: tuple[int, int], num_batches: int
) -> int:
    """Pre-flight check that the counts can fill ``num_batches`` disjoint batches.

    Trim SQD deals one pool of bitstrings among its batches without repetition,
    so with fewer distinct postselected bitstrings than batches some batch is
    empty, and qiskit-addon-sqd 0.14.0 and 0.14.1 then fail inside the batch
    solver. In a collective solve that failure lands on whichever rank owns the
    empty batch, possibly not rank 0, while the others wait in the gather, so
    the check belongs before SQD is entered at all.

    Args:
        counts: The fragment's counts.
        norb: Number of fragment orbitals.
        nelec: ``(neleca, nelecb)``.
        num_batches: The solver's batch count.

    Returns:
        The number of distinct postselected bitstrings.

    Raises:
        ValueError: If that number is below ``num_batches``.
    """
    postselected = len(postselected_bitstrings(counts, norb, nelec))
    if postselected < num_batches:
        raise ValueError(
            f"only {postselected} distinct bitstrings have {nelec} electrons, "
            f"fewer than the {num_batches} disjoint batches SQD would deal them "
            "into; sample more shots, add noise to the simulator, or lower "
            "num_batches"
        )
    return postselected


def solve_sci_batch_fulqrum(
    ci_batches: list[tuple[np.ndarray, np.ndarray]],
    h1: np.ndarray,
    h2: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    tol: float = 1e-12,
) -> list[SCIResult]:
    """Diagonalize each SQD batch with Fulqrum, as ``docs/lassqd.ipynb`` does.

    The start vector is the lowest diagonal element rather than ARPACK's random
    default, which is what makes repeated runs of this example bitwise equal.
    """
    import fulqrum as fq
    import scipy.sparse.linalg
    from fulqrum.convert.integrals import integrals_to_fq_fermionic_op

    fermionic_op = integrals_to_fq_fermionic_op(
        one_body_integrals=np.ascontiguousarray(h1),
        two_body_integrals=np.ascontiguousarray(h2),
    )
    hsub = fq.SubspaceHamiltonian(
        fermionic_op.extended_jw_transformation(),
        fq.Subspace([["0" * norb], ["0" * norb]]),
    )

    results = []
    for ints_a, ints_b in ci_batches:
        strs_a = [format(x, f"0{norb}b") for x in ints_a]
        strs_b = [format(x, f"0{norb}b") for x in ints_b]
        subspace = fq.Subspace([strs_a, strs_b])
        hsub.update_subspace(subspace)
        diagonal = hsub.diagonal_vector()
        if len(subspace) == 1:
            energy = diagonal[0].real
            vec = np.ones(1, dtype=hsub.dtype)
        else:
            projected_hamiltonian = hsub.to_csr_linearoperator_fast()
            v0 = np.zeros(len(subspace), dtype=hsub.dtype)
            v0[np.argmin(diagonal)] = 1.0
            eigs, vecs = scipy.sparse.linalg.eigsh(
                projected_hamiltonian, k=1, which="SA", tol=tol, v0=v0
            )
            energy, vec = eigs[0], vecs[:, 0]
        occupancies = subspace.get_orbital_occupancies(
            probs=np.abs(vec) ** 2, norb=norb
        )
        # Fulqrum and SQD order the alpha/beta product space oppositely.
        state = SCIState(
            amplitudes=np.ascontiguousarray(vec.reshape(len(strs_b), len(strs_a)).T),
            ci_strs_a=ints_a,
            ci_strs_b=ints_b,
            norb=norb,
            nelec=nelec,
        )
        results.append(
            SCIResult(energy=energy, sci_state=state, orbital_occupancies=occupancies)
        )
    return results


def sci_solver(
    name: str, options: Mapping[str, object]
) -> Callable[..., list[SCIResult]]:
    """The batch solver called ``name``, configured from ``options``.

    Args:
        name: ``"fulqrum"`` for :func:`solve_sci_batch_fulqrum`, or ``"pyscf"``
            for the SQD addon's ``solve_sci_batch``. Fulqrum is source-only and
            the repository's CI excludes it, so ``"pyscf"`` is the fallback.
        options: SQD settings; see :data:`BATCH_SOLVER_KEYS`.

    Returns:
        A callable ``(ci_batches, h1, h2, norb, nelec) -> list[SCIResult]``.

    Raises:
        ValueError: If ``name`` is not one of :data:`SCI_SOLVERS`.
    """
    if name == "fulqrum":
        return partial(solve_sci_batch_fulqrum, tol=options["tol"])
    if name == "pyscf":
        return partial(solve_sci_batch, tol=options["tol"])
    raise ValueError(f"unknown sci_solver {name!r}, expected one of {SCI_SOLVERS}")


def make_solver(
    ifrag: int,
    options: Mapping[str, object],
    *,
    sci_solver_name: str = "fulqrum",
    callback: Callable[[list[SCIResult]], None] | None = None,
) -> FragmentSQD:
    """The trim-SQD solver for fragment ``ifrag``.

    Its random stream is :func:`settings.solver_root`, so the serial reference
    and the distributed run give this fragment the same stream however the
    fragments are ordered. ``FragmentSQD`` spawns one child from the generator,
    and each later solve continues from the ``SQDState`` record rather than from
    this object.

    Args:
        ifrag: Fragment index.
        options: SQD settings, from :func:`settings.sqd_options`.
        sci_solver_name: Which batch solver to use; see :func:`sci_solver`.
        callback: Receives each recovery iteration's reported results.

    Returns:
        The solver, with its initial state.
    """
    options = dict(options)
    missing = sorted(
        set(POLICY_KEYS + BATCH_SOLVER_KEYS + SOLVER_KEYS + ("samples_per_batch",))
        - set(options)
    )
    if missing:
        raise ValueError(f"SQD settings are missing {', '.join(missing)}")
    return FragmentSQD(
        samples_per_batch=options["samples_per_batch"],
        policy=TrimPolicy(**{key: options[key] for key in POLICY_KEYS}),
        sci_solver=sci_solver(sci_solver_name, options),
        seed=settings.solver_root(ifrag),
        callback=callback,
        **{key: options[key] for key in SOLVER_KEYS},
    )
