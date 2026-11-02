"""LASSQD's interface to upstream mrh's RDM-based LASSCF.

mrh solves the fragments from inside its kernel, through fragment kernels.
LASSQD enters ``las.kernel`` twice per cycle at the same orbitals and starting
RDMs, with kernels that do not solve anything:

1. :func:`capture_embedding` installs kernels that record each fragment's
   Hamiltonian and stops mrh once all are recorded. mrh builds every fragment's
   Hamiltonian before its fragment loop, so the fragments can then be solved
   outside mrh, independently.
2. :func:`replay_orbital_step` installs kernels that check each Hamiltonian
   against the recorded one and return the stored fragment results, and lets
   mrh take one orbital step. mrh computes the energy from the RDMs; the
   fragment energies it receives are only logged.

No other module calls mrh.
"""

from collections.abc import Callable, Sequence

import numpy as np
from mrh.my_pyscf.mcscf.lasscf_rdm import (
    LASSCFNoSymm,
    get_init_guess_rdm,
    make_fcibox,
)

from lassqd.records import Embedding, Fragment, FragmentResult, LASState

__all__ = [
    "LASSCFNoSymm",
    "capture_embedding",
    "fragment_hamiltonians",
    "initial_state",
    "replay_orbital_step",
]

# ``kernel(norb, nelec, h0, h1s, h2) -> (e, dm1s, dm2)``.
FragmentKernel = Callable[
    [int, tuple[int, int], float, np.ndarray, np.ndarray],
    tuple[float, np.ndarray, np.ndarray],
]

# Largest difference allowed between a replayed and a captured Hamiltonian.
REPLAY_ATOL = 1e-10


def _set_fragment_kernels(las: LASSCFNoSymm, kernels: Sequence[FragmentKernel]) -> None:
    """Solve fragment ``i`` with ``kernels[i](norb, nelec, h0, h1s, h2) -> (e, dm1s, dm2)``.

    Each fragment keeps the spin and multiplicity ``las`` was built with
    (``spin_sub``), which upstream mrh needs to write its checkpoint file.

    Args:
        las: RDM-based LASSCF object; its ``fciboxes`` are replaced in place.
        kernels: One kernel per fragment, in fragment order. Each receives the
            fragment's orbital count, ``(neleca, nelecb)``, constant energy,
            spin-separated one-electron integrals of shape ``(2, norb, norb)`` and
            two-electron integrals of shape ``(norb,) * 4``, and returns the
            fragment energy, spin-separated 1-RDMs of shape ``(2, norb, norb)`` and
            spin-summed 2-RDM of shape ``(norb,) * 4``.
    """
    las.fciboxes = [
        make_fcibox(
            las.mol,
            kernel=kernel,
            spin=fcibox.fcisolvers[0].spin,
            smult=fcibox.fcisolvers[0].smult,
        )
        for kernel, fcibox in zip(kernels, las.fciboxes)
    ]


def initial_state(las: LASSCFNoSymm, mo_coeff: np.ndarray) -> LASState:
    """mrh's initial-guess fragment RDMs at ``mo_coeff``, as a :class:`LASState`.

    These are the RDMs ``las.kernel`` starts from when it is given none.

    Args:
        las: RDM-based LASSCF object.
        mo_coeff: Molecular orbital coefficients.

    Returns:
        The orbitals with the initial-guess RDMs in their active space.
    """
    casdm1frs, casdm2fr = get_init_guess_rdm(las, mo_coeff, las.get_h2eff(mo_coeff))
    return LASState(
        mo_coeff,
        tuple(np.stack(dm1rs) for dm1rs in casdm1frs),
        tuple(np.stack(dm2r) for dm2r in casdm2fr),
    )


def capture_embedding(las: LASSCFNoSymm, state: LASState) -> Embedding:
    """Capture every fragment's Hamiltonian at ``state`` and find its sampling basis.

    Args:
        las: RDM-based LASSCF object.
        state: Orbitals to build the fragment Hamiltonians at, and the fragment
            RDMs that set each fragment's embedding in the others.

    Returns:
        The fragments of ``state``, each with its LAS orbitals and ROHF basis.
    """
    hamiltonians = fragment_hamiltonians(las, state)
    bounds = las.ncore + np.cumsum([0, *las.ncas_sub])
    fragments = [
        Fragment.from_hamiltonian(
            norb,
            tuple(nelec),
            h0,
            h1s,
            h2,
            index=i,
            las_orbitals=state.mo_coeff[:, start:stop],
        )
        for i, ((h0, h1s, h2), norb, nelec, start, stop) in enumerate(
            zip(hamiltonians, las.ncas_sub, las.nelecas_sub, bounds[:-1], bounds[1:])
        )
    ]
    return Embedding(state, fragments, las._scf.get_ovlp())


class _HamiltoniansCaptured(Exception):
    pass


def fragment_hamiltonians(
    las: LASSCFNoSymm, state: LASState
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Collect the Hamiltonians ``las.kernel`` would pass to each fragment kernel.

    The upstream kernel is run up to its first CI step and then stopped, so these
    are exactly the Hamiltonians :func:`replay_orbital_step` receives at the same
    state. ``las.fciboxes`` is restored afterwards.

    Args:
        las: RDM-based LASSCF object.
        state: Orbitals to evaluate the Hamiltonians at, and the fragment RDMs
            that set each fragment's embedding in the others.

    Returns:
        ``[(h0, h1s, h2), ...]`` in fragment order: the constant energy,
        spin-separated one-electron integrals of shape ``(2, norb, norb)`` and
        two-electron integrals of shape ``(norb,) * 4``, in the LAS basis.

    Raises:
        RuntimeError: If ``las.kernel`` returns without visiting every fragment.
    """
    return _capture_hamiltonians(
        las, state.mo_coeff, list(state.casdm1frs), list(state.casdm2fr)
    )


def _capture_hamiltonians(
    las: LASSCFNoSymm,
    mo_coeff: np.ndarray,
    casdm1frs: Sequence[np.ndarray] | None,
    casdm2fr: Sequence[np.ndarray] | None,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """:func:`fragment_hamiltonians` from loose arrays; None RDMs use mrh's guess."""
    hams = [None] * las.nfrags

    def capture(ifrag):
        def kernel(norb, nelec, h0, h1s, h2):
            hams[ifrag] = (h0, h1s, h2)
            if all(h is not None for h in hams):
                raise _HamiltoniansCaptured
            # Placeholder for fragments visited before the last one; never used.
            return 0.0, np.zeros((2, norb, norb)), np.zeros((norb,) * 4)

        return kernel

    fciboxes = las.fciboxes
    try:
        _set_fragment_kernels(las, [capture(i) for i in range(las.nfrags)])
        las.kernel(mo_coeff, casdm1frs=casdm1frs, casdm2fr=casdm2fr)
    except _HamiltoniansCaptured:
        pass
    else:
        raise RuntimeError("las.kernel returned without calling every fragment kernel")
    finally:
        las.fciboxes = fciboxes
    return hams


def replay_orbital_step(
    las: LASSCFNoSymm, embedding: Embedding, results: Sequence[FragmentResult]
) -> tuple[float, LASState]:
    """Take one LASSCF orbital step from ``embedding.state`` with solved fragments.

    ``las.kernel`` is entered at the embedding's orbitals and *starting* RDMs,
    with ``max_cycle_macro=1`` and ``max_cycle_rdmjk=0``. Each fragment kernel
    checks that mrh passes the captured Hamiltonian and returns that fragment's
    stored result; mrh then computes the coupled LAS energy, takes one orbital
    step and canonicalizes. ``las.fciboxes``, ``las.max_cycle_macro`` and
    ``las.max_cycle_rdmjk`` are restored afterwards, including on failure.

    Args:
        las: RDM-based LASSCF object, set up like the one the embedding was
            captured with.
        embedding: The fragment problems, from :func:`capture_embedding`.
        results: One result per fragment, in fragment order, solved for
            ``embedding.fragments``.

    Returns:
        The total energy and the LAS state after the step.

    Raises:
        ValueError: If mrh passes a Hamiltonian other than the captured one.
        RuntimeError: If mrh does not visit every fragment exactly once.
    """
    fragments = embedding.fragments
    if len(results) != len(fragments):
        raise ValueError(f"{len(results)} results for {len(fragments)} fragments")
    visits = [0] * len(fragments)

    def replay(fragment, result):
        def kernel(norb, nelec, h0, h1s, h2):
            visits[fragment.index] += 1
            if visits[fragment.index] > 1:
                raise RuntimeError(f"mrh visited fragment {fragment.index} twice")
            _check_hamiltonian(fragment, norb, nelec, h0, h1s, h2)
            return result.e_tot, result.dm1s, result.dm2

        return kernel

    state = embedding.state
    saved = las.fciboxes, las.max_cycle_macro, las.max_cycle_rdmjk
    try:
        las.max_cycle_macro, las.max_cycle_rdmjk = 1, 0
        _set_fragment_kernels(las, [replay(f, r) for f, r in zip(fragments, results)])
        las.kernel(
            state.mo_coeff,
            casdm1frs=list(state.casdm1frs),
            casdm2fr=list(state.casdm2fr),
        )
    finally:
        las.fciboxes, las.max_cycle_macro, las.max_cycle_rdmjk = saved
    if visits != [1] * len(fragments):
        raise RuntimeError(f"mrh visited the fragments {visits} times")
    return las.e_tot, LASState(las.mo_coeff, tuple(las.casdm1frs), tuple(las.casdm2fr))


def _check_hamiltonian(fragment, norb, nelec, h0, h1s, h2) -> None:
    """Raise ValueError unless mrh's fragment problem is the captured one."""
    i = fragment.index
    if (int(norb), tuple(int(n) for n in np.ravel(nelec))) != (
        fragment.norb,
        fragment.nelec,
    ):
        raise ValueError(
            f"fragment {i}: mrh passed a different orbital or electron count"
        )
    for name, value, captured in (
        ("h0", h0, fragment.h0),
        ("h1s", h1s, fragment.h1s_las),
        ("h2", h2, fragment.h2_las),
    ):
        value = np.asarray(value)
        if value.shape != np.shape(captured) or not np.allclose(
            value, captured, rtol=0, atol=REPLAY_ATOL
        ):
            raise ValueError(
                f"fragment {i}: mrh passed a {name} that differs from the captured one"
            )
