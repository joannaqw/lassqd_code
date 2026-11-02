"""LASSQD's interface to upstream mrh's RDM-based LASSCF.

Each hybrid cycle runs LASSCF twice at the same orbitals:

1. ``fragment_hamiltonians`` collects the fragment Hamiltonians, which are used
   to build the circuits that get sampled.
2. ``set_fragment_kernels`` installs kernels that solve SQD on the samples, and
   ``las.kernel`` then takes one orbital step.
"""

from collections.abc import Callable, Sequence

import numpy as np
from mrh.my_pyscf.mcscf.lasscf_rdm import (
    LASSCFNoSymm,
    get_init_guess_rdm,
    make_fcibox,
)

from lassqd.records import LASState

__all__ = [
    "LASSCFNoSymm",
    "fragment_hamiltonians",
    "initial_state",
    "set_fragment_kernels",
]

# ``kernel(norb, nelec, h0, h1s, h2) -> (e, dm1s, dm2)``, e.g. :class:`lassqd.sqd.FragmentSQD`.
FragmentKernel = Callable[
    [int, tuple[int, int], float, np.ndarray, np.ndarray],
    tuple[float, np.ndarray, np.ndarray],
]


def set_fragment_kernels(las: LASSCFNoSymm, kernels: Sequence[FragmentKernel]) -> None:
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


class _HamiltoniansCaptured(Exception):
    pass


def fragment_hamiltonians(
    las: LASSCFNoSymm, state: LASState
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Collect the Hamiltonians ``las.kernel`` would pass to each fragment kernel.

    The upstream kernel is run up to its first CI step and then stopped, so these
    are exactly the Hamiltonians the SQD pass will receive at the same orbitals and
    starting RDMs. ``las.fciboxes`` is restored afterwards.

    Args:
        las: RDM-based LASSCF object.
        state: Orbitals to evaluate the Hamiltonians at, and the fragment RDMs
            that set each fragment's embedding in the others.

    Returns:
        ``[(h0, h1s, h2), ...]`` in fragment order, with the argument shapes
        described in :func:`set_fragment_kernels`.

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
    set_fragment_kernels(las, [capture(i) for i in range(las.nfrags)])
    try:
        las.kernel(mo_coeff, casdm1frs=casdm1frs, casdm2fr=casdm2fr)
    except _HamiltoniansCaptured:
        pass
    else:
        raise RuntimeError("las.kernel returned without calling every fragment kernel")
    finally:
        las.fciboxes = fciboxes
    return hams
