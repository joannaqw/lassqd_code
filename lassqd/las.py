"""LASSQD's interface to upstream mrh's RDM-based LASSCF.

Each hybrid cycle runs LASSCF twice at the same orbitals:

1. ``fragment_hamiltonians`` collects the fragment Hamiltonians, which are used
   to build the circuits that get sampled.
2. ``set_fragment_kernels`` installs kernels that solve SQD on the samples, and
   ``las.kernel`` then takes one orbital step.
"""

from collections.abc import Callable, Sequence

import numpy as np
from mrh.my_pyscf.mcscf.lasscf_rdm import LASSCFNoSymm, make_fcibox

__all__ = ["LASSCFNoSymm", "fragment_hamiltonians", "set_fragment_kernels"]

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


class _HamiltoniansCaptured(Exception):
    pass


def fragment_hamiltonians(
    las: LASSCFNoSymm,
    mo_coeff: np.ndarray,
    casdm1frs: Sequence[np.ndarray] | None = None,
    casdm2fr: Sequence[np.ndarray] | None = None,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Collect the Hamiltonians ``las.kernel`` would pass to each fragment kernel.

    The upstream kernel is run up to its first CI step and then stopped, so these
    are exactly the Hamiltonians the SQD pass will receive at the same orbitals and
    starting RDMs. ``las.fciboxes`` is restored afterwards.

    Args:
        las: RDM-based LASSCF object.
        mo_coeff: Molecular orbital coefficients to evaluate the Hamiltonians at.
        casdm1frs: Fragment 1-RDMs to start from, which set each fragment's
            embedding in the others; None uses mrh's initial guess.
        casdm2fr: Fragment 2-RDMs to start from, matching ``casdm1frs``.

    Returns:
        ``[(h0, h1s, h2), ...]`` in fragment order, with the argument shapes
        described in :func:`set_fragment_kernels`.

    Raises:
        RuntimeError: If ``las.kernel`` returns without visiting every fragment.
    """
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
