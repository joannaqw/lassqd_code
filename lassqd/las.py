"""LASSQD's interface to upstream mrh's RDM-based LASSCF.

Each hybrid cycle runs LASSCF twice at the same orbitals:

1. ``fragment_hamiltonians`` collects the fragment Hamiltonians, which are used
   to build the circuits that get sampled.
2. ``set_fragment_kernels`` installs kernels that solve SQD on the samples, and
   ``las.kernel`` then takes one orbital step.
"""

import numpy as np
from mrh.my_pyscf.mcscf.lasscf_rdm import LASSCFNoSymm, make_fcibox

__all__ = ["LASSCFNoSymm", "set_fragment_kernels", "fragment_hamiltonians"]


def set_fragment_kernels(las, kernels):
    """Solve fragment ``i`` with ``kernels[i](norb, nelec, h0, h1s, h2) -> (e, dm1s, dm2)``.

    Each fragment keeps the spin and multiplicity ``las`` was built with
    (``spin_sub``), which upstream mrh needs to write its checkpoint file.
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


def fragment_hamiltonians(las, mo_coeff):
    """Return ``[(h0, h1s, h2), ...]`` that ``las.kernel(mo_coeff)`` would pass to
    each fragment kernel.

    The upstream kernel is run up to its first CI step and then stopped, so these
    are exactly the Hamiltonians the SQD pass will receive at the same orbitals.
    ``las.fciboxes`` is restored afterwards.
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
        las.kernel(mo_coeff)
    except _HamiltoniansCaptured:
        pass
    else:
        raise RuntimeError("las.kernel returned without calling every fragment kernel")
    finally:
        las.fciboxes = fciboxes
    return hams
