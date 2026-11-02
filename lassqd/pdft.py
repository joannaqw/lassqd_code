"""LAS-PDFT on an SQD wave function.

Like LAS-PDFT, the on-top energy is evaluated once, on a converged LAS wave
function: a :class:`lassqd.records.LASState`, such as ``result.state`` after
:func:`lassqd.run_lassqd`. Only root 0 of the fragment RDMs is used.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg

if TYPE_CHECKING:
    from mrh.my_pyscf.mcscf.lasci import LASCINoSymm

    from lassqd.records import LASState


def make_casdm1s(casdm1frs: Sequence[np.ndarray]) -> np.ndarray:
    """Build the block-diagonal spin-separated active-space 1-RDM from root 0.

    Args:
        casdm1frs: Per-fragment spin-separated 1-RDMs, shape ``(nroots, 2, n_i, n_i)``.

    Returns:
        The active-space 1-RDM, shape ``(2, ncas, ncas)``.
    """
    return np.stack(
        [
            linalg.block_diag(*[dm1rs[0][ispin] for dm1rs in casdm1frs])
            for ispin in (0, 1)
        ],
        axis=0,
    )


def make_casdm2(
    casdm1frs: Sequence[np.ndarray], casdm2fr: Sequence[np.ndarray]
) -> np.ndarray:
    """Build the spin-summed active-space 2-RDM of the LAS product state from root 0.

    Diagonal blocks are the fragment 2-RDMs; off-diagonal blocks are the Coulomb
    and exchange products of the fragment 1-RDMs.

    Args:
        casdm1frs: Per-fragment spin-separated 1-RDMs, shape ``(nroots, 2, n_i, n_i)``.
        casdm2fr: Per-fragment spin-summed 2-RDMs, shape ``(nroots, n_i, n_i, n_i, n_i)``.

    Returns:
        The active-space 2-RDM, shape ``(ncas,) * 4``.
    """
    ncas_sub = [dm1rs.shape[-1] for dm1rs in casdm1frs]
    ncas_cum = np.cumsum([0] + ncas_sub)
    ncas = ncas_cum[-1]
    casdm2 = np.zeros((ncas,) * 4)
    for isub, dm2r in enumerate(casdm2fr):
        i, j = ncas_cum[isub], ncas_cum[isub + 1]
        casdm2[i:j, i:j, i:j, i:j] = dm2r[0]
    for (isub1, dm1s1), (isub2, dm1s2) in combinations(enumerate(casdm1frs), 2):
        i, j = ncas_cum[isub1], ncas_cum[isub1 + 1]
        k, l = ncas_cum[isub2], ncas_cum[isub2 + 1]
        dma1, dmb1 = dm1s1[0]
        dma2, dmb2 = dm1s2[0]
        # Coulomb slice: e.g., [1,2,2,1]
        casdm2[i:j, i:j, k:l, k:l] = np.multiply.outer(dma1 + dmb1, dma2 + dmb2)
        casdm2[k:l, k:l, i:j, i:j] = casdm2[i:j, i:j, k:l, k:l].transpose(2, 3, 0, 1)
        # Exchange slice: e.g., [2,1,1,2]
        casdm2[i:j, k:l, k:l, i:j] = -(
            np.multiply.outer(dma1, dma2) + np.multiply.outer(dmb1, dmb2)
        ).transpose(0, 3, 2, 1)
        casdm2[k:l, i:j, i:j, k:l] = casdm2[i:j, k:l, k:l, i:j].transpose(1, 0, 3, 2)
    return casdm2


def lassqd_pdft_energy(
    las: LASCINoSymm, state: LASState, ot: str = "tPBE"
) -> tuple[float, float, list[float]]:
    """Compute the MC-PDFT energy of a LAS wave function.

    This uses mrh's LAS-PDFT with the RDMs supplied directly instead of built from
    CI vectors.

    Args:
        las: LAS object defining the molecule and fragment partitioning.
        state: Orbitals and the fragment RDMs in their active space.
        ot: On-top functional, e.g. ``"tPBE"``.

    Returns:
        ``(e_tot, e_ot, e_states)``: the MC-PDFT total energy, its on-top
        exchange-correlation component, and the per-state MC-PDFT energies, as
        returned by mrh's ``compute_pdft_energy_``.
    """
    from mrh.my_pyscf.mcpdft.laspdft import get_mcpdft_child_class

    pdft = get_mcpdft_child_class(las, ot=ot)
    casdm1s = make_casdm1s(state.casdm1frs)
    casdm2 = make_casdm2(state.casdm1frs, state.casdm2fr)
    pdft.make_one_casdm1s = lambda ci=None, state=0, **kwargs: casdm1s
    pdft.make_one_casdm2 = lambda ci=None, state=0, **kwargs: casdm2
    e_tot, e_ot, e_states = pdft.compute_pdft_energy_(state.mo_coeff)
    return e_tot, e_ot, e_states
