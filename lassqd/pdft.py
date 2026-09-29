"""LAS-PDFT on an SQD wave function.

Like LAS-PDFT, the on-top energy is evaluated once, on a converged LAS wave
function: orbitals plus the fragment RDMs expressed in those orbitals' active
space. After :func:`lassqd.run_lassqd` that is ``result.mo_coeff`` with
``result.casdm1frs`` / ``result.casdm2fr`` (equivalently ``las.mo_coeff`` with
``las.casdm1frs`` / ``las.casdm2fr``).

``casdm1frs[i]`` has shape ``(nroots, 2, n_i, n_i)`` and ``casdm2fr[i]`` shape
``(nroots, n_i, n_i, n_i, n_i)``, as stored by mrh's RDM-based LASSCF; only root 0
is used.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from itertools import combinations
from typing import TYPE_CHECKING

import h5py
import numpy as np
from scipy import linalg

if TYPE_CHECKING:
    from mrh.my_pyscf.mcscf.lasci import LASCINoSymm


def save_rdms(
    filename: str | os.PathLike,
    casdm1frs: Sequence[np.ndarray],
    casdm2fr: Sequence[np.ndarray],
    mo_coeff: np.ndarray | None = None,
) -> None:
    """Write fragment RDMs (and optionally the matching orbitals) to HDF5.

    Args:
        filename: HDF5 file to write; overwritten if it exists.
        casdm1frs: Per-fragment spin-separated 1-RDMs, shape ``(nroots, 2, n_i, n_i)``
            or ``(2, n_i, n_i)``. They are stored with a leading root axis.
        casdm2fr: Per-fragment spin-summed 2-RDMs, shape ``(nroots, n_i, n_i, n_i, n_i)``
            or ``(n_i, n_i, n_i, n_i)``. They are stored with a leading root axis.
        mo_coeff: Orbitals the RDMs are expressed in, if they should be stored too.
    """
    with h5py.File(filename, "w") as f:
        dm1_group = f.create_group("casdm1frs")
        dm2_group = f.create_group("casdm2frs")
        for i, (dm1, dm2) in enumerate(zip(casdm1frs, casdm2fr)):
            dm1_group.create_dataset(str(i), data=np.reshape(dm1, (-1,) + np.shape(dm1)[-3:]))
            dm2_group.create_dataset(str(i), data=np.reshape(dm2, (-1,) + np.shape(dm2)[-4:]))
        if mo_coeff is not None:
            f.create_dataset("mo_coeff", data=mo_coeff)


def load_rdms(
    filename: str | os.PathLike,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray | None]:
    """Read fragment RDMs and orbitals written by :func:`save_rdms`.

    Args:
        filename: HDF5 file to read.

    Returns:
        ``(casdm1frs, casdm2fr, mo_coeff)``, with a leading root axis on every RDM.
        ``mo_coeff`` is None if the file does not contain orbitals.
    """
    with h5py.File(filename, "r") as f:
        keys = sorted(f["casdm1frs"].keys(), key=int)
        casdm1frs = [f["casdm1frs"][k][()] for k in keys]
        casdm2fr = [f["casdm2frs"][k][()] for k in keys]
        mo_coeff = f["mo_coeff"][()] if "mo_coeff" in f else None
    return casdm1frs, casdm2fr, mo_coeff


def make_casdm1s(casdm1frs: Sequence[np.ndarray]) -> np.ndarray:
    """Build the block-diagonal spin-separated active-space 1-RDM from root 0.

    Args:
        casdm1frs: Per-fragment spin-separated 1-RDMs, shape ``(nroots, 2, n_i, n_i)``.

    Returns:
        The active-space 1-RDM, shape ``(2, ncas, ncas)``.
    """
    return np.stack(
        [linalg.block_diag(*[dm1rs[0][ispin] for dm1rs in casdm1frs]) for ispin in (0, 1)],
        axis=0,
    )


def make_casdm2(casdm1frs: Sequence[np.ndarray], casdm2fr: Sequence[np.ndarray]) -> np.ndarray:
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
    las: LASCINoSymm,
    casdm1frs: Sequence[np.ndarray],
    casdm2fr: Sequence[np.ndarray],
    mo_coeff: np.ndarray,
    ot: str = "tPBE",
) -> float:
    """Compute the MC-PDFT total energy of a LAS wave function.

    The wave function is ``(mo_coeff, casdm1frs, casdm2fr)``. This uses mrh's
    LAS-PDFT with the RDMs supplied directly instead of built from CI vectors.

    Args:
        las: LAS object defining the molecule and fragment partitioning.
        casdm1frs: Per-fragment spin-separated 1-RDMs, shape ``(nroots, 2, n_i, n_i)``.
        casdm2fr: Per-fragment spin-summed 2-RDMs, shape ``(nroots, n_i, n_i, n_i, n_i)``.
        mo_coeff: Orbitals whose active space the RDMs are expressed in.
        ot: On-top functional, e.g. ``"tPBE"``.

    Returns:
        The MC-PDFT total energy.
    """
    from mrh.my_pyscf.mcpdft.laspdft import get_mcpdft_child_class

    pdft = get_mcpdft_child_class(las, ot=ot)
    casdm1s = make_casdm1s(casdm1frs)
    casdm2 = make_casdm2(casdm1frs, casdm2fr)
    pdft.make_one_casdm1s = lambda ci=None, state=0, **kwargs: casdm1s
    pdft.make_one_casdm2 = lambda ci=None, state=0, **kwargs: casdm2
    e_tot, e_ot, e_states = pdft.compute_pdft_energy_(mo_coeff)
    return e_tot
