"""
Functions for transforming occupancy arrays.

.. currentmodule:: sqsd.utils.occupancies

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   flip_orbital_occupancies
"""

from __future__ import annotations

import numpy as np


def flip_orbital_occupancies(occupancies: np.ndarray) -> np.ndarray:
    """
    Flip an orbital occupancy array.

    This function reformats a 1D array formatted like:

        ``[occ_a_0, occ_a_1, occ_a_N, occ_b_0, ..., occ_b_N]``

    To an array formatted like:

        ``[occ_a_N, ..., occ_a_0, occ_b_N, ..., occ_b_0]``

    where ``N`` is the number of orbitals.
    """
    num_orbitals = occupancies.shape[0] // 2
    occ_up = occupancies[:num_orbitals]
    occ_dn = occupancies[num_orbitals:]
    occ_out = np.zeros(2 * num_orbitals)
    occ_out[:num_orbitals] = np.flip(occ_up)
    occ_out[num_orbitals:] = np.flip(occ_dn)

    return occ_out
