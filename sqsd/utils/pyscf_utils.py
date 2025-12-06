"""
Functions for preparing and interpreting pyscf-specific data structures.

.. currentmodule:: sqsd.utils.pyscf_utils

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   bitstring_matrix_to_sorted_addresses
"""

import numpy as np


def bitstring_matrix_to_sorted_addresses(
    bitstring_matrix: np.ndarray, open_shell: bool = False
) -> tuple[list[int], list[int]]:
    """
    Convert a bitstring matrix into ``pyscf`` address representation.

    The ``pyscf.fci`` package requires Slater determinants be represented by
    sorted base-10 integers, which they call determinant addresses. This function
    converts a bitstring matrix to a ``pyscf`` address representation.

    To be explicit, this function separates each bitstring in ``bitstring_matrix``
    in half, translates each set of bits into integer representations, and appends
    them to their respective lists. Those lists are sorted and output from this function.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring
        open_shell: A flag specifying whether unique addresses from the left and right
            halves of the bitstrings should be kept separate. If ``False``, addresses
            from the left and right halves of the bitstrings are combined into a single
            set of unique addresses. That combined set will be returned for both the left
            and right bitstrings.

    Returns:
        A length-2 tuple of sorted, base-10 determinant addresses representing the left
        and right halves of the bitstrings, respectively.
    """
    addresses_left_set = set()
    addresses_right_set = set()
    num_orbitals = bitstring_matrix.shape[1] // 2
    for bs in bitstring_matrix:
        addresses_left_set.add(
            int(np.array2string(bs[:num_orbitals].astype(int), separator="")[1:-1], 2)
        )
        addresses_right_set.add(
            int(np.array2string(bs[num_orbitals:].astype(int), separator="")[1:-1], 2)
        )

    if open_shell:
        addresses_left = sorted(addresses_left_set)
        addresses_right = sorted(addresses_right_set)
    else:
        addresses_left = addresses_right = sorted(addresses_left_set.union(addresses_right_set))

    return addresses_left, addresses_right
