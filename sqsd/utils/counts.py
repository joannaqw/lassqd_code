"""
Functions for transforming counts dictionaries.

.. currentmodule:: sqsd.utils.counts

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   counts_to_arrays
   generate_counts_uniform
   generate_counts_bipartite_hamming
   normalize_counts_dict
"""

from __future__ import annotations

import copy

import numpy as np


def counts_to_arrays(counts: dict[str, float | int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a counts dictionary into arrays.

    Args:
        counts: The dictionary to convert

    Returns:
        A tuple containing:
            - A 2D array representing the sampled bitstrings. Each row represents a
              bitstring, and each element is a ``bool`` representation of the
              bit's value
            - A 1D array containing the probability with which each bitstring was sampled
    """
    prob_dict = normalize_counts_dict(counts)
    bs_mat = np.array([[bit == "1" for bit in bitstring] for bitstring in prob_dict])
    freq_arr = np.array(list(prob_dict.values()))

    return bs_mat, freq_arr


def generate_counts_uniform(
    num_samples: int, num_bits: int, rand_seed: None | int = None
) -> dict[str, int]:
    """
    Generate a bitstring counts dictionary of samples drawn from the uniform distribution.

    Args:
        num_samples: The number of samples to draw
        num_bits: The number of bits in the bitstrings
        rand_seed: A seed for controlling randomness

    Returns:
        A dictionary mapping bitstrings of length ``num_bits`` to the
        number of times they were sampled.
    """
    np.random.seed(rand_seed)
    sample_dict: dict[str, int] = {}
    # Use numpy to generate a random matrix of bit values and
    # convert it to a dictionary of bitstring samples
    bts_matrix = np.random.choice([0, 1], size=(num_samples, num_bits))
    for i in range(num_samples):
        bts_arr = bts_matrix[i, :].astype("int")
        bts = np.array2string(bts_arr, separator="")[1:-1]
        sample_dict[bts] = sample_dict.get(bts, 1) + 1

    return sample_dict


def generate_counts_bipartite_hamming(
    num_samples: int,
    num_bits: int,
    hamming_left: int,
    hamming_right: int,
    rand_seed: None | int = None,
) -> dict[str, int]:
    """
    Generate a bitstring counts dictionary with specified bipartite hamming weight.

    Args:
        num_samples: The number of samples to draw
        num_bits: The number of bits in the bitstrings
        hamming_left: The hamming weight on the left half of each bitstring
        hamming_right: The hamming weight on the right half of each bitstring
        rand_seed: A seed for controlling randomness

    Returns:
        A dictionary mapping bitstrings to the number of times they were sampled.
        Each half of each bitstring in the output dictionary will have a hamming
        weight as specified by the inputs.
    """
    if num_bits % 2 != 0:
        raise ValueError("num_bits must be an even integer.")

    np.random.seed(rand_seed)

    sample_dict: dict[str, int] = {}
    for _ in range(num_samples):
        # Pick random bits to flip such that the left and right hamming weights are correct
        up_flips = np.random.choice(np.arange(num_bits // 2), hamming_left, replace=False).astype(
            "int"
        )
        dn_flips = np.random.choice(np.arange(num_bits // 2), hamming_right, replace=False).astype(
            "int"
        )

        # Create a bitstring with the chosen bits flipped
        bts_arr = np.zeros(num_bits)
        bts_arr[up_flips] = 1
        bts_arr[dn_flips + num_bits // 2] = 1
        bts_arr = bts_arr.astype("int")
        bts = np.array2string(bts_arr, separator="")[1:-1]

        # Add the bitstring to the sample dict
        sample_dict[bts] = sample_dict.get(bts, 1) + 1

    return sample_dict


def normalize_counts_dict(counts: dict[str, float | int], inplace=True) -> dict[str, float]:
    """Convert a counts dictionary into a probability dictionary."""
    if not inplace:
        counts = copy.deepcopy(counts)
    total_counts = float(sum(counts.values()))

    return {bs: count / total_counts for bs, count in counts.items()}
