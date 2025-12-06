"""
Functions for creating batches of samples from a bitstring matrix.

.. currentmodule:: sqsd.subsampling

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   subsample
   postselect_and_subsample
"""

from __future__ import annotations

import numpy as np

from .configuration_recovery import post_select_by_hamming_weight


def postselect_and_subsample(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    hamming_left: int,
    hamming_right: int,
    samples_per_batch: int,
    num_batches: int,
    rand_seed: int | None = None,
) -> list[np.ndarray]:
    """
    Subsample batches of bit arrays with correct hamming weight from an input ``bitstring_matrix``.

    Bitstring samples with incorrect hamming weight on either their left or right half will not
    be sampled.

    Each individual batch will be sampled without replacement from the input ``bitstring_matrix``.
    Samples will be replaced after creation of each batch, so different batches may contain
    identical samples.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring.
        probabilities: A 1D array specifying a probability distribution over the bitstrings
        hamming_left: The target hamming weight for the left half of sampled bitstrings
        hamming_right: The target hamming weight for the right half of sampled bitstrings
        samples_per_batch: The number of samples to draw for each batch
        num_batches: The number of batches to generate
        rand_seed: A seed to control random behavior

    Returns:
        A list of bitstring matrices with correct hamming weight subsampled from the input bitstring matrix
    """
    # Post-select only bitstrings with correct hamming weight
    mask_postsel = post_select_by_hamming_weight(bitstring_matrix, hamming_left, hamming_right)
    bs_mat_postsel = bitstring_matrix[mask_postsel]
    probs_postsel = probabilities[mask_postsel]
    probs_postsel = np.abs(probs_postsel) / np.sum(np.abs(probs_postsel))

    return subsample(bs_mat_postsel, probs_postsel, samples_per_batch, num_batches, rand_seed)


def subsample(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    samples_per_batch: int,
    num_batches: int,
    rand_seed: int | None = None,
) -> list[np.ndarray]:
    """
    Subsample batches of bit arrays from an input ``bitstring_matrix``.

    Each individual batch will be sampled without replacement from the input ``bitstring_matrix``.
    Samples will be replaced after creation of each batch, so different batches may contain
    identical samples.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring.
        probabilities: A 1D array specifying a probability distribution over the bitstrings
        samples_per_batch: The number of samples to draw for each batch
        num_batches: The number of batches to generate
        rand_seed: A seed to control random behavior

    Returns:
        A list of bitstring matrices subsampled from the input bitstring matrix.
    """
    if samples_per_batch < 1:
        raise ValueError("samples_per_batch must be a positive integer.")
    if num_batches < 1:
        raise ValueError("num_batches must be a positive integer.")

    np.random.seed(rand_seed)
    num_bitstrings = bitstring_matrix.shape[0]

    # If the number of requested samples is >= the number of bitstrings, return
    # num_batches copies of the input array.
    randomly_sample = True
    if samples_per_batch >= num_bitstrings:
        randomly_sample = False
        indices = np.arange(num_bitstrings).astype("int")

    # Create batches of samples
    batches = []
    for _ in range(num_batches):
        if randomly_sample:
            indices = np.random.choice(
                np.arange(num_bitstrings).astype("int"),
                samples_per_batch,
                replace=False,
                p=probabilities,
            )

        batches.append(bitstring_matrix[indices])

    return batches
