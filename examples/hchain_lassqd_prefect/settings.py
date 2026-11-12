"""Constants and random-number streams shared by every stage of this example.

This module is imported in the flow process as well as in every worker, so it
imports NumPy and the standard library only: never mrh, PySCF, Qiskit or
qiskit-addon-sqd. See the README for why that boundary matters.

Every random stream is a pure function of ``(SEED, cycle, ifrag)``, so a worker
process can re-derive the stream it needs without receiving any state from the
process that ran the previous stage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

# Seed of every random stream, as in docs/lassqd.ipynb.
SEED = 12345

# --- The H20 chain of docs/lassqd.ipynb: two H10 (10e,10o) singlets. ---

# Atoms per fragment, nearest-neighbour spacing within a fragment, and the extra
# distance between the closest atoms of the two fragments, in Angstrom.
FRAGMENT_ATOMS = 10
SPACING = 1.0
GAP = 2.0
BASIS = "sto-6g"
# PySCF and mrh compare their allocations with max_memory, in MB. The SQD
# subspaces live in the same process.
MAX_MEMORY = 16_000

NFRAGS = 2
NCAS_SUB = (10, 10)
NELECAS_SUB = ((5, 5), (5, 5))
# Spin multiplicities 2S+1: both H10 fragments are singlets.
SPIN_SUB = (1, 1)

# --- Circuits and sampling. ---

# One circuit per fragment, so each fragment's counts come from its own pub.
GROUPS = ((0,), (1,))
SHOTS = 10_000
# LUCJ ansatz: one layer, nearest-neighbour same-spin interactions and
# opposite-spin pairs (p, p) for p = 0, 4, 8.
N_REPS = 1
ALPHA_BETA_STRIDE = 4

# --- SQD defaults, overridable per run through params.json. ---

SQD_DEFAULTS = {
    "max_iterations": 2,
    "num_batches": 5,
    "samples_per_batch": 100,
    "max_dim": 150,
    "trim_ratio": 0.7,
    "carryover_ratio": 1.0,
    "symmetrize_spin": True,
    # eigsh tolerance of the batch solver
    "tol": 1e-12,
}

# --- The macrocycle loop. ---

MAX_CYCLES = 10
CONV_TOL = 1e-6

# Modules whose contents define the numerical problem. problem.json records
# their digests so a worker built from a different revision is detected rather
# than silently mixed into a run.
PROBLEM_MODULES = ("settings.py", "problem.py", "sqd_settings.py")


def _root() -> list[np.random.Generator]:
    """The seed's independent child streams: sampler first, then one per fragment."""
    return np.random.default_rng(SEED).spawn(NFRAGS + 1)


def sampler_rng(cycle: int) -> np.random.Generator:
    """Macrocycle ``cycle``'s sampler stream, re-derived in every process.

    ``run_lassqd`` instead holds one sampler object whose stream advances across
    macrocycles and across the pubs of one job, and no record persists it; see
    the README's note on the serial reference.
    """
    return _root()[0].spawn(cycle + 1)[-1]


def solver_root(ifrag: int) -> np.random.Generator:
    """Fragment ``ifrag``'s solver stream, as ``docs/lassqd.ipynb`` derives it.

    ``FragmentSQD(seed=solver_root(i))`` spawns one child from it, so the serial
    reference and the distributed run give fragment ``i`` the same stream
    whatever order the fragments are solved in.
    """
    return _root()[1 + ifrag]


def sqd_options(overrides: dict | None = None) -> dict:
    """:data:`SQD_DEFAULTS` with ``overrides`` applied; unknown keys are rejected."""
    options = dict(SQD_DEFAULTS)
    if overrides:
        unknown = sorted(set(overrides) - set(options))
        if unknown:
            raise ValueError(f"unknown SQD settings: {', '.join(unknown)}")
        options.update(overrides)
    return options


def module_digests() -> dict[str, str]:
    """SHA-256 of each of :data:`PROBLEM_MODULES`, as stored in ``problem.json``."""
    here = Path(__file__).parent
    return {
        name: hashlib.sha256((here / name).read_bytes()).hexdigest()
        for name in PROBLEM_MODULES
    }
