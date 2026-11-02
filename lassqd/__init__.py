"""LASSQD: SQD as the fragment solver for LASSCF.

Exports are imported on first use, so ``lassqd.records`` can be used without
importing mrh, PySCF or Qiskit.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lassqd.circuits import (
        FragmentCircuits,
        build_circuits,
        collect_samples,
        submit_circuits,
    )
    from lassqd.hybrid import (
        CycleResult,
        HybridResult,
        run_lassqd,
        solve_fragments,
        update_orbitals,
    )
    from lassqd.las import (
        LASSCFNoSymm,
        capture_embedding,
        fragment_hamiltonians,
        initial_state,
        replay_orbital_step,
    )
    from lassqd.pdft import lassqd_pdft_energy
    from lassqd.records import (
        Embedding,
        Fragment,
        FragmentResult,
        FragmentSamples,
        LASState,
        SQDState,
    )
    from lassqd.sqd import FragmentSQD

_EXPORTS = {
    "CycleResult": "lassqd.hybrid",
    "Embedding": "lassqd.records",
    "Fragment": "lassqd.records",
    "FragmentCircuits": "lassqd.circuits",
    "FragmentResult": "lassqd.records",
    "FragmentSQD": "lassqd.sqd",
    "FragmentSamples": "lassqd.records",
    "HybridResult": "lassqd.hybrid",
    "LASSCFNoSymm": "lassqd.las",
    "LASState": "lassqd.records",
    "SQDState": "lassqd.records",
    "build_circuits": "lassqd.circuits",
    "capture_embedding": "lassqd.las",
    "collect_samples": "lassqd.circuits",
    "fragment_hamiltonians": "lassqd.las",
    "initial_state": "lassqd.las",
    "lassqd_pdft_energy": "lassqd.pdft",
    "replay_orbital_step": "lassqd.las",
    "run_lassqd": "lassqd.hybrid",
    "solve_fragments": "lassqd.hybrid",
    "submit_circuits": "lassqd.circuits",
    "update_orbitals": "lassqd.hybrid",
}

__all__ = [
    "CycleResult",
    "Embedding",
    "Fragment",
    "FragmentCircuits",
    "FragmentResult",
    "FragmentSQD",
    "FragmentSamples",
    "HybridResult",
    "LASSCFNoSymm",
    "LASState",
    "SQDState",
    "build_circuits",
    "capture_embedding",
    "collect_samples",
    "fragment_hamiltonians",
    "initial_state",
    "lassqd_pdft_energy",
    "replay_orbital_step",
    "run_lassqd",
    "solve_fragments",
    "submit_circuits",
    "update_orbitals",
]


def __getattr__(name: str):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'lassqd' has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
