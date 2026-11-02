"""LASSQD: SQD as the fragment solver for LASSCF.

Exports are imported on first use, so ``lassqd.records`` can be used without
importing mrh, PySCF or Qiskit.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lassqd.hybrid import (
        CycleResult,
        FragmentCircuits,
        FragmentSamples,
        HybridResult,
        build_circuits,
        collect_samples,
        run_lassqd,
        solve_cycle,
        submit_circuits,
    )
    from lassqd.las import (
        LASSCFNoSymm,
        fragment_hamiltonians,
        initial_state,
        set_fragment_kernels,
    )
    from lassqd.pdft import lassqd_pdft_energy
    from lassqd.records import LASState
    from lassqd.sqd import FragmentSQD

_EXPORTS = {
    "CycleResult": "lassqd.hybrid",
    "FragmentCircuits": "lassqd.hybrid",
    "FragmentSQD": "lassqd.sqd",
    "FragmentSamples": "lassqd.hybrid",
    "HybridResult": "lassqd.hybrid",
    "LASSCFNoSymm": "lassqd.las",
    "LASState": "lassqd.records",
    "build_circuits": "lassqd.hybrid",
    "collect_samples": "lassqd.hybrid",
    "fragment_hamiltonians": "lassqd.las",
    "initial_state": "lassqd.las",
    "lassqd_pdft_energy": "lassqd.pdft",
    "run_lassqd": "lassqd.hybrid",
    "set_fragment_kernels": "lassqd.las",
    "solve_cycle": "lassqd.hybrid",
    "submit_circuits": "lassqd.hybrid",
}

__all__ = [
    "CycleResult",
    "FragmentCircuits",
    "FragmentSQD",
    "FragmentSamples",
    "HybridResult",
    "LASSCFNoSymm",
    "LASState",
    "build_circuits",
    "collect_samples",
    "fragment_hamiltonians",
    "initial_state",
    "lassqd_pdft_energy",
    "run_lassqd",
    "set_fragment_kernels",
    "solve_cycle",
    "submit_circuits",
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
