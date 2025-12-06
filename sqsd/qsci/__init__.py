"""
Modules for performing quantum selected configuration interaction (QSCI).

.. currentmodule:: sqsd.qsci

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   solve_pyscf
   optimize_orbitals

Submodules
==========

.. autosummary::
   :toctree:

   pyscf_solver
   orbital_optimization
"""

from .orbital_optimization import optimize_orbitals
from .pyscf_solver import solve_pyscf
from .pyscf_solver2 import solve_pyscf2
__all__ = [
    "solve_pyscf",
    "optimize_orbitals", "solve_pyscf2",
]
