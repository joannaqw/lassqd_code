"""Fragment ROHF orbitals.

LASSCF hands each fragment its Hamiltonian in the LAS (localized active)
orbital basis. :meth:`lassqd.records.Fragment.from_hamiltonian` rotates it once
to the fragment's own ROHF orbitals, found by running ROHF on a fake molecule
whose integrals are the fragment Hamiltonian. Circuits are prepared and samples
are interpreted in that basis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pyscf import gto

from lassqd.records import from_mo, to_mo

__all__ = ["fragment_rohf", "from_mo", "to_mo"]

if TYPE_CHECKING:
    from pyscf.scf.rohf import ROHF


def fragment_rohf(
    h1: np.ndarray, h2: np.ndarray, norb: int, nelec: tuple[int, int]
) -> ROHF:
    """Run ROHF on the fragment Hamiltonian ``(h1, h2)`` in an orthonormal basis.

    Only ``|neleca - nelecb|`` matters: the orbitals of a spin-free Hamiltonian do
    not depend on which spin is in excess.

    Args:
        h1: One-electron integrals, shape ``(norb, norb)``.
        h2: Two-electron integrals in chemists' notation, shape ``(norb,) * 4``.
        norb: Number of fragment orbitals.
        nelec: ``(neleca, nelecb)``.

    Returns:
        The converged ROHF object; its ``mo_coeff`` are the fragment orbitals.
    """
    neleca, nelecb = nelec
    mol = gto.M(verbose=0)
    mol.nelectron = neleca + nelecb
    mol.spin = abs(neleca - nelecb)
    mol.nao = norb
    mf = mol.ROHF()
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(norb)
    mf._eri = h2
    mf.kernel()
    return mf
