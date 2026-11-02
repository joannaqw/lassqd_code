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

__all__ = ["fragment_rohf", "from_mo", "sampling_basis_rohf", "to_mo"]

if TYPE_CHECKING:
    from pyscf.scf.rohf import ROHF

    from lassqd.records import Fragment


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
        The ROHF object, which may not have converged (see ``mf.converged``);
        its ``mo_coeff`` are the fragment orbitals.
    """
    mf = _model_rohf(h1, h2, norb, nelec)
    mf.kernel()
    return mf


def sampling_basis_rohf(fragment: Fragment) -> ROHF:
    """The fragment's ROHF solution in its sampling basis, without solving again.

    The orbitals are the identity, with the stored occupations and orbital
    energies, and the integrals are ``fragment.h1`` and ``fragment.h2``. Use it
    to start correlated calculations, such as CCSD, in the sampling basis.

    Args:
        fragment: The fragment.

    Returns:
        An ROHF object whose ``mo_coeff`` is the identity.
    """
    # Writable copies of the read-only record arrays, for PySCF and its users.
    h1, h2 = np.array(fragment.h1), np.array(fragment.h2)
    mf = _model_rohf(h1, h2, fragment.norb, fragment.nelec)
    mf.mo_coeff = np.eye(fragment.norb)
    mf.mo_occ = np.array(fragment.mo_occ)
    mf.mo_energy = np.array(fragment.mo_energy)
    mf.converged = fragment.rohf_converged
    mf.e_tot = mf.energy_tot(mf.make_rdm1())
    return mf


def _model_rohf(
    h1: np.ndarray, h2: np.ndarray, norb: int, nelec: tuple[int, int]
) -> ROHF:
    """An unsolved ROHF object for ``(h1, h2)`` on a fake molecule."""
    neleca, nelecb = nelec
    mol = gto.M(verbose=0)
    mol.nelectron = neleca + nelecb
    mol.spin = abs(neleca - nelecb)
    mol.nao = norb
    mf = mol.ROHF()
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(norb)
    mf._eri = h2
    return mf
