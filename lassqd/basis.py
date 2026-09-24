"""Fragment orbital bases.

LASSCF hands each fragment kernel its Hamiltonian in the LAS (localized active)
orbital basis. Both the circuit builder and the SQD solver first rotate it to the
fragment's own ROHF orbitals, found by running ROHF on a fake molecule whose
integrals are the fragment Hamiltonian.
"""

import numpy as np
from pyscf import gto


def fragment_rohf(h1, h2, norb, nelec):
    """Run ROHF on the fragment Hamiltonian ``(h1, h2)`` in an orthonormal basis.

    ``nelec`` is ``(neleca, nelecb)``. Only ``|neleca - nelecb|`` matters: the
    orbitals of a spin-free Hamiltonian do not depend on which spin is in excess.
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


def to_mo(mo_coeff, h1, h2):
    """Rotate one- and two-electron integrals into the ``mo_coeff`` basis."""
    C = mo_coeff
    h1_mo = np.einsum("pi,pr,rj->ij", C, h1, C, optimize=True)
    h2_mo = np.einsum("pi,rj,prqs,qk,sl->ijkl", C, C, h2, C, C, optimize=True)
    return h1_mo, h2_mo


def from_mo(mo_coeff, dm1s, dm2):
    """Rotate spin-separated 1-RDMs and a spin-summed 2-RDM out of the ``mo_coeff``
    basis. ``mo_coeff`` is orthonormal, so its inverse is its transpose."""
    C = mo_coeff
    dm1s = np.stack([C @ dm @ C.T for dm in dm1s], axis=0)
    dm2 = np.einsum("ip,jr,prqs,kq,ls->ijkl", C, C, dm2, C, C, optimize=True)
    return dm1s, dm2


def fragment_mo_basis(h1, h2, norb, nelec):
    """Return ``(mo_coeff, h1_mo, h2_mo)``: the fragment ROHF orbitals and the
    fragment Hamiltonian in that basis."""
    mo_coeff = fragment_rohf(h1, h2, norb, nelec).mo_coeff
    return (mo_coeff,) + to_mo(mo_coeff, h1, h2)
