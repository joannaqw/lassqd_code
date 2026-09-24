import itertools

import pytest
from pyscf import gto, scf
from qiskit import QuantumCircuit

from lassqd import LASSCFNoSymm


def full_counts(norb, nelec, count=10):
    """Counts containing every determinant with ``nelec`` electrons, so SQD spans
    the whole fragment space and is exact. Alpha is the right half of each key."""
    counts = {}
    for occ_a in itertools.combinations(range(norb), nelec[0]):
        for occ_b in itertools.combinations(range(norb), nelec[1]):
            bits = ["0"] * (2 * norb)
            for p in occ_a:
                bits[2 * norb - 1 - p] = "1"
            for p in occ_b:
                bits[norb - 1 - p] = "1"
            counts["".join(bits)] = count
    return counts


def full_space_sampler(las):
    """Stand-in for a quantum sampler: ignores the circuit, returns full-space counts."""
    return lambda circuit: [full_counts(n, ne) for n, ne in zip(las.ncas_sub, las.nelecas_sub)]


def empty_circuit(h1, h2, norb, nelec):
    return QuantumCircuit(2 * norb)


@pytest.fixture
def h4():
    """Two closed-shell H2 fragments, (2e, 2o) each, in 6-31G."""
    mol = gto.M(atom="H 0 0 0; H 0.8 0 0; H 3.0 0 0; H 3.8 0 0", basis="6-31g", verbose=0)
    mf = scf.RHF(mol).run()
    las = LASSCFNoSymm(mf, (2, 2), ((1, 1), (1, 1)), spin_sub=(1, 1))
    mo = las.localize_init_guess(([0, 1], [2, 3]), las.sort_mo([1, 2, 3, 4]))
    return mf, las, mo


@pytest.fixture
def h6():
    """Two H3 fragments with opposite spin polarization, ((2,1),(1,2)), like FeFe's
    ((4,2),(2,4))."""
    mol = gto.M(
        atom="H 0 0 0; H 0.9 0 0; H 1.8 0 0; H 4.0 0 0; H 4.9 0 0; H 5.8 0 0",
        basis="sto-3g",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    las = LASSCFNoSymm(mf, (3, 3), ((2, 1), (1, 2)), spin_sub=(2, 2))
    mo = las.localize_init_guess(([0, 1, 2], [3, 4, 5]), mf.mo_coeff)
    return mf, las, mo
