import numpy as np
import pytest
from conftest import full_counts
from pyscf import fci
from qiskit_addon_sqd.fermion import solve_sci

from examples.fefe_lassqd.sci_solver import solve_sci_batch
from lassqd import FragmentSQD


@pytest.fixture
def hamiltonian():
    # Four orbitals give 16 determinants for (3, 1) or (1, 3) electrons,
    # enough to exercise the example's 10-root solve with both spin polarizations.
    h1 = np.diag([0.0, 0.8, 1.7, 3.0])
    h1 += np.diag([0.1, 0.2, 0.3], 1) + np.diag([0.1, 0.2, 0.3], -1)
    h2 = np.zeros((4,) * 4)
    for p in range(4):
        h2[p, p, p, p] = 0.25
    return h1, h2


@pytest.mark.parametrize("nelec", [(3, 1), (1, 3)])
@pytest.mark.parametrize("nroots", [1, 10])
def test_example_batch_solver_returns_lowest_root(hamiltonian, nelec, nroots):
    h1, h2 = hamiltonian
    strings = tuple(fci.cistring.make_strings(range(4), n) for n in nelec)
    batches = [strings, (strings[0][:3], strings[1])]
    results = solve_sci_batch(batches, h1, h2, 4, nelec, nroots=nroots)
    assert len(results) == len(batches)
    for batch, result in zip(batches, results):
        ref = solve_sci(batch, h1, h2, 4, nelec, spin_sq=2.0, tol=1e-16)
        assert np.isclose(result.energy, ref.energy)
        assert np.allclose(result.rdm1, ref.rdm1)
        assert np.allclose(result.rdm2, ref.rdm2)
        assert np.allclose(result.orbital_occupancies, ref.orbital_occupancies)
    assert np.isclose(results[0].sci_state.spin_square(), 2.0)


@pytest.mark.parametrize("nelec", [(3, 1), (1, 3)])
def test_example_solver_as_fragment_kernel(hamiltonian, nelec):
    h1, h2 = hamiltonian
    h0 = 1.5
    solver = FragmentSQD(
        20,
        max_iterations=2,
        num_batches=2,
        sci_solver=solve_sci_batch,
        carryover_threshold=1e-3,
    )
    solver.counts = full_counts(4, nelec)
    e, dm1s, dm2 = solver(4, nelec, h0, h1, h2)
    ref = fci.addons.fix_spin_(fci.direct_spin1.FCI(), ss=2.0)
    assert np.isclose(e, h0 + ref.kernel(h1, h2, 4, nelec)[0])
    assert np.isclose(
        e, h0 + np.einsum("ij,sij->", h1, dm1s) + 0.5 * np.einsum("ijkl,ijkl", h2, dm2)
    )
    assert np.allclose(solver.s_hist, 2.0)
    assert all(len(strings) > 0 for strings in solver.carryover_strings)
