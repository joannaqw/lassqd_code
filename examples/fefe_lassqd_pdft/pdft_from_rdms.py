"""SQD-PDFT (tPBE) energy of a saved LASSQD wave function (``RDMS/casdm.h5``)."""

import numpy as np
from pyscf import gto, lib, scf

from lassqd import LASSCFNoSymm, lassqd_pdft_energy, load_rdms

lib.logger.TIMER_LEVEL = lib.logger.INFO
basis = {"Fe": "6-31g", "C": "6-31g", "H": "6-31g", "O": "6-31g", "N": "6-31g"}
mol = gto.M(atom="fefe.xyz", verbose=4, spin=0, charge=4, basis=basis)
mf = scf.ROHF(mol)
mf.init_guess = "atom"
mf = mf.density_fit()
mf.kernel()
las = LASSCFNoSymm(mf, (5, 5), ((4, 2), (2, 4)), spin_sub=(3, 3))

casdm1frs, casdm2fr, mo_coeff = load_rdms("RDMS/casdm.h5")
if mo_coeff is None:
    # Files written before the orbitals were stored alongside the RDMs.
    mo_coeff = np.load("current_orb.npy")
print("SQDPDFT energy", lassqd_pdft_energy(las, casdm1frs, casdm2fr, mo_coeff, ot="tPBE"))
