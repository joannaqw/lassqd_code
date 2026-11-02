"""SQD-PDFT (tPBE) energy of a saved LASSQD wave function (``RDMS/latest/casdm.h5``)."""

import sys
from pathlib import Path

from pyscf import gto, lib, scf

from lassqd import LASSCF, LASState, lassqd_pdft_energy

path = Path("RDMS/latest/casdm.h5")
if not path.exists():
    sys.exit(
        f"{path} not found: run lassqd_fefe_ibm.py first, which writes the LASSQD "
        "wave function there"
    )

lib.logger.TIMER_LEVEL = lib.logger.INFO
basis = {"Fe": "6-31g", "C": "6-31g", "H": "6-31g", "O": "6-31g", "N": "6-31g"}
mol = gto.M(atom="fefe.xyz", verbose=4, spin=0, charge=4, basis=basis)
mf = scf.ROHF(mol)
mf.init_guess = "atom"
mf = mf.density_fit()
mf.kernel()
las = LASSCF(mf, (5, 5), ((4, 2), (2, 4)), spin_sub=(3, 3))

state = LASState.load(path)
e_pdft, _, _ = lassqd_pdft_energy(las, state, ot="tPBE")
print("SQDPDFT energy", e_pdft)
