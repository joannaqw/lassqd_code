import h5py
import os
from pyscf import gto, scf, lib, ao2mo, tools
from mrh.my_pyscf.mcscf.lasscf_rdm2 import extremeAsynLASSCF, make_fcibox
import numpy as np
from pyscf.mcscf import avas
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
lib.logger.TIMER_LEVEL=lib.logger.INFO
mol_name='fefe.xyz'
basis_paper={'Fe': '6-31g','C':'6-31g', 'H':'6-31g','O':'6-31g','N':'6-31g'}
mol=gto.M(atom='fefe.xyz',verbose=4,spin=0,charge=4,basis=basis_paper)
mol.build()
mf=scf.ROHF(mol)
mf.init_guess='atom'
mf=mf.density_fit()
mf.kernel()
ncas,nelecas,guess_mo_coeff=avas.kernel(mf,['Fe 3d'] ,minao=mol.basis)
mo_list = [100,101,102,103,104,105,106,107,108,109]
las =extremeAsynLASSCF(mf,(5,5),((4,2),(2,4)),spin_sub =(3,3))
nfrag = 2   
ncas_sub = np.array([5,5])
norb_cas = np.sum(ncas_sub)
last_mo = np.load('current_orb.npy') 
from sqdpdft import sqdpdft_energy,load_rdms_from_hdf5
casdm1frs_sqd,casdm2fr_sqd = load_rdms_from_hdf5('RDMS/casdm.h5',nfrag)
mo_sqd = np.load('current_orb.npy')
print('SQDPDFT energy',sqdpdft_energy(las,nfrag,norb_cas,ncas_sub,casdm1frs_sqd,casdm2fr_sqd,mo_sqd,'tPBE'))
