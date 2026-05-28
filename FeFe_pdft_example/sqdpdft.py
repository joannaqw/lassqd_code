import pyscf
from pyscf import fci
from mrh.my_pyscf.mcpdft.laspdft import get_mcpdft_child_class
import types
import h5py
import numpy as np
from scipy import linalg
from itertools import combinations

def load_rdms_from_hdf5(filename, nfrag):
    with h5py.File(filename, 'r') as f:
        dm1s = [None] * nfrag
        if 'casdm1frs' in f:
            dm1_group = f['casdm1frs']
            for i in range(nfrag):
                key = str(i)
                if key in dm1_group:
                    dm1s[i] = dm1_group[key][()]
        dm2s = [None] * nfrag
        if 'casdm2frs' in f:
            dm2_group = f['casdm2frs']
            for i in range(nfrag):
                key = str(i)
                if key in dm2_group:
                    dm2s[i] = dm2_group[key][()]
        
        return dm1s, dm2s
def process_loaded_1rdms(casdm1frs):
    casdm1s = np.stack([np.stack ([linalg.block_diag (*[dm1rs[iroot][ispin]
                                                            for dm1rs in casdm1frs])
                                        for ispin in (0, 1)], axis=0)
                            for iroot in range (1)], axis=0)
    return casdm1s[0]

def process_loaded_rdm2(casdm2fr,norb_cas,casdm1frs,ncas_sub):
    casdm2 = np.zeros ((norb_cas,norb_cas,norb_cas,norb_cas))
    ncas_cum = np.cumsum ([0] + ncas_sub.tolist ())
    state = 0
    # Diagonal
    for isub, dm2_r in enumerate (casdm2fr):
        i = ncas_cum[isub]
        j = ncas_cum[isub+1]
        casdm2[i:j, i:j, i:j, i:j] = dm2_r[state]
    # Off-diagonal
    for (isub1, dm1s1_r), (isub2, dm1s2_r) in combinations (enumerate (casdm1frs), 2):
        i = ncas_cum[isub1]
        j = ncas_cum[isub1+1]
        k = ncas_cum[isub2]
        l = ncas_cum[isub2+1]
        dma1, dmb1 = dm1s1_r[state][0], dm1s1_r[state][1]
        dma2, dmb2 = dm1s2_r[state][0], dm1s2_r[state][1]
        # Coulomb slice: e.g., [1,2,2,1]
        casdm2[i:j, i:j, k:l, k:l] = np.multiply.outer (dma1+dmb1, dma2+dmb2)
        casdm2[k:l, k:l, i:j, i:j] = casdm2[i:j, i:j, k:l, k:l].transpose (2,3,0,1)
        # Exchange slice: e.g., [2,1,1,2]
        casdm2[i:j, k:l, k:l, i:j] = -(np.multiply.outer (dma1, dma2)
                                       +np.multiply.outer (dmb1, dmb2)).transpose (0,3,2,1)
        casdm2[k:l, i:j, i:j, k:l] = casdm2[i:j, k:l, k:l, i:j].transpose (1,0,3,2)
    return casdm2

def sqdpdft_energy(las,nfrag,norb_cas,ncas_sub,casdm1frs,casdm2fr,mo,ot='tPBE'): 
    sqdpdft = get_mcpdft_child_class(las, ot=ot)
    rdm1 = process_loaded_1rdms(casdm1frs)
    rdm2 = process_loaded_rdm2(casdm2fr, norb_cas, casdm1frs, ncas_sub) 

    def make_one_casdm1s(ci,state=0):
        return rdm1

    def make_one_casdm2(ci,state=0):
        return rdm2

    sqdpdft.make_one_casdm1s = make_one_casdm1s
    sqdpdft.make_one_casdm2 = make_one_casdm2
    return sqdpdft.compute_pdft_energy_(mo)

