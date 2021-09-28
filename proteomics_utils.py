import numpy as np
from pyteomics import mass
import copy 
aa_comp = copy.deepcopy(mass.std_aa_comp)
#aa_comp['M'] = aa_comp['M'] + mass.Composition({'O':1})    

def precursor2peptide_mass(precursor_mz,charge):
  proton = mass.calculate_mass(formula='H')
  return (precursor_mz - proton)*charge

def theoretical_peptide_mass(peptide,charge,average=True):
    return mass.fast_mass(peptide,type='M',charge=charge,average=average,aa_comp=aa_comp)

def trim_peaks_list(mzs,intensities,MAX_N_PEAKS,pad=True):
    if mzs.shape[-1]<=MAX_N_PEAKS and pad:
        mzs = np.pad(mzs,((0,MAX_N_PEAKS-(mzs.shape[-1]))), 'constant', constant_values=0)
        intensities = np.pad(intensities,((0,MAX_N_PEAKS-(intensities.shape[-1]))), 'constant', constant_values=0)    
        return mzs,intensities
    else:
        indices = np.argsort(intensities)[-MAX_N_PEAKS:][::-1] # take only highest=MAX_N_PEAKS peaks
        # use boolean mask to keep order according to original mz-sorting:
        mask = np.zeros_like(intensities).astype(bool)
        mask[indices] = True
        return mzs[mask], intensities[mask]