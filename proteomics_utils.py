import numpy as np
from pyteomics import mass, parser, mgf
import copy 
import numpy as np
aa = parser.std_amino_acids
non_canonical = ['B','Z','X','J','U']
pad = '_'
aa_with_pad = np.concatenate([[pad],aa,non_canonical])    
len_aa = len(aa_with_pad)

aa_comp = copy.deepcopy(mass.std_aa_comp)

from load_config import CONFIG
MAX_N_PEAKS = CONFIG['MAX_N_PEAKS']
NORMALIZATION_METHOD = CONFIG['NORMALIZATION_METHOD']

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

def trim_peaks_list_v2(mzs,intensities,MAX_N_PEAKS,PAD_N_PEAKS,pad=True):
    indices = np.argsort(intensities)[-MAX_N_PEAKS:][::-1] # take only highest=MAX_N_PEAKS peaks
        # use boolean mask to keep order according to original mz-sorting:
    mask = np.zeros_like(intensities).astype(bool)
    mask[indices] = True
    mzs, intensities = mzs[mask], intensities[mask]

    mzs = np.pad(mzs,((0,PAD_N_PEAKS-(mzs.shape[0]))), 'constant', constant_values=0)
    intensities = np.pad(intensities,((0,PAD_N_PEAKS-(intensities.shape[0]))), 'constant', constant_values=0)    
    return mzs,intensities

def normalize_intensities(intensities,method=NORMALIZATION_METHOD):
    if method == "ion_current":
        denominator = np.sum(intensities**2)        
    elif method == "L2":
        denominator = np.linalg.norm(intensities,ord=2)
    elif method == "sum":   
        denominator = np.sum(intensities)
    elif method == "max":  
        denominator = np.max(intensities)
    return intensities/denominator

def get_features(entry):
    mzs = entry['m/z array']
    intensities = entry['intensity array']
    if 'scans' in entry['params']:
        scans = int(entry['params']['scans'])
    else:
        scans = int(entry['params']['title'].split('.')[-1])               
    pepmass = entry['params']['pepmass']
    pepmass = float(pepmass[0])              
    charge = int(entry['params']['charge'][0])
    mzs = np.array(mzs)
    intensities = np.array(intensities)
    mzs, intensities = mzs,normalize_intensities(intensities,method=NORMALIZATION_METHOD)
    mzs, intensities = trim_peaks_list_v2(mzs, intensities, MAX_N_PEAKS=MAX_N_PEAKS, PAD_N_PEAKS=500)

    out_dict = {'mzs':mzs,
                'intensities':intensities,
                'scans':scans,
                'precursorMZ':pepmass,
                'charge':charge,}
            
    return out_dict

def parse_mgf_npy(file_location):
    with mgf.read(file_location,use_index=True) as mgf_reader:
        spectra = iter([get_features(x) for x in mgf_reader])
    return spectra