from numba import jit
from numba import vectorize
import numpy as np
from pyteomics import mass, parser, mgf
from pyteomics import cmass
import copy 
import numpy as np
import ms_deisotope
from itertools import product, chain, repeat
aa = parser.std_amino_acids
non_canonical = ['B','Z','X','J','U']
pad = '_'
aa_with_pad = np.concatenate([[pad],aa,non_canonical])    
len_aa = len(aa_with_pad)

aa_comp = copy.deepcopy(mass.std_aa_comp)
aa_mass = dict(mass.std_aa_mass)
db = mass.Unimod()

from load_config import CONFIG
MAX_N_PEAKS = CONFIG['MAX_N_PEAKS']
PAD_N_PEAKS = 200
NORMALIZATION_METHOD = CONFIG['NORMALIZATION_METHOD']
MAX_SITES_PER_MOD = CONFIG['MAX_SITES_PER_MOD']
CONSIDER_VAR_MODS = True
MAX_N_ISOFORMS_PER_PEPTIDE = CONFIG['MAX_N_ISOFORMS_PER_PEPTIDE']

def precursor2peptide_mass(precursor_mz,charge):
  #proton = mass.calculate_mass(formula='H')
  proton = 1.0072765
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

def trim_peaks_list_v2(mzs,intensities,MAX_N_PEAKS,PAD_N_PEAKS=PAD_N_PEAKS,pad=True):
    indices = np.argsort(intensities)[-MAX_N_PEAKS:][::-1] # take only highest=MAX_N_PEAKS peaks
        # use boolean mask to keep order according to original mz-sorting:
    mask = np.zeros_like(intensities).astype(bool)
    mask[indices] = True
    mzs, intensities = mzs[mask], intensities[mask]
    if not pad:
      return mzs,intensities
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

def remove_precursor(mzs,intensities,precursorMZ,remove_precursor_range=(-1.5,1.5)):
  _indices = mzs >= (precursorMZ+remove_precursor_range[0])
  indices_ = mzs <= (precursorMZ+remove_precursor_range[1])
  indices = _indices & indices_  
  return mzs[~indices],intensities[~indices]

def deconv(peaks):
  deconvoluted_peaks, _ = ms_deisotope.deconvolute_peaks(peaks, averagine=ms_deisotope.peptide,scorer=ms_deisotope.MSDeconVFitter(10.))
  masses = np.array([peak.mz for peak in deconvoluted_peaks.peaks])
  intensities = np.array([peak.intensity for peak in deconvoluted_peaks.peaks])
  charges = np.array([peak.charge for peak in deconvoluted_peaks.peaks])
  index = np.argsort(masses)
  return masses[index], intensities[index], charges[index]

def get_features(entry):
    mzs = entry['m/z array']
    intensities = entry['intensity array']
    if 'scans' in entry['params']:
        scans = int(entry['params']['scans'])
    else:
        try:
           scans = int(entry['params']['title'].split(',')[0].split('index: ')[1]) 
        except:
           scans = int(entry['params']['title'].split('.')[-2])               
    pepmass = entry['params']['pepmass']
    pepmass = float(pepmass[0])              
    charge = int(entry['params']['charge'][0])
    mzs = np.array(mzs)
    intensities = np.array(intensities)
    #mzs, intensities = mzs,normalize_intensities(intensities,method=NORMALIZATION_METHOD)
    #mzs, intensities = trim_peaks_list_v2(mzs, intensities, MAX_N_PEAKS=MAX_N_PEAKS, PAD_N_PEAKS=PAD_N_PEAKS)

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

def create_dictionary_fromyaml(mods,category):
  group2mass={}
  for k,v in mods[category].items():
    comp = db.by_title(v[1])
    assert type(comp) == dict, '%s, is not UNIMOD!'%v[1]
    aa_mass[v[0]] = mass.calculate_mass(composition=comp['composition'],absolute=True)
    group2mass[k] = aa_mass[v[0]]
  return group2mass
    
def convert2tuples(dictionary):
  f = lambda k,v: (k,np.round(v,4))
  return [f(k,v) for k, v in dictionary.items()]

@jit(nopython=True)
def compare_frags(query_frag: np.ndarray, db_frag: np.ndarray, method:str, frag_tol: float, ppm:bool=False) -> np.ndarray:
    """
    SOURCE, adapted from: https://github.com/MannLabs/alphapept
    Compare query and database frags and find hits
    Args:
        query_frag (np.ndarray): Array with query fragments.
        db_frag (np.ndarray): Array with database fragments.
        method (str): 'first', accept first match or 'all', report all matches. 
        frag_tol (float): Fragment tolerance for search.
        ppm (bool, optional): Use ppm as unit or Dalton. Defaults to False.
    Returns:
        np.ndarray: Array with reported hits.
    """
    q_max = len(query_frag)
    d_max = len(db_frag)
    hits = np.zeros(d_max, dtype=np.int16)
    q, d = 0, 0  # q > query, d > database
    while q < q_max and d < d_max:
        mass1 = query_frag[q]
        mass2 = db_frag[d]
        delta_mass = mass1 - mass2

        if ppm:
            sum_mass = mass1 + mass2
            mass_difference = 2 * delta_mass / sum_mass * 1e6
        else:
            mass_difference = delta_mass

        if abs(mass_difference) <= frag_tol:
            hits[d] = q + 1  # Save query position +1 (zero-indexing)
            d += 1
            if method == 'first':
                q += 1  # Only one query for each db element
        elif delta_mass < 0:
            q += 1
        elif delta_mass > 0:
            d += 1

    return hits

# @jit(nopython=True)
def get_search_space_ppm(query: float, db: np.ndarray, mass_tol_ppm=10.) -> slice:
  actual_delta_mass = mass_tol_ppm * query / 1e6
  left = query - actual_delta_mass
  right = query + actual_delta_mass
  left = np.searchsorted(db,left,side='left')
  right = np.searchsorted(db,right,side='right')
  if left==right:
    left = left - 1
    right = right + 1
  return slice(left,right)

def get_search_space_Da(query: float, db: np.ndarray, MIN_DELTA_MASS=-150.,MAX_DELTA_MASS=500.) -> slice:
  assert MIN_DELTA_MASS<=0.0, 'MIN_DELTA_MASS needs to be negative (e.g. -150 [Da])'
  left = query + MIN_DELTA_MASS
  right = query + MAX_DELTA_MASS
  left = np.searchsorted(db,left,side='left')
  right = np.searchsorted(db,right,side='right')
  if left==right:
    left = left - 1
    right = right + 1
  return slice(left,right)

fixed_group2mass = create_dictionary_fromyaml(CONFIG,'fixed_mods')
var_group2mass = create_dictionary_fromyaml(CONFIG,'var_mods')
nterm_var_group2mass = create_dictionary_fromyaml(CONFIG,'nterm_var_mods')
cterm_var_group2mass = create_dictionary_fromyaml(CONFIG,'cterm_var_mods')

fixed_group2mass = convert2tuples(fixed_group2mass)
var_group2mass = convert2tuples(var_group2mass)
nterm_var_group2mass = convert2tuples(nterm_var_group2mass)
cterm_var_group2mass = convert2tuples(cterm_var_group2mass)

def mass_isoforms(peptide,
                  fixed_group2mass=fixed_group2mass, 
                  var_group2mass=var_group2mass,
                  nterm_var_group2mass=nterm_var_group2mass,
                  cterm_var_group2mass=cterm_var_group2mass,
                  max_mods=MAX_SITES_PER_MOD):

  fixed_mass=0.0
  for k,v in fixed_group2mass:
    fixed_mass += v*peptide.count(k)

  actual_mods = np.zeros((len(var_group2mass),max_mods+1))
  for i,(k,v) in enumerate(var_group2mass):
    for l in range(0, min(peptide.count(k),max_mods)+1):
      actual_mods[i,l] = l*v

  nterm_actual_mods = np.zeros((len(nterm_var_group2mass),max_mods+1))
  for i,(k,v) in enumerate(nterm_var_group2mass):
    if k == '*': # if amino acids == "X", means any amino acid
      nterm_actual_mods[i,0] = v
    elif peptide[0] == k:
      nterm_actual_mods[i,0] = v

  cterm_actual_mods = np.zeros((len(cterm_var_group2mass),max_mods+1))
  for i,(k,v) in enumerate(cterm_var_group2mass):
    if k == '*': # if amino acids == "X", means any amino acid
      cterm_actual_mods[i,0] = v
    elif peptide[-1] == k:
      cterm_actual_mods[i,0] = v

  all_mods = [actual_mods,nterm_actual_mods,cterm_actual_mods]
  actual_mods = np.concatenate(all_mods,axis=0)

  actual_mods = list(product(*list(actual_mods)))
  actual_mods = list(map(sum,actual_mods))
  actual_mods = list([0.0])+actual_mods # append 0.0 to have unmodified mass at pos 0
  actual_mods = np.array(list(set(actual_mods)))
  

  peptide_mass = cmass.fast_mass(str(peptide),charge=0)

  return actual_mods+fixed_mass+peptide_mass

def precompute_keys(fixed_group2mass=fixed_group2mass, 
                  var_group2mass=var_group2mass,
                  nterm_var_group2mass=nterm_var_group2mass,
                  cterm_var_group2mass=cterm_var_group2mass,
                  max_mods=MAX_SITES_PER_MOD):

  key_blueprint = ("%s"*len(var_group2mass)+":%s_%s")

  actual_mods =[[0]*(max_mods+1) for i in range(len(var_group2mass))]
  actual_keys =[['']*(max_mods+1) for i in range(len(var_group2mass))]

  for i,(k,v) in enumerate(var_group2mass):
    for l in range(0, max_mods+1):
      actual_mods[i][l] = l*v
      actual_keys[i][l] = l*k
  
  actual_keys.append([_[0] for _ in nterm_var_group2mass+[['',0]]])
  actual_keys.append([_[0] for _ in cterm_var_group2mass+[['',0]]])

  actual_keys = list(product(*list(actual_keys)))

  actual_keys = [key_blueprint%tup for tup in actual_keys]
  
  actual_mods.append([_[1] for _ in nterm_var_group2mass+[['',0]]])
  actual_mods.append([_[1] for _ in cterm_var_group2mass+[['',0]]])  
  actual_mods = list(product(*list(actual_mods)))
  actual_mods = list(map(sum,actual_mods))
  return dict(zip(actual_keys,actual_mods))

def consolidate_keys(keys_deltamass,
                     fixed_group2mass=fixed_group2mass, 
                      var_group2mass=var_group2mass,
                      nterm_var_group2mass=nterm_var_group2mass,
                      cterm_var_group2mass=cterm_var_group2mass,
                      max_mods=MAX_SITES_PER_MOD):
  
  key_blueprint = ("%s"*len(var_group2mass)+":%s_%s")
  sub_delta_masses = []
  keys = list(keys_deltamass.keys())
  for key in keys:
    subkeys = []
    for i,(k,v) in enumerate(var_group2mass):
      subkeys.append([l*k for l in range(0, key.count(k)+1)])
    subkeys.append([_[0] for _ in nterm_var_group2mass+[['',0]] if _[0] in key])
    subkeys.append([_[0] for _ in cterm_var_group2mass+[['',0]] if _[0] in key])
    subkeys = list(product(*list(subkeys)))
    subkeys = [key_blueprint%tup for tup in subkeys]
    sub_delta_masses_per_key = [keys_deltamass[subkey] for subkey in subkeys]
    sub_delta_masses.append(sub_delta_masses_per_key)

  return dict(zip(keys,sub_delta_masses))

keys_deltamass = precompute_keys()
consolidated_keys_deltamass = consolidate_keys(keys_deltamass)

def mass_isoforms_lookup(peptide,
                         consolidated_keys_deltamass = consolidated_keys_deltamass,
                         max_mods=MAX_SITES_PER_MOD,
                         consider_var_mods=CONSIDER_VAR_MODS,
                         fixed_group2mass=fixed_group2mass,):

  peptide_mass = cmass.fast_mass(str(peptide),charge=0)

  fixed_mass=0.0
  for k,v in fixed_group2mass:
    fixed_mass += v*peptide.count(k)

  if not consider_var_mods:
    return np.atleast_1d(peptide_mass+fixed_mass)

  key_blueprint = ("%s"*len(var_group2mass)+":%s_%s")

  mods_tuple = ()

  for i,(k,v) in enumerate(var_group2mass):
    counts = min(peptide.count(k),max_mods)
    mods_tuple = mods_tuple + (k*counts,)

  for i,(k,v) in enumerate(nterm_var_group2mass):
    if peptide[0] == k:
      mods_tuple = mods_tuple + (k,)

  if len(mods_tuple) < len(var_group2mass)+1:
    mods_tuple = mods_tuple + ('',)

  for i,(k,v) in enumerate(cterm_var_group2mass):
    if peptide[-1] == k:
      mods_tuple = mods_tuple + (k,)
  
  if len(mods_tuple) < len(var_group2mass)+2:
    mods_tuple = mods_tuple + ('',)

  key = key_blueprint%mods_tuple

  actual_mods = consolidated_keys_deltamass[key]
  results = peptide_mass+fixed_mass+np.array(actual_mods)
  results.resize(MAX_N_ISOFORMS_PER_PEPTIDE)  
  return results

def add_nested_indices(nested_arrays):
  #repeats = list(map(len,nested_arrays))
  #indices = list(range(len(nested_arrays)))
  #results = np.zeros(np.sum(repeats))
  #for i,r in enumerate(repeats):
  #  results[i:i+r] = np.repeat(indices[i],r)
  #return results
  #return np.concatenate(list(map(lambda x: np.repeat(*x), list(zip(indices,repeats)))))
  return np.array(list(chain.from_iterable(list(map(lambda _: repeat(_[0],len(_[1])),enumerate(nested_arrays)))))).astype('int32')

def np_add_nested_indices(nested_arrays):
  dict_numpy_arrays = {}
  def take_or_add(dictionary,key):
    try:
      return dictionary[str(key)]
    except:
      arr = np.ones(key)
      dictionary[str(key)] = arr
      return arr
  return np.concatenate(list(map(lambda _: _[0]*take_or_add(dict_numpy_arrays,len(_[1])) ,enumerate(nested_arrays))))