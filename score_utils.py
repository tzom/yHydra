import tensorflow as tf
import math
import numpy as np
from load_config import CONFIG

MAX_N_FRAGMENTS = CONFIG['MAX_N_FRAGMENTS']#200
TOLERANCE_DALTON = CONFIG['TOLERANCE_DALTON']#200
MAX_N_PEAKS = CONFIG['MAX_N_PEAKS']#200
MIN_MATCHING_PEAKS = CONFIG['MIN_MATCHING_PEAKS']
MAX_FRAGMENT_CHARGE = 2
MS2_TOLERANCE = CONFIG['MS2_TOLERANCE']
MS2_USE_PPM = CONFIG['MS2_USE_PPM']

from pyteomics import mass,cmass,parser
from itertools import product
from numba import jit,vectorize,float32,int32,int64, guvectorize
import numba as nb
from tqdm import tqdm
from proteomics_utils import compare_frags, aa_mass, aa_comp, create_dictionary_fromyaml

db = mass.Unimod()
# aa_comp = dict(mass.std_aa_comp)
# aa_mass = dict(mass.std_aa_mass)
fixed_mods = create_dictionary_fromyaml(CONFIG,'fixed_mods')
for aa,delta_mass in fixed_mods.items():
    aa_mass[aa] = aa_mass[aa] + delta_mass
#aa_mass['C'] = aa_mass['C'] + mass.calculate_mass(composition=db.by_title('Carbamidomethyl')['composition'])
#aa_mass['m'] = aa_mass['M'] + mass.calculate_mass(composition=db.by_title('Oxidation')['composition'])

# def get_fragments_from_sequence(peptide, types=('b','y'), maxcharge=2,aa_mass=aa_mass):
#     """
#     The function generates all possible m/z for fragments of types
#     `types` and of charges from 1 to `maxharge`.
#     """
#     fragmented_peptide = peptide#parser.parse(peptide,split=True)
#     for i,_ in enumerate(fragmented_peptide):
#         for ion_type in types:
#             for charge in range(1, maxcharge+1):
#                 #print(fragmented_peptide[:(i+1)])
#                 #print(fragmented_peptide[(i):])
#                 if ion_type[0] in 'abc':
#                     yield cmass.fast_mass(fragmented_peptide[:(i+1)], ion_type=ion_type, charge=charge, aa_mass=aa_mass)
#                 else:
#                     yield cmass.fast_mass(fragmented_peptide[i:], ion_type=ion_type, charge=charge, aa_mass=aa_mass)

def get_fragments_from_sequence(peptide, types=('b','y'), maxcharge=2,aa_mass=aa_mass):
    """
    The function generates all possible m/z for fragments of types
    `types` and of charges from 1 to `maxharge`.
    """
    ions = np.empty((len(peptide),len(types)*maxcharge),dtype=np.float32) 
    
    for i,_ in enumerate(peptide):
        counter=0
        for ion_type in types:
            for charge in range(1, maxcharge+1):
              if ion_type[0] in 'abc':
                  ions[i,counter] = cmass.fast_mass(peptide[:(i+1)], ion_type=ion_type, charge=charge, aa_mass=aa_mass)         
              else:
                  ions[i,counter] = cmass.fast_mass(peptide[i:], ion_type, charge)
              counter+=1      
    out = np.sort(ions,axis=0)       
    out.resize((42,len(types)*maxcharge))
    return out
                    
def trim_ions(ions:int,MAX_N_FRAGMENTS):
    if len(ions)<=MAX_N_FRAGMENTS:
        ions = np.pad(ions,((0,MAX_N_FRAGMENTS-(ions.shape[0]))), 'constant', constant_values=0)
        return ions
    else:
        return ions[:MAX_N_FRAGMENTS] #TODO: this has to be replaced! Longer Peptides should be discarded or increase MAX_PEPTIDE_LENGTH

def calc_ions(x):
    #peptideSequence,charge = x
    #peptideSequence = x
    ions = get_fragments_from_sequence(x,maxcharge=MAX_FRAGMENT_CHARGE,aa_mass=aa_mass)
    #ions = np.array(sorted(get_fragments_from_sequence(x,maxcharge=MAX_FRAGMENT_CHARGE,aa_mass=aa_mass)))
    #ions = trim_ions(ions,MAX_N_FRAGMENTS=MAX_N_FRAGMENTS)
    #ions.resize(MAX_N_FRAGMENTS)
    return ions

def baseline_peak_matching(q,k,v):

    def squared_dist(A,B): 
        expanded_a = tf.expand_dims(A, 2)
        expanded_b = tf.expand_dims(B, 3)
        distances = tf.sqrt(tf.math.squared_difference(expanded_a, expanded_b))
        distances = tf.reduce_sum(distances, 4)
        return distances

    # q = tf.squeeze(q)
    # k = tf.squeeze(k)
    # v = tf.squeeze(v)

    q = tf.expand_dims(q,-1)
    k = tf.expand_dims(k,-1)
    v = tf.expand_dims(v,-1)

    # print(tf.shape(q))
    # print(tf.shape(k))
    # print(tf.shape(v))

    q = tf.cast(q,tf.float32)
    k = tf.cast(k,tf.float32)
    v = tf.cast(v,tf.float32)
    
    matching_pads = tf.matmul(q,k,transpose_b=True) # Mask for matching pads
    matching_pads = tf.where(matching_pads==0.0,0.0,1.0) # Mask for matching pads

    D = squared_dist(k,q)
    eps = tf.keras.backend.epsilon()
    D = tf.where(D<TOLERANCE_DALTON,1./(D+eps),0.0)
    D *= matching_pads # set to zero, where pads matched
    D = tf.reduce_max(D, axis=-2, keepdims=True)
    D = tf.where(D>0.0,1.0,0.0)

    N_MATCHES = tf.reduce_sum(D,(-1), keepdims=True)    
    MIN_MATCH = tf.where(N_MATCHES>float(MIN_MATCHING_PEAKS),N_MATCHES,-1.0)

    D *= MIN_MATCH

    #factorial = tf.squeeze(N_MATCHES)
    #factorial = tf.math.pow(factorial,12)
    #factorial = tf.exp(tf.math.lgamma(factorial + 1.0))
    #factorial = tf.clip_by_value(factorial,0,10.0**30)
    
    output = tf.matmul(D,tf.pow(v,0.5)) 
    return output, D, None

def scoring(mzs:np.array, intensities:np.array, ions:np.array): 
    q,k,v = ions,mzs,intensities # [batch_size,topk_ions],[batch_size,n_peaks],[batch_size,n_peaks]
    

    #q = positional_encoding_tf(q,embed_dim)
    k = tf.expand_dims(k,1)
    #k = positional_encoding_tf(k,embed_dim)    
    v = tf.expand_dims(v,1)
    #v = tf.expand_dims(v,-1)
    #x,_ = scaled_dot_product_attention(q,k,v,mask=None)
    
    x,_,factorial = baseline_peak_matching(q,k,v)

    pos_score = tf.reduce_sum(x,axis=(-2,-1))# * factorial   
    best_score = tf.reduce_max(pos_score,axis=-1)
    best_score_index = tf.argmax(pos_score,axis=-1)

    #
    #print(best_score)
    #print(tf.shape(pos_score))

    # q,k,v = ions,mzs,intensities

    # q = positional_encoding_tf(q,embed_dim)
    # k = positional_encoding_tf(k,embed_dim)
    # v = tf.expand_dims(v,-1)
    
    
    # x,_ = scaled_dot_product_attention(q,k,v,mask=None)
    # neg_score = tf.reduce_sum(x,axis=(-2,-1))

    #print(neg_score)
    #pos_scores.extend(pos_score)
    #neg_scores.extend(neg_score)

    return np.int32(best_score_index), np.float32(best_score), pos_score

# @jit(nopython=True)
# def compare_frags(query_frag: np.ndarray, db_frag: np.ndarray, frag_tol: float, ppm:bool=False) -> np.ndarray:
#     """Compare query and database frags and find hits
#     Args:
#         query_frag (np.ndarray): Array with query fragments.
#         db_frag (np.ndarray): Array with database fragments.
#         frag_tol (float): Fragment tolerance for search.
#         ppm (bool, optional): Use ppm as unit or Dalton. Defaults to False.
#     Returns:
#         np.ndarray: Array with reported hits.
#     """
#     q_max = len(query_frag)
#     d_max = len(db_frag)
#     hits = np.zeros(d_max, dtype=np.int16)
#     q, d = 0, 0  # q > query, d > database
#     while q < q_max and d < d_max:
#         mass1 = query_frag[q]
#         mass2 = db_frag[d]
#         delta_mass = mass1 - mass2

#         if ppm:
#             sum_mass = mass1 + mass2
#             mass_difference = 2 * delta_mass / sum_mass * 1e6
#         else:
#             mass_difference = delta_mass

#         if abs(mass_difference) <= frag_tol:
#             hits[d] = q + 1  # Save query position +1 (zero-indexing)
#             d += 1
#             q += 1  # Only one query for each db element
#         elif delta_mass < 0:
#             q += 1
#         elif delta_mass > 0:
#             d += 1

#     return hits

@jit(nopython=True)
def baseline_peak_matching_np(q,k,v):
    B,T = q.shape[0],q.shape[1]
    scores = np.zeros((B,T))
    for b in range(B):
        for t in range(T):
            q_ = q[b,t]
            k_ = k[b]
            v_ = v[b]

            q_ = q_[q_.nonzero()]
            k_ = k_[k_.nonzero()]
            v_ = v_[v_.nonzero()]
            
            hits = compare_frags(q_,k_,'first',MS2_TOLERANCE,MS2_USE_PPM)
            
            if np.sum(hits!=0)<MIN_MATCHING_PEAKS:
                scores[b,t] = -1.0
            else:
                N = q_.shape[0]
                b_1_mask = np.arange(N)[0::2]+1
                b_2_mask = np.arange(N)[1::2]+1
                y_1_mask = np.arange(N)[2::2]+1
                y_2_mask = np.arange(N)[3::2]+1
                
                b_1 = np.array([v_[i] for i,index in enumerate(hits) if index in b_1_mask])
                b_2 = np.array([v_[i] for i,index in enumerate(hits) if index in b_2_mask])
                y_1 = np.array([v_[i] for i,index in enumerate(hits) if index in y_1_mask])
                y_2 = np.array([v_[i] for i,index in enumerate(hits) if index in y_2_mask])

                N_b = len(b_1)+len(b_2)
                N_y = len(y_1)+len(y_2)
                hyperscore = (np.sum(b_1)+np.sum(b_2)) + (np.sum(y_1)+np.sum(y_2))
                hyperscore = hyperscore * N_b * N_y
                scores[b,t] = hyperscore
    return scores

@jit(nopython=True)
def scoring_np(mzs:np.array, intensities:np.array, ions:np.array): 
    q,k,v = ions,mzs,intensities # [batch_size,topk_ions],[batch_size,n_peaks],[batch_size,n_peaks]
    x = baseline_peak_matching_np(q,k,v)
    pos_score = x
    best_score = np.max(pos_score)
    best_score_index = np.argmax(pos_score)

    return np.int32(best_score_index), np.float32(best_score), pos_score

@jit(nopython=True)
def get_matching_intensities(query_frag,query_intensity,db_frag,ppm=True,frag_tol=20.):
    q_max = query_frag.shape[0]
    d_max = db_frag.shape[0]
    frag_tol = np.float32(frag_tol)
    
    #hits = np.zeros(d_max, dtype=np.int64)
    matches = np.zeros(d_max, dtype=np.float32)
    q, d = 0, 0  # q > query, d > database
    while q < q_max and d < d_max:
        mass1 = query_frag[q]
        mass2 = db_frag[d]
        if mass2 == 0.0:
            return matches
        delta_mass = mass1 - mass2               
        if ppm:
            sum_mass = mass1 + mass2
            mass_difference = 2 * delta_mass / sum_mass * 1e6
        else:
            mass_difference = delta_mass
        if abs(mass_difference) <= frag_tol:
           matches[d] = query_intensity[q]
           d += 1
           q += 1  # Only one query for each db element
        elif delta_mass < 0:
           q += 1
        elif delta_mass > 0:
           d += 1
    return matches

@jit(nopython=True,parallel=True)
def log_factorial(arr):
  N = len(arr)#.shape
  out = np.empty(N,dtype=np.float32)
  for i in nb.prange(N):
    out[i] = np.log(np.math.gamma(arr[i]+1))
  return out

LOG_FACTORIAL_LOOKUP_TABLE_100 = np.array([  0.       ,   0.       ,   0.6931472,   1.7917595,   3.1780539,
                                             4.787492 ,   6.5792513,   8.525162 ,  10.604603 ,  12.801827 ,
                                            15.104413 ,  17.502308 ,  19.987215 ,  22.552164 ,  25.191221 ,
                                            27.899271 ,  30.67186  ,  33.505074 ,  36.395447 ,  39.339886 ,
                                            42.335617 ,  45.38014  ,  48.47118  ,  51.606674 ,  54.78473  ,
                                            58.003605 ,  61.261703 ,  64.55754  ,  67.88974  ,  71.25704  ,
                                            74.65823  ,  78.092224 ,  81.55796  ,  85.05447  ,  88.580826 ,
                                            92.13618  ,  95.719696 ,  99.33061  , 102.9682   , 106.63176  ,
                                            110.32064  , 114.03421  , 117.77188  , 121.53308  , 125.31727  ,
                                            129.12393  , 132.95258  , 136.80272  , 140.67392  , 144.56575  ,
                                            148.47777  , 152.40959  , 156.36084  , 160.33113  , 164.32011  ,
                                            168.32744  , 172.3528   , 176.39584  , 180.4563   , 184.53383  ,
                                            188.62817  , 192.73904  , 196.86618  , 201.00932  , 205.1682   ,
                                            209.34259  , 213.53224  , 217.73694  , 221.95644  , 226.19055  ,
                                            230.43904  , 234.70172  , 238.9784   , 243.26884  , 247.5729   ,
                                            251.8904   , 256.22113  , 260.56494  , 264.92166  , 269.2911   ,
                                            273.67313  , 278.06757  , 282.4743   , 286.89313  , 291.32394  ,
                                            295.7666   , 300.22095  , 304.68686  , 309.16418  , 313.65283  ,
                                            318.15265  , 322.6635   , 327.1853   , 331.7179   , 336.26117  ,
                                            340.81506  , 345.3794   , 349.95413  , 354.5391   , 359.13422  ],
                        dtype='float32')

@jit(nopython=True)
#@vectorize([float32(int64)])
def log_factorial_lookup(n):
    # if n > 100:
    #     raise ValueError
    return LOG_FACTORIAL_LOOKUP_TABLE_100[n]

@jit(nopython=True,parallel=True)
def fast_log_factorial(arr):
  N = np.atleast_1d(arr).shape[0]
  out = np.empty(N,dtype=np.float32)
  for i in nb.prange(N):
    out[i] = log_factorial_lookup(arr[i])
  return out

@guvectorize(['(float32[:,:,:],float32[:],int32[:])'],
              '(n,f,t)->(),()',
              target='cpu')
def post_process_scores(scores,out_a,out_b): #[n_candidates,MAX_N_FRAGMENTS,types]

  if scores.shape[0]==0:
    out_a[0] = -1.0
    out_b[0] = 0
    return 

  b_ions_0 = scores[...,0]
  b_ions_1 = scores[...,1]
  y_ions_0 = scores[...,2]
  y_ions_1 = scores[...,3]
  # b_ions = np.sum(b_ions_0,-1) + np.sum(b_ions_1,-1)
  # y_ions = np.sum(y_ions_0,-1) + np.sum(y_ions_1,-1)

  N_b_ions = np.count_nonzero(b_ions_0,1) + np.count_nonzero(b_ions_1,1)
  N_y_ions = np.count_nonzero(y_ions_0,1) + np.count_nonzero(y_ions_1,1)
  actual_hits = N_b_ions + N_y_ions  
  eps = 0.00001
  tmp = -np.ones(scores.shape[0],dtype=np.float32)
  for i in nb.prange(scores.shape[0]):
    if actual_hits[i]>=MIN_MATCHING_PEAKS:
        b_ions = np.sum(b_ions_0[i],-1) + np.sum(b_ions_1[i],-1)
        y_ions = np.sum(y_ions_0[i],-1) + np.sum(y_ions_1[i],-1)
        tmp[i] = np.log(b_ions+eps) + np.log(y_ions+eps) + log_factorial_lookup(N_b_ions[i]) + log_factorial_lookup(N_y_ions[i])
  #tmp = -np.ones(scores.shape[0],dtype=np.float32)
  #tmp = np.log(b_ions) + np.log(y_ions) + log_factorial_lookup(N_b_ions) + log_factorial_lookup(N_y_ions)
  #tmp = np.where(actual_hits>=MIN_MATCHING_PEAKS,tmp,-1.0)
  # if tmp.shape[0]>1:
  #   best_score = tmp[best_index]
  #   best_index = np.argmax(tmp)    
  # else:
  #   best_score = -1.0
  #   best_index = 0

  best_index = np.argmax(tmp)  
  best_score = tmp[best_index]    

  out_a[0] = best_score
  out_b[0] = best_index
  return
  #return out #[n_candidates]

#@jit(nopython=True,parallel=False)
# @guvectorize(['(float32[:],float32[:],float32[:,:,:],float32[:,:,:])'],
#               '(k),(k),(n,f,t)->(n,f,t)',
#               target='parallel')
def get_matching_intensities_vec(mzs,intensities,ions):
  scores = np.empty((len(ions),42,4),dtype=np.float32)
  ion_products = product(range(len(ions)),range(4))
  def assign_score_per_ion(ion_product):
    i,type_charge = ion_product
    scores[i,:,type_charge] = get_matching_intensities(mzs,intensities,ions[i,...,type_charge])
    return None
  list(map(assign_score_per_ion,ion_products))
  # for i in nb.prange(len(ions)):
  #   for type_charge in nb.prange(4):
  #       s = get_matching_intensities(mzs,intensities,ions[i,...,type_charge])
  #       scores[i,:,type_charge] = s
  # #out[...] = scores
  return scores

@guvectorize(['(float32[:], float32[:], float32[:,:], float32[:,:])'],
             '(q),(q),(f,t)->(f,t)',                
             target='cpu')
def get_matching_intensities_vectorized(mzs,intensities,ions,out):
  for type_charge in nb.prange(ions.shape[-1]):
    out[:,type_charge] = get_matching_intensities(mzs,intensities,ions[...,type_charge])
  return

def match_score_vectorized(mzs,intensities,ions):
  tmp = get_matching_intensities_vectorized(mzs,intensities,ions)
  tmp = post_process_scores(tmp)
  return tmp

@jit(nopython=True,parallel=True)
def get_matching_intensities_parallel(mzs,intensities,ions):
  scores = np.empty((len(ions),42,4),dtype=np.float32)
  for i in nb.prange(len(ions)):
    for type_charge in nb.prange(4):
        scores[i,:,type_charge] = get_matching_intensities(mzs,intensities,ions[i,...,type_charge])
  return scores

@jit(nopython=True)
def reverse_sort(a,ind):
  o = np.empty(a.shape)  
  o[ind] = a
  return o

def sort_score_desort(mzs,intensities,ions):
    results = []
    for i in tqdm(range(len(ions))): 
        _ = ions[i]
        a,b = mzs[i], intensities[i]

        #_ = np.reshape(_,-1)
        #mass_sorted = tf.argsort(_).numpy()
        #_ = np.take_along_axis(_,mass_sorted,axis=-1)

        #scores = get_matching_intensities(a,b, _)
        #scores = reverse_sort(scores,mass_sorted)
        #scores = np.reshape(scores,(-1,MAX_N_FRAGMENTS))
        scores = get_matching_intensities_parallel(a,b, _)
        scores = post_process_scores(scores)
        results.append(scores)
    return results

@jit(nopython=False,parallel=True)
def sort_score_desort_parallel(mzs,intensities,ions):
  results = []
  for i in nb.prange(len(ions)):
    _ = ions[i]
    a,b = mzs[i], intensities[i]

    _ = np.reshape(_,-1)
    mass_sorted = np.argsort(_)
    _ = _[mass_sorted]
    
    scores = get_matching_intensities(a,b, _)
    scores = reverse_sort(scores,mass_sorted)
    scores = np.reshape(scores,(-1,MAX_N_FRAGMENTS))
    scores = post_process_scores(scores)
    results.append(scores)
  return results