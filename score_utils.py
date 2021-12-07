import tensorflow as tf
import numpy as np
from load_config import CONFIG

MAX_N_FRAGMENTS = CONFIG['MAX_N_FRAGMENTS']#200
TOLERANCE_DALTON = CONFIG['TOLERANCE_DALTON']#200
MAX_N_PEAKS = CONFIG['MAX_N_PEAKS']#200
MIN_MATCHING_PEAKS = CONFIG['MIN_MATCHING_PEAKS']

VARIABLE_MODS = {'ox':'M'}
VARIABLE_MODS = {}

from pyteomics import mass,cmass,parser

db = mass.Unimod()
aa_comp = dict(mass.std_aa_comp)
aa_mass = dict(mass.std_aa_mass)

aa_mass['C'] = aa_mass['C'] + mass.calculate_mass(composition=db.by_title('Carbamidomethyl')['composition'])
aa_mass['m'] = aa_mass['M'] + mass.calculate_mass(composition=db.by_title('Oxidation')['composition'])

def get_fragments_from_sequence(peptide, types=('b','y'), maxcharge=2,aa_mass=aa_mass):
    """
    The function generates all possible m/z for fragments of types
    `types` and of charges from 1 to `maxharge`.
    """
    fragmented_peptide = peptide#parser.parse(peptide,split=True)
    for i,_ in enumerate(fragmented_peptide):
        for ion_type in types:
            for charge in range(1, maxcharge+1):
                #print(fragmented_peptide[:(i+1)])
                #print(fragmented_peptide[(i):])
                if ion_type[0] in 'abc':
                    yield cmass.fast_mass(fragmented_peptide[:(i+1)], ion_type=ion_type, charge=charge, aa_mass=aa_mass)
                else:
                    yield cmass.fast_mass(fragmented_peptide[i:], ion_type=ion_type, charge=charge, aa_mass=aa_mass)
                    
def trim_ions(ions:int,MAX_N_FRAGMENTS):
    if len(ions)<=MAX_N_FRAGMENTS:
        ions = np.pad(ions,((0,MAX_N_FRAGMENTS-(ions.shape[0]))), 'constant', constant_values=0)
        return ions
    else:
        return ions[:MAX_N_FRAGMENTS] #TODO: this has to be replaced! Longer Peptides should be discarded or increase MAX_PEPTIDE_LENGTH

def calc_ions(x):
    peptideSequence,charge = x 
    ions = np.array(sorted(get_fragments_from_sequence(peptideSequence,maxcharge=int(charge),aa_mass=aa_mass)))
    ions = trim_ions(ions,MAX_N_FRAGMENTS=MAX_N_FRAGMENTS)
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

    output = tf.matmul(D, v) 
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