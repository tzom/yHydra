import os, sys
sys.path.append("../dnovo3")
sys.path.append("../Neonomicon")
sys.path.append("../pyteomics_snippets")
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

#aa_comp['C'] = aa_comp['C'] + composition=db.by_title('Carbamidomethyl')['composition']
#aa_comp['ox'] = db.by_title('Oxidation')['composition']

def positional_encoding_tf(positions, d_model):

    def get_angles_tf(pos, i, d_model):
        pos = tf.cast(pos,tf.float32)
        i = tf.cast(i,tf.float32)
        #angle_rates = 1 / tf.math.pow(10000., (2 * (i//2)) / tf.cast(d_model,tf.float32))
        angle_rates = 1 / tf.math.pow(10000., (2 * (i//2)) / tf.cast(d_model,tf.float32))
        return pos * angle_rates

    input_shape = tf.shape(positions)
    output_shape = tuple(input_shape)+(d_model,)

    positions = tf.expand_dims(positions,axis=-1)# positions = positions[:, np.newaxis]
    i_s = tf.range(d_model)
    i_s = tf.expand_dims(i_s,axis=0) # np.arange(d_model)[np.newaxis, :]
    angle_rads = get_angles_tf(positions,
                            i_s,
                            d_model)

    # apply sin to even indices in the array; 2i
    #angle_rads[:, 0::2] = tf.math.sin(angle_rads[:, 0::2])
    evens = tf.math.sin(angle_rads[:,:,:,0::2])

    # apply cos to odd indices in the array; 2i+1
    #angle_rads[:, 1::2] = tf.math.cos(angle_rads[:, 1::2])
    odds = tf.math.cos(angle_rads[:,:,:,1::2])

    angle_rads = tf.stack([evens,odds],axis=-1)
    pos_encoding = angle_rads
    pos_encoding = tf.reshape(pos_encoding,output_shape)

    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask=None):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / (dk/2.0)#tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  #scaled_attention_logits = tf.map_fn(lambda x: tf.math.pow(x,1),scaled_attention_logits)  
  #attention_weights = scaled_attention_logits
  # apply tolerance
  attention_weights = tf.where(scaled_attention_logits>0.999,1.0,0.0) 

  # argmax along axis, to ensure peak_k from k gets "one" peak_q (at most) from q assigned.
  indices = tf.argmax(attention_weights,axis=-1)
  one_hot = tf.one_hot(indices, tf.shape(attention_weights)[-1],axis=-1)
  attention_weights *= one_hot
  #attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

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
    

def trim_ions(ions:int,MAX_N_FRAGMENTS):
    if len(ions)<=MAX_N_FRAGMENTS:
        ions = np.pad(ions,((0,MAX_N_FRAGMENTS-(ions.shape[0]))), 'constant', constant_values=0)
        return ions
    else:
        return ions[:MAX_N_FRAGMENTS] #TODO: this has to be replaced! Longer Peptides should be discarded or increase MAX_PEPTIDE_LENGTH

####### MULTIPROCESSING (before '__main__') #######
###################################################


input_specs_npy = {
    "mzs": np.float32,
    "intensities": np.float32,
    "usi": str,
    #"charge": float,
    "precursorMZ": float,
}

def parse_json_npy_(file_location): return parse_json_npy(file_location,specs=input_specs_npy)

#from get_fragments_from_sequence import get_fragments_from_sequence

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

def calc_ions(x):
    peptideSequence,charge = x 
    ions = np.array(sorted(get_fragments_from_sequence(peptideSequence,maxcharge=int(charge),aa_mass=aa_mass)))
    ions = trim_ions(ions,MAX_N_FRAGMENTS=MAX_N_FRAGMENTS)
    return ions

####### MULTIPROCESSING (before '__main__') #######
###################################################



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



def trim_peaks_list_(x): 
    from preprocessing import normalize_intensities
    from utils import batched_list,unbatched_list
    from proteomics_utils import theoretical_peptide_mass,trim_peaks_list
    mzs, intensities = x
    mzs, intensities = mzs, normalize_intensities(intensities)
    return trim_peaks_list(mzs, intensities,MAX_N_PEAKS=MAX_N_PEAKS)

if __name__ == '__main__':

    import glob, os
    from tqdm import tqdm
    import multiprocessing

    from tf_data_json import USIs, parse_json_npy
    from usi_magic import parse_usi
    from utils import batched_list
    # import seaborn as sns
    import matplotlib.pyplot as plt

    N = 640
    batch_size = 128
    print('globbing files...')
    files = glob.glob('../../scratch/USI_files/PXD007963/**/*.json')
    np.random.seed(0)
    np.random.shuffle(files)
    files = files[:N]
    ds = USIs(files=files,batch_size=batch_size).get_dataset().take(int(N/batch_size))


    true_peptides = []
    true_pepmasses = []
    true_ID = []
    true_pep_charge = []
    true_mzs = []
    true_intensities = []

    p = multiprocessing.Pool(128)
    if True:
        print('getting true peptides...')
        for psm in tqdm(list(p.map(parse_json_npy_, files))):
            #charge=psm['charge']
            precursorMZ=float(psm['precursorMZ'])
            usi=str(psm['usi'])
            collection_identifier, run_identifier, index, charge, peptideSequence, positions = parse_usi(usi)            

            true_mzs.append(psm['mzs'])
            true_intensities.append(psm['intensities'])
            true_pep_charge.append([peptideSequence,int(charge)])
            true_peptides.append(peptideSequence)
            true_pepmasses.append(float(charge)*precursorMZ)
            true_ID.append(usi)


    print('getting ions...')
    true_theor_ions = list(p.map(calc_ions,tqdm(true_pep_charge)))    
    true_theor_ions = list(batched_list(true_theor_ions,1))    # [n,topk_ions]
    
    pos_scores, neg_scores = [],[]
    for i,spectra in enumerate(tqdm(ds.as_numpy_iterator())):
        true_theor_ions_ = true_theor_ions[i:i+batch_size] # [batch_size,topk_ions]
        true_theor_ions_ = np.reshape(true_theor_ions_,(batch_size,1,-1)) #[batch_size,topk_ions,MAX_N_FRAGMENTS]

        mzs,intensities = spectra[0][:,:,0], spectra[0][:,:,1]
        mzs, intensities = zip(*list(p.map(trim_peaks_list_,list(zip(mzs,intensities)))))
        #mzs = np.expand_dims(mzs,axis=1)
        #intensities = np.expand_dims(intensities,axis=1)
        #true_theor_ions_ = np.expand_dims(true_theor_ions_,axis=1)
        #print(mzs.shape,intensities.shape,true_theor_ions_.shape)
        best_score_index, best_score, pos_score = scoring(mzs=mzs,intensities=intensities,ions=true_theor_ions_)
        _, neg_score, _ = scoring(mzs=mzs,intensities=intensities,ions=np.array(true_theor_ions[10-i-1]))
        if max(np.log(neg_score+1))>3.:
            index = np.argmax(np.log(neg_score+1))
            neg_ions = np.array(true_theor_ions[10-i-1])[0]
            print(neg_ions.shape)
            _,stemlines,_=plt.stem(mzs[index],intensities[index],linefmt='C0-',markerfmt=' ',basefmt=' ',use_line_collection=True,label='acquired peak')
            _,stemlines,_=plt.stem(neg_ions,-np.ones_like(neg_ions),linefmt='C0-',markerfmt=' ',basefmt=' ',use_line_collection=True,label='acquired peak')
            plt.setp(stemlines, 'linewidth', 0.3)
            plt.savefig('figures/fp_matches.png')
        pos_scores.extend(best_score)
        neg_scores.extend(neg_score)
    plt.figure()
    plt.hist(np.log(np.reshape(pos_scores,-1)+1),alpha=0.5,bins=50)
    plt.hist(np.log(np.reshape(neg_scores,-1)+1),alpha=0.5,bins=50)
    plt.savefig('figures/wavelet_score_dist.png')