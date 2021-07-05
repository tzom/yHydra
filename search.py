import sys

from scipy.spatial.distance import euclidean
from tensorflow.python.eager.context import device 
sys.path.append("..")
import glob, os, json
import tensorflow as tf
device = '/GPU:0'
if True:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    device = '/CPU:0'
else:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0],True)



import random
from tf_data_json import USIs,parse_json_npy,parse_usi
from tf_data_mgf import MGF
from load_model import spectrum_embedder,sequence_embedder
from fasta2db import theoretical_peptide_mass

from tqdm import tqdm
import numpy as np

AUTOTUNE=tf.data.AUTOTUNE

#DB_DIR = './db'
DB_DIR = './db_miscleav_1'
#DB_DIR = '../../Neonomicon/PXD007963/db'

db_embedded_peptides = np.load(os.path.join(DB_DIR,"embedded_peptides.npy"))
db_peptides = np.load(os.path.join(DB_DIR,"peptides.npy"))
db_pepmasses = np.load(os.path.join(DB_DIR,"pepmasses.npy"))

sorted_indices = np.argsort(db_pepmasses)

db_embedded_peptides=db_embedded_peptides[sorted_indices]
db_peptides=db_peptides[sorted_indices]
db_pepmasses=db_pepmasses[sorted_indices]

DELTA_MASS = 200

if True:
    ####### MASS BUCKETS #######
    ######################################
    def bucket_indices(X,n_buckets=10):
        from sklearn.preprocessing import KBinsDiscretizer
        est = KBinsDiscretizer(n_bins=n_buckets, encode='ordinal', strategy='quantile')
        masses = np.expand_dims(X,axis=-1)
        mass_bucket_indices = np.squeeze(est.fit_transform(masses))
        #in_bucket_indices = mass_bucket_indices==bucket
        buckets = [np.arange(X.shape[0])[mass_bucket_indices==bucket] for bucket in range(n_buckets)]
        #print(np.bincount(mass_bucket_indices.astype(np.int32)))
        #print(est.bin_edges_)
        return buckets,est

    def get_lowest_highest_bucket(est,mass,delta_mass=DELTA_MASS):
        lowest,highest = np.squeeze(est.transform(np.expand_dims([mass-delta_mass,mass+delta_mass],axis=-1)))
        return int(lowest),int(highest)

    def get_space(mass,est,buckets):
        lowest, highest = get_lowest_highest_bucket(est,mass=mass)
        space = np.concatenate(buckets[lowest:highest+1])
        return space

    buckets,est = bucket_indices(db_pepmasses,100)

    space = get_space(mass=4014.23,est=est,buckets=buckets)

    ####### MASS BUCKETS #######
    ######################################

print('fire up datasets...')
N = 10000
#N = 163410

if True:
    files = glob.glob(os.path.join('../../Neonomicon/PXD007963/**/','*.json'))
    #files = glob.glob(os.path.join('../../Neonomicon/files/test/**/','*.json'))
    #files = glob.glob(os.path.join('../../Neonomicon/dump','*.json'))
    random.seed(0)
    random.shuffle(files)

    ds = USIs(files[:N],batch_size=1,buffer_size=1).get_dataset().unbatch()
    ds_spectra = ds.map(lambda x,y: x).batch(64)
    ds_peptides = ds.map(lambda x,y: y).batch(256)

if False:
    file =  "../../crux_mock_search/PXD006118/Run1_U4_2000ng.mgf"
    ds = MGF([file]).get_dataset().take(N).unbatch()
    ds_spectra = ds.map(lambda x,y: x).batch(256)
    ds = MGF([file]).get_dataset().take(N).unbatch()
    ds_scans = ds.map(lambda x,y: y).batch(1).as_numpy_iterator()

true_peptides = []
true_pepmasses = []

input_specs_npy = {
    "mzs": np.float32,
    "intensities": np.float32,
    "usi": str,
    #"charge": float,
    "precursorMZ": float,
}

if True:
    print('getting true peptides...')
    for psm in tqdm(list(map(lambda file_location: parse_json_npy(file_location,specs=input_specs_npy), files))):
        #charge=psm['charge']
        precursorMZ=float(psm['precursorMZ'])
        usi=str(psm['usi'])
        collection_identifier, run_identifier, index, charge, peptideSequence, positions = parse_usi(usi)
        true_peptides.append(peptideSequence)
        true_pepmasses.append(float(charge)*precursorMZ)

    #theoretical_pepmasses = np.array(list(map(theoretical_peptide_mass,true_peptides)))

#print(list(zip(true_pepmasses,theoretical_pepmasses)))
with tf.device(device):
    print('embedding spectra...')
    for _ in tqdm(range(1)):        
        embedded_spectra = spectrum_embedder.predict(ds_spectra)
        #print('embedding peptides...')
        #embedded_peptides = sequence_embedder.predict(ds_peptides)

def append_dim(X,new_dim,axis=1):
    return np.concatenate((X, np.expand_dims(new_dim,axis=axis)), axis=axis)

embedded_spectra = embedded_spectra

from sklearn.neighbors import NearestNeighbors
import faiss

#query = embedded_peptides
query = embedded_spectra
db = db_embedded_peptides


#query = append_dim(embedded_peptides,true_pepmasses)
#query = append_dim(embedded_spectra,theoretical_pepmasses)
#db = append_dim(db_embedded_peptides,db_pepmasses)

#query = np.expand_dims(true_pepmasses,axis=-1)
#db = np.expand_dims(db_pepmasses,axis=-1)

norm = lambda x : np.sqrt(np.inner(x,x))
diff = lambda x,y : norm(x-y)

def get_index(DB,k=50,metric='euclidean',method='sklearn',use_gpu=False):
    print('indexing...')
    if method=='sklearn':
        if metric=='euclidean':
            p=2        
        for _ in tqdm(range(1)):
            index = NearestNeighbors(n_neighbors=k,p=p,n_jobs=1)
            index.fit(DB)
        return index

    if method=='faiss':
        for _ in tqdm(range(1)):
            d = DB.shape[-1]
            if metric=='euclidean':
                index_flat = faiss.IndexFlatL2(d)
                #index_flat = faiss.IndexIVFFlat(index_flat, d, 100)
            if use_gpu:
                res = faiss.StandardGpuResources()
                index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
            index_flat.add(DB)
        return index_flat

def perform_search(query,k,index,method='sklearn'):
    print('searching...')
    if method=='sklearn':           
        for _ in tqdm(range(1)):
            I = index.kneighbors(query, k, return_distance=False)
            return I
    if method=='faiss':           
        for _ in tqdm(range(1)):
            D,I = index.search(query, k)
            return I



if False:
    with open("/mnt/data/crux_mock_search/comet.psms.json",'r') as f:
        true_peptides = json.load(f)

    from preprocessing import get_sequence_of_indices,trim_sequence
    peptide_indices = list(map(lambda x: trim_sequence(get_sequence_of_indices(x),MAX_PEPTIDE_LENGTH=42), true_peptides.values()))
    peptide_indices = np.reshape(peptide_indices,(-1,42))
    db = sequence_embedder.predict(peptide_indices,batch_size=4096)
    db_peptides = np.array(list(true_peptides.values()))

print(db.shape)
k = 50


index = get_index(db,k=k,metric='euclidean',method='faiss',use_gpu=False)
I = perform_search(query=query,k=k,index=index,method='faiss')

# I = []
# for i,query_i in enumerate(query):
#     query_i = np.expand_dims(query_i,0)
#     true_pepmass = true_pepmasses[i]
#     space = get_space(true_pepmass,est=est,buckets=buckets)
#     index = get_index(db[space],k=k,metric='euclidean',method='faiss',use_gpu=False)
#     I_i = perform_search(query=query_i,k=k,index=index,method='faiss')
#     I_i = space[I_i]
#     I.append(I_i)
# I = np.array(I)
# I = np.reshape(I,(N,k))

#scans = np.array(list(ds_scans)).flatten()

k_accuracy=[]
identified_peptides = []
identified_peptides_in_topk = []

for i,k50 in tqdm(enumerate(db_peptides[I])):
    #scan = str(scans[i])
    #print(i,k50)
    try: 
        #identified_peptide = true_peptides[scan]
        identified_peptide = true_peptides[i]
        identified_peptides.append(identified_peptide)
        #print(set(k50))
        #print(set([true_peptides[scan]]))
        intersection = set(k50).intersection(set([identified_peptide]))
        if len(intersection) > 0:
            identified_peptides_in_topk.append(identified_peptide)
            k_accuracy.append(1)
        else:
            k_accuracy.append(0)

    except:
        k_accuracy.append(-1)
        #print("not_identified")

k_accuracy = np.array(k_accuracy)

print('accuracy 0:',sum(k_accuracy==0))
print('accuracy 1:',sum(k_accuracy==1))
print('accuracy-1:',sum(k_accuracy==-1))

result_peptides_set = set(db_peptides[I].flatten().tolist())

in_db = set(true_peptides).intersection(set(db_peptides))
intersection_all = result_peptides_set.intersection(set(true_peptides))
intersection_searched = result_peptides_set.intersection(set(identified_peptides))


print(len(in_db))
print(len(intersection_all))
print(len(intersection_searched))
print(len(set(identified_peptides)))
print(len(set(identified_peptides_in_topk)))

# for i,embedded_peptide in enumerate(query):
#     k_neighbours = db_embedded_peptides[I][i].tolist()
#     k_neighbours = [tuple(x) for x in k_neighbours]
#     print(tuple(embedded_peptide.tolist()) in set(k_neighbours))
