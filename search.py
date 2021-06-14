import sys 
sys.path.append("..")
import glob, os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
from tf_data_json import USIs,parse_json_npy,parse_usi
from check_embedding import spectrum_embedder,sequence_embedder
from fasta2db import theoretical_peptide_mass

from tqdm import tqdm
import numpy as np
import tensorflow as tf

AUTOTUNE=tf.data.AUTOTUNE

DB_DIR = './db'

db_embedded_peptides = np.load(os.path.join(DB_DIR,"embedded_peptides.npy"))
db_peptides = np.load(os.path.join(DB_DIR,"peptides.npy"))
db_pepmasses = np.load(os.path.join(DB_DIR,"pepmasses.npy"))

from sklearn.preprocessing import KBinsDiscretizer

est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
masses = np.expand_dims(db_pepmasses,axis=-1)
mass_buckets = est.fit_transform(masses)

print(np.bincount(np.squeeze(mass_buckets.astype(np.int32)))) 
print(est.bin_edges_)
quit()

N = 100

files = glob.glob(os.path.join('../../Neonomicon/files/**/','*.json'))
#files = glob.glob(os.path.join('../../Neonomicon/dump','*.json'))
random.seed(0)
random.shuffle(files)

ds = USIs(files[-N:],batch_size=1,buffer_size=1).get_dataset().unbatch()
ds_spectra = ds.map(lambda x,y: x).batch(256)
ds_peptides = ds.map(lambda x,y: y).batch(256)

true_peptides = []

for psm in tqdm(list(map(lambda file_location: parse_json_npy(file_location), files[-N:]))):
    usi = str(psm['usi'])
    collection_identifier, run_identifier, index, charge, peptideSequence, positions = parse_usi(usi)
    true_peptides.append(peptideSequence)
true_pepmasses = np.array(list(map(theoretical_peptide_mass,true_peptides)))

embedded_spectra = spectrum_embedder.predict(ds_spectra)
embedded_peptides = sequence_embedder.predict(ds_peptides)

def append_dim(X,new_dim,axis=1):
    return np.concatenate((X, np.expand_dims(new_dim,axis=axis)), axis=axis)

embedded_spectra = embedded_spectra

from sklearn.neighbors import NearestNeighbors

#query = append_dim(embedded_peptides,true_pepmasses)
query = embedded_spectra
db = db_embedded_peptides

#query = np.expand_dims(true_pepmasses,axis=-1)
#db = np.expand_dims(db_pepmasses,axis=-1)

print(db.shape)
k = 50
tree = NearestNeighbors(n_neighbors=k,p=2,n_jobs=-1)
tree.fit(db)
I = tree.kneighbors(query, k, return_distance=False)

result_peptides_set = set(db_peptides[I].flatten().tolist())

intersection = result_peptides_set.intersection(set(true_peptides))
in_db = set(true_peptides).intersection(set(db_peptides))

#print(list(zip(true_peptides,result_peptides_set)))

print(len(in_db))
print(len(intersection))

# for i,embedded_peptide in enumerate(query):
#     k_neighbours = db_embedded_peptides[I][i].tolist()
#     k_neighbours = [tuple(x) for x in k_neighbours]
#     print(tuple(embedded_peptide.tolist()) in set(k_neighbours))
