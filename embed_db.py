import sys 
sys.path.append("..")
from tf_data_json import parse_peptide
from check_embedding import spectrum_embedder,sequence_embedder
import glob, os
import json
from tqdm import tqdm
import numpy as np
import tensorflow as tf

AUTOTUNE=tf.data.AUTOTUNE

DB_DIR = './db_miscleav_1'

peptides = np.load(os.path.join(DB_DIR,"peptides.npy"))

def parse_peptide_(peptide):
    try:
        return parse_peptide(peptide)
    except:
        print(peptide)

peptides = list(map(parse_peptide,tqdm(peptides)))

def get_dataset(peptides,batch_size=4096):
    output_dtype = [tf.int32]
    ds = tf.data.Dataset.from_tensor_slices(peptides)
    #ds = ds.map(lambda x: tf.numpy_function(lambda x: parse_peptide(x), [x], output_dtype),
    #ds = ds.map(lambda x: tf.numpy_function(lambda x: parse_peptide(str(x)[2:-1]), [x], output_dtype),
    #            num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

peptides_ds = get_dataset(peptides)

for p in tqdm(peptides_ds):
    pass

embedded_peptides = sequence_embedder.predict(peptides_ds)

print(len(embedded_peptides))

np.save(os.path.join(DB_DIR,"embedded_peptides.npy"),embedded_peptides)