import sys 
import glob, os
import argparse
parser = argparse.ArgumentParser(description='convert')
parser.add_argument('--DB_DIR', default='./DB', type=str, help='path to db file')
parser.add_argument('--GPU', default='-1', type=str, help='GPU id')
args = parser.parse_args()
GPU = args.GPU
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
sys.path.append("../dnovo3")

from preprocessing import aa_with_pad, MAX_PEPTIDE_LENGTH
from load_model import spectrum_embedder,sequence_embedder

from tqdm import tqdm
import numpy as np
import tensorflow as tf

def get_sequence_of_indices(sequence: list, aa_list: list=list(aa_with_pad)):
    indices = [aa_list.index(aa) for aa in sequence]
    return indices

def trim_sequence(sequence:str,MAX_PEPTIDE_LENGTH=MAX_PEPTIDE_LENGTH):
    sequence = sequence.ljust(MAX_PEPTIDE_LENGTH,'_')
    return list(sequence)

AUTOTUNE=tf.data.experimental.AUTOTUNE

DB_DIR=args.DB_DIR
BATCH_SIZE=4*4096

def parse_peptide_(peptide):
    peptide = trim_sequence(peptide)
    peptide = get_sequence_of_indices(peptide)
    return peptide

if __name__ == '__main__':    
    peptides = np.load(os.path.join(DB_DIR,"peptides.npy"))#[:1000000]

    peptides = list(map(trim_sequence,tqdm(peptides)))
    peptides = list(map(get_sequence_of_indices,tqdm(peptides)))
    
    peptides = np.array(peptides,dtype=np.int32)
    print(peptides.shape)

    def get_dataset(peptides,batch_size=BATCH_SIZE):
        def peptide_generator():
            for i in range(0, len(peptides), batch_size):
                yield peptides[i:i + batch_size]
        ds = tf.data.Dataset.from_generator(peptide_generator,tf.int32)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    peptides_ds = get_dataset(peptides)

    for p in tqdm(peptides_ds):
        pass

    for _ in tqdm(range(1)):
        embedded_peptides = sequence_embedder.predict(peptides_ds)

    print(len(embedded_peptides))

    np.save(os.path.join(DB_DIR,"embedded_peptides.npy"),embedded_peptides)