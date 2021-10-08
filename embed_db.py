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

def batched_list(list_of_elements:list,batch_size:int=2):
  for i in range(0, len(list_of_elements), batch_size):
      yield list_of_elements[i:i + batch_size]

def p_b_map(function,pool,elements:list,batch_size:int):
    def do_the_function_on_batch(function,batch):
      return list(map(function,batch))
    l = tqdm(batched_list(elements,batch_size))
    u = list(pool.imap(lambda x: do_the_function_on_batch(function, x),l,1))
    u = [item for sublist in u for item in sublist]
    return u

if __name__ == '__main__':
    import multiprocessing

    peptides = np.load(os.path.join(DB_DIR,"peptides.npy"))#[:1000000]

    #peptides = list(map(trim_sequence,tqdm(peptides)))
    #peptides = list(map(get_sequence_of_indices,tqdm(peptides)))

    with multiprocessing.pool.ThreadPool() as p:
        peptides = p_b_map(trim_sequence,p,peptides,batch_size=1000)
        peptides = p_b_map(get_sequence_of_indices,p,peptides,batch_size=1000)
    
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