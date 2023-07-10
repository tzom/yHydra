import sys 
import glob, os
#os.environ['YHYDRA_CONFIG'] = sys.argv[1]
import sys,os 
import numpy as np
from sharedMemory import sharedMemory
from proteomics_utils import mass_isoforms_lookup, add_nested_indices, np_add_nested_indices, MAX_N_ISOFORMS_PER_PEPTIDE
from score_utils import calc_ions
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from concurrent.futures import ProcessPoolExecutor

from load_config import CONFIG
BATCH_SIZE_PEPTIDES = 4096#CONFIG['BATCH_SIZE_PEPTIDES']#*4*4096
CHUNK_SIZE = 100
from utils import batched_list, unbatched_list, define_slices, get_vectorized_func, get_global_func

N_CORES = 128#int(os.environ['SLURM_CPUS_PER_TASK'])

RESULTS_DIR = CONFIG['RESULTS_DIR']
MAX_N_FRAGMENTS = CONFIG['MAX_N_FRAGMENTS']#200
MAX_N_ISOFORMS_PER_PEPTIDE = CONFIG['MAX_N_ISOFORMS_PER_PEPTIDE']#50

def calc_masses(REVERSE_DECOY=False,usememmap=True):
    if REVERSE_DECOY:
        DB_DIR = RESULTS_DIR+'/rev/db'
    else:
        DB_DIR = RESULTS_DIR+'/forward/db'    

    peptides = np.load(os.path.join(DB_DIR,"peptides.npy"))
    print('calculate peptides masses...')
    print('N peptides: %s'%len(peptides))


    # nested_masses = np.memmap('/tmp/output', dtype=np.float32,
    #                 shape=(len(peptides),1), mode='w+')
    nested_masses_handle = sharedMemory(type=np.float64, shape=(len(peptides),MAX_N_ISOFORMS_PER_PEPTIDE),name='nested_masses')
    nested_masses = nested_masses_handle.array
    
    if len(peptides)<=CHUNK_SIZE:
        chunksize = 1
    else:
        chunksize = CHUNK_SIZE

    slices = define_slices(len(peptides),chunksize)    

    mass_isoforms_lookup_v = get_vectorized_func(mass_isoforms_lookup,signature='()->(k)')
    mass_isoforms_lookup_global_v = get_global_func(mass_isoforms_lookup_v,input_mmap=peptides,output_mmap=nested_masses)

    with Pool(64) as p:
        p.map(mass_isoforms_lookup_global_v,tqdm(slices),chunksize=chunksize)

    np.save(os.path.join(DB_DIR,"peptide_masses.npy"),nested_masses)
    nested_masses_handle.release()

    # ions_mmap = np.memmap('/tmp/output', dtype=np.float32,
    #               shape=(len(peptides),42,4), mode='w+')

    # calc_ions_v = get_vectorized_func(calc_ions,signature='()->(k,m)')
    # calc_ions_global_v = get_global_func(calc_ions_v,input_mmap=peptides,output_mmap=ions_mmap)

    # with Pool(256) as p:
    #     p.map(calc_ions_global_v,tqdm(slices),chunksize=1000)


    
if __name__=='__main__':
    
    print(N_CORES)
    calc_masses(REVERSE_DECOY=False)
