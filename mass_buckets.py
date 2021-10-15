
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
from load_config import CONFIG

DELTA_MASS = 500
N_BUCKETS_OPEN = CONFIG['N_BUCKETS_OPEN']#12
SPREAD = 1.
MIN_PEPTIDE_MASS = CONFIG['MIN_PEPTIDE_MASS']#500
MAX_PEPTIDE_MASS = CONFIG['MAX_PEPTIDE_MASS']#5000

####### MASS BUCKETS #######
######################################
def bucket_indices(X,strategy='quantile', # quantile / uniform
                     n_buckets=10):
    
    est = KBinsDiscretizer(n_bins=n_buckets, encode='ordinal', strategy=strategy)
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

def get_lowest_highest_bucket_edges(est,mass,delta_mass=DELTA_MASS):
    lowest,highest = np.squeeze(est.transform(np.expand_dims([mass-delta_mass,mass+delta_mass],axis=-1)))
    return est.bin_edges_[0][[int(lowest),int(highest)]]

def get_space(mass,est,buckets):
    lowest, highest = get_lowest_highest_bucket(est,mass=mass)
    space = np.concatenate(buckets[lowest:highest+1])
    return space

def get_inbucket(masses,est):
    return np.squeeze(est.transform(masses)).astype(np.int16)

def add_bucket_adress(embeddings,masses,est,offset=0):
    masses = np.reshape(masses,(-1,1))
    addresses = est.transform(masses)+offset
    addresses = np.clip(addresses,0,N_BUCKETS-1)
    #masses = np.reshape(masses,(-1))    
    addresses = np.array(addresses).astype(dtype=np.uint8)    
    addresses = SPREAD * np.unpackbits(addresses, axis=1)
    addresses = addresses.astype(np.float32)
    return np.concatenate([embeddings,addresses],axis=-1)

from pyteomics import mass,cmass
def get_peptide_mass(peptide):
    return cmass.fast_mass(str(peptide),charge=0) 

####### MASS BUCKETS #######
######################################

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='convert')
    parser.add_argument('--DB_DIR', default='./DB', type=str, help='path to db file')
    args = parser.parse_args()
    DB_DIR=args.DB_DIR
    import numpy as np
    import os
    
    from tqdm import tqdm

    print('loading peptides ...')
    db_peptides = np.load(os.path.join(DB_DIR,"peptides.npy"))
    
    print('calc masses ...')
    from embed_db import p_b_map
    import multiprocessing
    #db_pepmasses = np.array(list(map(get_peptide_mass,tqdm(db_peptides))))
    with multiprocessing.pool.ThreadPool() as p:
        db_pepmasses = p_b_map(get_peptide_mass,p,db_peptides,batch_size=1000)
        db_pepmasses = np.array(db_pepmasses)
    
    inmassrange_indices =  (db_pepmasses >= MIN_PEPTIDE_MASS) & (db_pepmasses <= MAX_PEPTIDE_MASS)

    db_pepmasses = db_pepmasses[inmassrange_indices]
    db_peptides = db_peptides[inmassrange_indices]

    # print('sorting ...')
    # sorted_indices = np.argsort(db_pepmasses)
    # #db_embedded_peptides=db_embedded_peptides[sorted_indices]
    # db_peptides=db_peptides[sorted_indices]
    # db_pepmasses=db_pepmasses[sorted_indices]


    db_embedded_peptides = np.random.uniform(size=(len(db_peptides),64))
    n = 25000
    embedded_spectra = np.random.uniform(size=(n,64))
    true_pepmasses = np.random.uniform(size=(n,1))*4000.+500.

    buckets,est = bucket_indices(db_pepmasses,'uniform',N_BUCKETS_OPEN)
    print(list(map(len,buckets)))    

    db_embedded_peptides = add_bucket_adress(db_embedded_peptides,db_pepmasses,est)

    embedded_spectra = add_bucket_adress(embedded_spectra,true_pepmasses,est)

    # db_pepmasses = np.reshape(db_pepmasses,(-1,1))
    # db_addresses = est.transform(db_pepmasses)
    # db_pepmasses = np.reshape(db_pepmasses,(-1))    
    # db_addresses = np.array(db_addresses).astype(dtype=np.uint8)
    # db_addresses = SPREAD * np.unpackbits(db_addresses, axis=1)
    # db_addresses = db_addresses.astype(np.float32)
    # db_embedded_peptides = np.concatenate([db_embedded_peptides,db_addresses],axis=-1)

    # true_pepmasses = np.reshape(true_pepmasses,(-1,1))
    # spectra_addresses = est.transform(true_pepmasses)
    # true_pepmasses = np.reshape(true_pepmasses,(-1))
    # spectra_addresses = np.array(spectra_addresses).astype(dtype=np.uint8)
    # spectra_addresses = SPREAD * np.unpackbits(spectra_addresses, axis=1)
    # spectra_addresses = spectra_addresses.astype(np.float32)
    # embedded_spectra = np.concatenate([embedded_spectra,spectra_addresses],axis=-1)


    lowest_db_mass = min(db_pepmasses)
    highest_db_mass = max(db_pepmasses)

    print('lowest_db_mass: %s'%lowest_db_mass)
    print('highest_db_mass: %s'%highest_db_mass)

    db_size = len(db_peptides)

    print(est.bin_edges_[0].shape)


    for x in np.linspace(lowest_db_mass,highest_db_mass,10):


        

        in_bucket = get_inbucket([[x]],est)
        print(in_bucket,est.bin_edges_[0][in_bucket],est.bin_edges_[0][in_bucket+1])

        lo,hi = get_lowest_highest_bucket_edges(est,x,DELTA_MASS)
        space_lo = get_space(mass=lo,est=est,buckets=buckets)
        space_hi = get_space(mass=hi,est=est,buckets=buckets)

        size_of_local_search_space = set(space_lo).union(set(space_hi))

        print(lo,hi)
        #print(x,len(space_lo),len(space_hi))
        print(len(size_of_local_search_space),len(size_of_local_search_space)/db_size)