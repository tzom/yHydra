import glob, json, os
if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
    use_gpu=True
else:
    use_gpu=False

import tensorflow as tf
#from read_alphapept_ms_data_hdf import parse_hdf_npy
from proteomics_utils import parse_mgf_npy
from proteomics_utils import normalize_intensities,trim_peaks_list_v2,MAX_N_PEAKS,NORMALIZATION_METHOD
from proteomics_utils import remove_precursor
from proteomics_utils import compare_frags, get_search_space_ppm, get_search_space_Da
from load_model import spectrum_embedder,sequence_embedder
from proteomics_utils import theoretical_peptide_mass,precursor2peptide_mass, mass_isoforms, mass_isoforms_lookup, add_nested_indices
from embed_db import p_b_map

from tqdm import tqdm
import numpy as np
#import multiprocessing
#from functools import reduce
from utils import batched_list, unbatched_list, get_vectorized_func

from load_config import CONFIG

K = CONFIG['K']
PREFIXED_K = 2*(K+1)#2048
#N_BUCKETS_NARROW = CONFIG['N_BUCKETS_NARROW']
#N_BUCKETS_OPEN = CONFIG['N_BUCKETS_OPEN']
BATCH_SIZE = CONFIG['BATCH_SIZE']
USE_STREAM = CONFIG['USE_STREAM']
MS1_TOLERANCE = CONFIG['MS1_TOLERANCE']
MS1_USE_PPM = CONFIG['MS1_USE_PPM']
MIN_DELTA_MASS = CONFIG['MIN_DELTA_MASS']
MAX_DELTA_MASS = CONFIG['MAX_DELTA_MASS']
OPEN_SEARCH = CONFIG['OPEN_SEARCH'] #TODO add to config
MIN_CHARGE = CONFIG['MIN_CHARGE']#2 #TODO add to config
MAX_CHARGE = CONFIG['MAX_CHARGE']#5 #TODO add to config
MIN_PEPTIDE_MASS = CONFIG['MIN_PEPTIDE_MASS']#500
MAX_PEPTIDE_MASS = CONFIG['MAX_PEPTIDE_MASS']#5000
PAD_N_PEAKS = 500
USE_CHARGE = False

AUTOTUNE=tf.data.AUTOTUNE

MSMS_OUTPUT_IN_RESULTS=True

DB_DIR = CONFIG['RESULTS_DIR']+'/forward/db'
DECOY_DB_DIR = CONFIG['RESULTS_DIR']+'/rev/db'

OUTPUT_DIR = CONFIG['RESULTS_DIR']

def load_npy_concat(files):
    _ = None
    for f in embedded_peptides_paths:
        x = np.load(f).astype(np.float32)
        if _ is None:
            _ = x
        else:
            _ = np.concatenate([_,x])
    return _

print('calc masses ...')
from embed_db import p_b_map
for _ in tqdm(range(1)):    
    peptide_masses = np.load(os.path.join(DB_DIR,"peptide_masses.npy"))
    decoy_peptide_masses = np.load(os.path.join(DECOY_DB_DIR,"peptide_masses.npy"))
    nested_masses = np.concatenate([peptide_masses,decoy_peptide_masses])

#embedded_peptides_paths = sorted(glob.glob(os.path.join(DB_DIR,"embedded_peptides*.npy")), key=os.path.getmtime)
#print(embedded_peptides_paths)
db_embedded_peptides = np.memmap(os.path.join(DB_DIR,"embedded_peptides.npy"), dtype=np.float32, mode='r', shape=(len(peptide_masses),64))
#db_embedded_peptides = load_npy_concat(embedded_peptides_paths)
#db_peptides = np.load(os.path.join(DB_DIR,"peptides.npy"))

#embedded_peptides_paths = sorted(glob.glob(os.path.join(DECOY_DB_DIR,"embedded_peptides*.npy")), key=os.path.getmtime)
#decoy_db_embedded_peptides = load_npy_concat(embedded_peptides_paths)
decoy_db_embedded_peptides = np.memmap(os.path.join(DECOY_DB_DIR,"embedded_peptides.npy"), dtype=np.float32, mode='r', shape=(len(decoy_peptide_masses),64))
#decoy_db_peptides = np.load(os.path.join(DECOY_DB_DIR,"peptides.npy"))

#from mass_buckets import bucket_indices, get_peptide_mass, MIN_PEPTIDE_MASS, MAX_PEPTIDE_MASS, add_bucket_adress


# print('concat target+decoy peptides ...')
# for _ in tqdm(range(1)):    
#     db_target_decoy_peptides = np.concatenate([db_peptides,decoy_db_peptides])



print('add indices...')
for _ in tqdm(range(1)):
    Is = np.multiply(np.ones_like(nested_masses,dtype=np.int32),np.expand_dims(np.arange(nested_masses.shape[0],dtype=np.int32),-1))
    #Is = np.reshape(Is,-1)

class index_into_target_decoy_db:
  def __init__(self,targets,decoys):
    self.n_a = targets.shape[0]
    self.targets = targets
    self.decoys = decoys
    self.shape = (self.targets.shape[0]+self.decoys.shape[0],)+self.targets.shape[1:]
  def __getitem__(self,indices):
    indices_a = indices[indices<self.n_a]
    indices_b = indices[indices>(self.n_a-1)]-self.n_a
    return np.concatenate([self.targets[indices_a],self.decoys[indices_b]])
  def astype(self,dtype):
    self.targets = self.targets.astype(dtype)
    self.decoys = self.decoys.astype(dtype)
    return self

def db_is_decoy_function(a,b):
  n_a = a.shape[0]
  n_b = b.shape[0]
  def get_is_decoy(index):
    return index>(n_a-1)
  return get_is_decoy

print('concat+sort indices...')
for _ in tqdm(range(1)):
    db_pepmasses = np.reshape(nested_masses,-1)
    print(db_pepmasses.dtype)

    sorted_index = np.argsort(db_pepmasses)
    db_pepmasses = db_pepmasses[sorted_index]    
    db_index = np.reshape(Is,-1)[sorted_index]
    print(db_pepmasses.shape)

    db_pepmasses_open = nested_masses[:,0]
    sorted_index_open = np.argsort(db_pepmasses_open)
    db_pepmasses_open = db_pepmasses_open[sorted_index_open]
    db_index_open = Is[:,0][sorted_index_open]

    #db = np.concatenate([db_embedded_peptides,decoy_db_embedded_peptides])
    db = index_into_target_decoy_db(db_embedded_peptides,decoy_db_embedded_peptides)
    db_is_decoy = db_is_decoy_function(db_embedded_peptides,decoy_db_embedded_peptides)

def ms1indices_f(x):
    # boolean_mask = compare_frags(np.array([x]),db_pepmasses,method='all',frag_tol=MS1_TOLERANCE,ppm=MS1_USE_PPM).nonzero()
    # return boolean_mask
    #return slice(boolean_mask.argmax(),boolean_mask[::-1].argmax())
    return get_search_space_ppm(x,db_pepmasses,mass_tol_ppm=MS1_TOLERANCE)

def ms1indices_f_open(x):
    include = get_search_space_Da(x,db_pepmasses_open,MIN_DELTA_MASS=MIN_DELTA_MASS,MAX_DELTA_MASS=MAX_DELTA_MASS)
    exclude = get_search_space_ppm(x,db_pepmasses_open,mass_tol_ppm=MS1_TOLERANCE)
    return exclude,include

def sorted_unique(x):
    x, ind = np.unique(x, return_index=True)
    return x[np.argsort(ind)]


# if OPEN_SEARCH:
#     _,est_open = bucket_indices(db_pepmasses_unmod,'uniform',N_BUCKETS_OPEN)
#_,est_narrow = bucket_indices(db_pepmasses,'uniform',N_BUCKETS_NARROW)


#if __name__ == '__main__':
def search(MGF,
           db = db,
           db_is_decoy = db_is_decoy,
           #db_embedded_peptides=db_embedded_peptides,
           #decoy_db_embedded_peptides=decoy_db_embedded_peptides,
           #db_target_decoy_peptides=db_target_decoy_peptides,
           #db_pepmasses=db_pepmasses,
           db_index=db_index,
           #db_pepmasses_open=db_pepmasses_open,
           db_index_open=db_index_open,):

    true_peptides = []
    true_precursorMZs = []
    true_pepmasses = []
    true_charges = []
    true_scan = []
    true_index = []
    true_mzs = []
    true_intensities = []
    preprocessed_spectra = []


    print('getting scan information...')
    for i,spectrum in enumerate(tqdm(parse_mgf_npy(MGF))):
    #for i,spectrum in enumerate(tqdm(parse_hdf_npy(MGF,calibrate_fragments=True,database_filename='/hpi/fs00/home/tom.altenburg/projects/test_alphapept/bruker_example/test_database.hdf'))):        
        mzs = spectrum['mzs']
        intensities = spectrum['intensities']

        #mzs = np.array(mzs)
        #intensities = np.array(intensities)            
        mzs_, intensities_ = mzs,normalize_intensities(intensities,method=NORMALIZATION_METHOD)
        #mzs_, intensities_ = remove_precursor(mzs_, intensities_, float(spectrum['precursorMZ']))
        mzs_, intensities_ = trim_peaks_list_v2(mzs_, intensities_, MAX_N_PEAKS=MAX_N_PEAKS, PAD_N_PEAKS=PAD_N_PEAKS)
        preprocessed_spectrum = np.stack((mzs_, intensities_),axis=-1)
        

        charge=int(spectrum['charge'])
        precursorMZ=float(spectrum['precursorMZ'])
        pepmass=precursor2peptide_mass(precursorMZ,int(charge))
        #precursorMZ=None
        #precursor_mass = float(spectrum['precursor_mass'])
        #pepmass=precursor_mass#precursor2peptide_mass(precursor_mass,int(charge))
        scan=int(spectrum['scans'])

        if pepmass<MIN_PEPTIDE_MASS:
            continue
        if pepmass>MAX_PEPTIDE_MASS:
            continue
        if charge<MIN_CHARGE:
            continue
        if charge>MAX_CHARGE:
            continue
        #if i<500000:
        #    continue

        if MSMS_OUTPUT_IN_RESULTS:
            true_mzs.append(mzs)
            true_intensities.append(intensities)

        preprocessed_spectra.append(preprocessed_spectrum)

        true_precursorMZs.append(precursorMZ)
        true_pepmasses.append(pepmass)
        true_charges.append(int(charge))
        true_scan.append(scan)
        true_index.append(i+1)
        true_peptides.append('')
        if i>500000:
            break

    print(len(true_peptides),len(true_precursorMZs),len(true_pepmasses),len(true_charges),len(true_scan),len(true_mzs),len(true_intensities))

    ######################################
    _ = np.argsort(true_pepmasses)
    true_pepmasses = np.array(true_pepmasses)[_]

    #true_mzs = list(np.array(true_mzs)[_])
    #true_intensities = list(np.array(true_intensities)[_])
    true_mzs = [true_mzs[i] for i in _]
    true_intensities = [true_intensities[i] for i in _]
    preprocessed_spectra = np.array(preprocessed_spectra)[_]
    true_precursorMZs = np.array(true_precursorMZs)[_]
    true_charges = np.array(true_charges)[_]
    true_scan = np.array(true_scan)[_]
    true_index = np.array(true_index)[_]
    true_peptides = np.array(true_peptides)[_]


    #hits = compare_frags(np.unique(true_pepmasses),db_pepmasses,method='all',frag_tol=MS1_TOLERANCE,ppm=MS1_USE_PPM)
    #in_db_index = hits!=0
    #print(len(db_pepmasses[hits!=0]))

    #print(list(zip(true_pepmasses,theoretical_pepmasses)))
    #with tf.device(device):
    print('embedding spectra...')
    for _ in tqdm(range(1)): 
        ds_spectra = np.array(preprocessed_spectra)
        if USE_CHARGE:
            ds_charge = np.array(true_charges)
            embedded_spectra = spectrum_embedder.predict([ds_spectra,ds_charge],batch_size=BATCH_SIZE)
        else:
            embedded_spectra = spectrum_embedder.predict([ds_spectra],batch_size=BATCH_SIZE)
        #print('embedding peptides...')
        #embedded_peptides = sequence_embedder.predict(ds_peptides)
    print(embedded_spectra.shape)

    def append_dim(X,new_dim,axis=1):
        return np.concatenate((X, np.expand_dims(new_dim,axis=axis)), axis=axis)

    embedded_spectra = embedded_spectra

    from sklearn.neighbors import NearestNeighbors
    import faiss

    #query = embedded_peptides
    query = embedded_spectra
    # db = np.concatenate([db_embedded_peptides,decoy_db_embedded_peptides])
    # db_is_decoy = np.concatenate([np.zeros(len(db_embedded_peptides),bool),np.ones(len(decoy_db_embedded_peptides),dtype=bool)])

    #db_n = db_index.shape[0]
    #print(db_n)
    # db = np.take_along_axis(db, db_index[:,np.newaxis], axis=0)
    # db_is_decoy = db_is_decoy[db_index]
    # db_target_decoy_peptides = db_target_decoy_peptides[db_index]

    # db = db[in_db_index,:]
    # db_is_decoy = db_is_decoy[in_db_index]
    # db_target_decoy_peptides = db_target_decoy_peptides[in_db_index]
    # db_pepmasses = db_pepmasses[in_db_index]

    ####### MASS BUCKETS #######
    ######################################

    # inmassrange_indices =  (db_pepmasses >= MIN_PEPTIDE_MASS) & (db_pepmasses <= MAX_PEPTIDE_MASS)
    # db_pepmasses = db_pepmasses[inmassrange_indices]
    # db_target_decoy_peptides = db_target_decoy_peptides[inmassrange_indices]
    # db = db[inmassrange_indices,:]
    # db_is_decoy = db_is_decoy[inmassrange_indices]

    ######################################
    ######### NARROW
    #_,est_narrow = bucket_indices(db_pepmasses,'uniform',N_BUCKETS_NARROW)

    # db_narrow = add_bucket_adress(db,db_pepmasses,est_narrow,N_BUCKETS=N_BUCKETS_NARROW)    
    # embedded_spectra_narrow = add_bucket_adress(embedded_spectra,true_pepmasses,est_narrow,0,N_BUCKETS=N_BUCKETS_NARROW)
    if False:
        def f_(x): 
            return x-np.floor(x)
        def f_0(x): 
            return split_number(x)

        # def split_number(x):
        #     x = list(str(int(np.floor(x))))
        #     pads = (6-len(x))*['0']
        #     x = pads + x
        #     x = np.array(x).astype(np.float32)
        #     return x*10.0

        # def split_number_digits(x,digits=2):
        #     x = list(str(x-np.floor(x)))
        #     x = x[2:2+digits]
        #     x = np.array(x).astype(np.float32)
        #     return x*10.0
        from numba import jit
        @jit(nopython=True)
        def split_number(n:np.float32):
            log_n =  np.int32(np.log10(n))
            ceil_log_n = np.int32(np.ceil(log_n))
            x = [(n//(10**i))%10 for i in range(ceil_log_n, -1, -1)]
            n_pads = (6-len(x))
            pads = [0.0 for i in range(n_pads)]
            x = np.array(pads + x)
            dec = np.array([10.0**i for i in range(5,-1,-1)])
            return x*dec*10.0

        @jit(nopython=True)
        def split_number_digits(n:np.float32,digits=1):
            n = (1+(n-np.floor(n)))*10**digits
            log_n =  np.int32(np.log10(n))
            ceil_log_n = np.int32(np.ceil(log_n))
            x = [(n//(10**i))%10 for i in range(ceil_log_n, -1, -1)]
            return np.array(x)[1:]

        true_pepmasses = np.array(true_pepmasses)
        app_db_pepmasses_0 = np.expand_dims(f_(db_pepmasses),-1)*1000.0
        #app_db_pepmasses_1 = np.array(list(map(split_number_digits,tqdm(db_pepmasses))))
        #app_db_pepmasses_2 = np.expand_dims(split_number(db_pepmasses),-1)*5.0
        app_db_pepmasses_2 = np.array(list(map(split_number,tqdm(db_pepmasses))))
        db_narrow = np.concatenate([db[db_index],app_db_pepmasses_2,app_db_pepmasses_0],axis=-1).astype(np.float32)   
        


        append_pepmasses_0 = np.expand_dims(f_(true_pepmasses),-1)*1000.0
        #append_pepmasses_1 = np.array(list(map(split_number_digits,tqdm(true_pepmasses))))
        #append_pepmasses_2 = np.expand_dims(split_number(true_pepmasses),-1)*5.0
        append_pepmasses_2 = np.array(list(map(split_number,tqdm(true_pepmasses))))
        embedded_spectra_narrow = np.concatenate([embedded_spectra,append_pepmasses_2,append_pepmasses_0],axis=-1).astype(np.float32)
    else:
        db_narrow = db.astype(np.float32)
        embedded_spectra_narrow = embedded_spectra.astype(np.float32)

        
    ######### NARROW
    ######################################

    ######################################
    ######### OPEN
    #_,est_open = bucket_indices(db_pepmasses,'uniform',N_BUCKETS_OPEN)
    # if OPEN_SEARCH:
    #     db_open = add_bucket_adress(db,db_pepmasses_unmod,est_open,N_BUCKETS=N_BUCKETS_OPEN)
    #     embedded_spectra_open_0 = add_bucket_adress(embedded_spectra,true_pepmasses,est_open,0,N_BUCKETS=N_BUCKETS_OPEN)

    #embedded_spectra_open_m1 = add_bucket_adress(embedded_spectra,true_pepmasses,est_open,-1,N_BUCKETS=N_BUCKETS_OPEN)
    #embedded_spectra_open_p1 = add_bucket_adress(embedded_spectra,true_pepmasses,est_open,+1,N_BUCKETS=N_BUCKETS_OPEN)
    ######### OPEN
    ######################################

    ######################################

    #query = embedded_spectra
    ####### MASS BUCKETS #######
    ######################################

    #query = append_dim(embedded_peptides,true_pepmasses)
    #query = append_dim(embedded_spectra,theoretical_pepmasses)
    #db = append_dim(db_embedded_peptides,db_pepmasses)

    #query = np.expand_dims(true_pepmasses,axis=-1)
    #db = np.expand_dims(db_pepmasses,axis=-1)

    norm = lambda x : np.sqrt(np.inner(x,x))
    diff = lambda x,y : norm(x-y)

    def get_index(DB,k=50,metric='euclidean',method='sklearn',use_gpu=use_gpu):
        #print('indexing...')
        if method=='sklearn':
            if metric=='euclidean':
                p=2        
            for _ in range(1):
                index = NearestNeighbors(n_neighbors=k,p=p,n_jobs=1)
                index.fit(DB)
            return index

        if method=='faiss':
            for _ in range(1):
                d = DB.shape[-1]
                if metric=='euclidean':
                    index_flat = faiss.IndexFlatL2(d)
                    #index_flat = faiss.IndexIVFFlat(index_flat, d, 100)
                if use_gpu:
                    res = faiss.StandardGpuResources()
                    index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
                #index_flat = faiss.IndexRefineFlat(index_flat)
                index_flat.add(DB)                
                #index_flat.k_factor = 10.0
            return index_flat

    def perform_search(query,k,index,method='sklearn'):
        #print('searching...')
        if method=='sklearn':           
            for _ in range(1):
                D,I = index.kneighbors(query, k, return_distance=True)
                return D,I
        if method=='faiss':           
            for _ in range(1):
                D,I = index.search(query, k)
                return D,I


    #index = get_index(db_narrow,k=2048,metric='euclidean',method='faiss',use_gpu=use_gpu)
    #D_narrow,I_narrow = perform_search(query=embedded_spectra_narrow,k=2048,index=index,method='faiss')

    def get_search_spaces(true_pepmasses,ms1indices_func):
        print('get local db')
        #with multiprocessing.Pool() as p:
        ms1indices = list(map(ms1indices_func,tqdm(true_pepmasses)))
        # ms1indices_f_vec = get_vectorized_func(ms1indices_f)
        # for _ in tqdm(range(1)):
        #     ms1indices = ms1indices_f_vec(true_pepmasses)
        return ms1indices

    # #f = lambda indices: db_narrow[indices]
    # f = lambda indices: db_narrow[db_index[indices]]
    # local_db = [f(indices) for indices in tqdm(ms1indices)]

    # print('get local db indices')
    # #IDs = np.arange(len(db_narrow))
    # #f = lambda indices: IDs[indices]
    # f = lambda indices: db_index[indices]
    # original_ids = [f(indices) for indices in tqdm(ms1indices)]
    # print('unique embeddings needed:',len(np.unique(np.concatenate(original_ids))))
 
    def sequential_search(embedded_spectra,ms1indices,db,db_index,knn_batch_size=1,method='faiss',use_gpu=use_gpu,open_search=False):

        def get_embeddings_by_index_fast(db,ids:list):
            splits = np.cumsum(list(map(len,ids)))[:-1]  
            ids_conc = np.concatenate(ids)
            embeddings = db[ids_conc]
            return np.split(embeddings,splits)

        def get_indices(needed_indices):      

            #original_ids = [db_index[indices] for indices in needed_indices]
            #original_ids = unbatched_list(original_ids)
            #original_ids = np.unique(original_ids)
            if type(needed_indices) is not tuple:
                start,stop = needed_indices.start,needed_indices.stop
                
                #start = min(start,db_n)
                #stop = min(stop,db_n)
                #print(start,stop)
                return slice(0,0),needed_indices#slice(start,stop)
                #original_ids = np.atleast_1d(db_index[slice(start,stop)])
                #return original_ids
            else:
                start_exclude,stop_exclude = needed_indices[0].start,needed_indices[0].stop
                start_include,stop_include = needed_indices[1].start,needed_indices[1].stop
                start_exclude = start_exclude-1
                stop_exclude = stop_exclude+1
                return slice(start_exclude,stop_exclude),slice(start_include,stop_include)
                
                #return slice(start_include,start_exclude-1),slice(stop_exclude+1,stop_include)
                #left = np.atleast_1d(db_index[slice(start_include,start_exclude-1)])
                #right = np.atleast_1d(db_index[slice(stop_exclude+1,stop_include)])
                #original_ids = np.concatenate([left,right])
                #print(original_ids.shape)
                #return np.atleast_1d(original_ids)
            
            # empty_db = np.atleast_2d(np.array([])).astype(np.float32)
            # if original_ids.shape[0]==0:
            #     return empty_db
            #original_ids = np.unique(original_ids)
            #original_ids = np.sort(original_ids)
            
        
        D,I = [],[]
        b = knn_batch_size
        print('batched indices...')
        #L = list(map(get_indices,tqdm((batched_list(ms1indices,1)))))
        L = list(map(get_indices,tqdm((ms1indices))))
        L = batched_list(L,knn_batch_size)
        # concatenate all indices per batch

        def unzip_slices(slices):
            return list(zip(*list(map(lambda x: (x.start, x.stop), slices))))

        def zip_slices(starts,stops):
            return list(map(lambda x: slice(*x),list(zip(starts,stops))))
        
        def concat_indices(x,db_index):
            exclude_slices = []
            include_slices = []
           
            for exclude,include in x:
                exclude_slices.append(exclude)
                include_slices.append(include)

            left,right = include_slices[0].start,include_slices[-1].stop           
            #include_slices = np.r_[(*include_slices,)].astype(np.int32)
            #exclude_slices = np.r_[(*exclude_slices,)].astype(np.int32)
            if not open_search:
                return db_index[left:right]
            #print(left,right, right-left)
            #return db_index[left:right]
            starts,stops = unzip_slices(exclude_slices)
            new_start = [left] + list(stops)
            new_stops = list(starts) + [right]
            new_slices = zip_slices(new_start,new_stops)
            new_slices = ['0'] + new_slices
            include_slices = np.r_[(*new_slices,)].astype(np.int32)
            #print(len(include_slices))

            #assert is_sorted(include_slices), 'include_slices not sorted %s %s'%(include_slices,exclude_slices)
            #assert is_sorted(exclude_slices), 'exclude_slices not sorted'
            
            #include_slices = np.setdiff1d(include_slices,exclude_slices)
            if include_slices.size==0:
                return np.array([],dtype=np.int32)
            db_n = db_index.shape[0]
            include_slices[-1] = min(include_slices[-1],db_n-1)
            combined_indices = db_index[include_slices]
            
            return combined_indices
            #return reduce(np.union1d, x)

        L = list(map(lambda x: concat_indices(x,db_index),tqdm(L)))
        print('lookup embeddings...')
        for _ in tqdm(range(1)):
            #E = get_embeddings_by_index_fast(db,L)
            pass
        print('faiss search...')
        #original_ids = L[0]
        #local_db = db[np.array(range(original_ids.shape[0])).astype(np.int32)]
        for i,original_ids in enumerate(tqdm(L)): 
            #print(original_ids) 
            local_db = db[original_ids]
            q = embedded_spectra[i*b:(i+1)*b]        
            index = get_index(local_db,k=PREFIXED_K,metric='euclidean',method='faiss',use_gpu=use_gpu)
            if not local_db.size: # if search space is empty
                empty_array = -np.ones((len(q),PREFIXED_K))
                D_,I_ = empty_array.astype(np.float32),empty_array.astype(np.int32)
            else:
                D_,I_ = perform_search(query=np.atleast_2d(q),k=PREFIXED_K,index=index,method='faiss')
            if original_ids.size:
                I_ = original_ids[I_]
            D.append(D_)
            I.append(I_)
        D = np.vstack(D)
        I = np.vstack(I)
        D,I
        print(D.shape,I.shape)
        return D,I
    
    print('created search spaces...')
    ms1indices = get_search_spaces(true_pepmasses,ms1indices_f)
    
    #search_space_sizes = [lambda x: x.stop - x.start for x in ms1indices]
    #print(np.max(search_space_sizes),np.mean(search_space_sizes))
    print('indexing + searching...')
    D_narrow,I_narrow = sequential_search(embedded_spectra_narrow,ms1indices,db_narrow,db_index,knn_batch_size=1,method='faiss',use_gpu=False)

    if OPEN_SEARCH:
        print('created search spaces...')
        ms1indices = get_search_spaces(true_pepmasses,ms1indices_f_open)
        print('indexing + searching...')
        D_open,I_open = sequential_search(embedded_spectra,ms1indices,db,db_index_open,knn_batch_size=16,method='faiss',use_gpu=False,open_search=True)

    # if OPEN_SEARCH:
    #   index = get_index(db_open,k=2048,metric='euclidean',method='faiss',use_gpu=use_gpu)
    #   D_open,I_open = perform_search(query=embedded_spectra_open_0,k=2048,index=index,method='faiss')    
    #   D=np.concatenate([D_narrow,D_open],axis=-1)
    #   I=np.concatenate([I_narrow,I_open],axis=-1)
    # else:
    #   D=D_narrow
    #   I=I_narrow
    # print(D.shape,I.shape)
    #index = get_index(db,k=K,metric='euclidean',method='faiss',use_gpu=use_gpu)
    #D,I = perform_search(query=embedded_spectra,k=K,index=index,method='faiss')

    # D_m1,I_m1 = perform_search(query=embedded_spectra_m1,k=K,index=index,method='faiss')
    # D_0,I_0 = perform_search(query=embedded_spectra_0,k=K,index=index,method='faiss')
    # D_p1,I_p1 = perform_search(query=embedded_spectra_p1,k=K,index=index,method='faiss')

    # D=np.concatenate([D_m1,D_0,D_p1],axis=-1)
    # I=np.concatenate([I_m1,I_0,I_p1],axis=-1)

    

    def get_avg_k(D,I,k=100):
        d_sorted = np.sort(D.flatten())
        cutoff = d_sorted[D.shape[0]*k]
        mask = D<cutoff
        i = [np.atleast_1d(I[s,mask[s,:]]) for s in range(I.shape[0])]
        d = [np.atleast_1d(D[s,mask[s,:]]) for s in range(D.shape[0])]
        return d, i

    ####### SEARCH RESULTS DATAFRAME #######
    ######################################
    import pandas as pd

    # is_decoy           =  list(db_is_decoy[I])
    # predicted_peptides = list(db_target_decoy_peptides[I])
    # predicted_distances = list(D)
    d_narrow,i_narrow = get_avg_k(D_narrow,I_narrow,k=K)
    if OPEN_SEARCH:
        d_open,i_open = get_avg_k(D_open,I_open,k=K)
        #d = d_open
        #i = i_open
        d = [np.concatenate(x) for x in list(zip(d_narrow,d_open))]
        i = [np.concatenate(x) for x in list(zip(i_narrow,i_open))]
    else:
        d = d_narrow
        i = i_narrow

    is_decoy           =  [np.atleast_1d(db_is_decoy(r)) for r in i]
    #predicted_peptides =  [np.atleast_1d(db_target_decoy_peptides[r]) for r in i]
    predicted_db_indices =  [np.atleast_1d(r) for r in i]
    predicted_distances = d

    import gc
    gc.collect()

    if not MSMS_OUTPUT_IN_RESULTS:
        true_mzs,true_intensities = None,None

    #print(len(predicted_peptides),len(predicted_distances))
    print(len(predicted_distances))
    raw_file=os.path.splitext(os.path.basename(MGF))[0]
    search_results = pd.DataFrame({
                                'raw_file':raw_file,
                                'scan':true_scan,
                                'index':true_index,
                                'is_decoy':is_decoy,
                                'precursorMZ':true_precursorMZs,    
                                'pepmass':true_pepmasses,
                                'charge':true_charges,
                                'peptide':true_peptides,
                                #'topk_peptides':predicted_peptides,
                                'topk_db_indices':predicted_db_indices,
                                'topk_distances':predicted_distances,
                                'mzs':true_mzs,
                                'intensities':true_intensities,                               
                                })

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    #search_results.to_csv(os.path.join(OUTPUT_DIR,'search_results.csv'),index=False)
    #search_results.to_hdf(os.path.join(OUTPUT_DIR,'search_results.h5'),key='search_results', mode='w')

    # with pd.HDFStore(os.path.join(OUTPUT_DIR,'search_results.h5')) as store:
    #     for i in tqdm(range(0,len(search_results),2000)):     
    #         rows = search_results.iloc[i:i+2000]
    #         store.put(raw_file+":%s"%i,rows)

    search_results.to_pickle(os.path.join(OUTPUT_DIR,'%s.search_results.pkl'%raw_file))
    # if os.path.exists(os.path.join(OUTPUT_DIR,'search_results.h5')):
    #     prev=pd.read_hdf(os.path.join(OUTPUT_DIR,'search_results.h5'),'search_results')      
    #     search_results = pd.concat([prev,search_results],ignore_index=True)
    #     search_results.to_hdf(os.path.join(OUTPUT_DIR,'search_results.h5'),key='search_results', mode='w')
    # else:
    #     search_results.to_hdf(os.path.join(OUTPUT_DIR,'search_results.h5'),key='search_results', mode='w')
    ####### SEARCH RESULTS DATAFRAME #######
    ######################################

