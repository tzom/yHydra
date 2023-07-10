from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
from time import sleep
from itertools import chain

def batched_list(x, batch_size):
	for i in range(0, len(x), batch_size):
		yield x[i:i + batch_size]

def unbatched_list(x):
  #return [item for sublist in x for item in sublist]
  #return list(chain.from_iterable(x))
  result = []
  for _list in x:
    result += _list
  return result

def join_list_of_indices(list_of_index_arrays,keep_splits=True):    
    if keep_splits:
        splits = list(map(len,list_of_index_arrays))
        splits = np.cumsum(splits)
        return np.concatenate(list_of_index_arrays),splits
    else:
        return np.concatenate(list_of_index_arrays)

def split_array_of_indices(index_array,splits):
    return np.split(index_array,splits)[:-1]

def zeros_like(arr):
    return np.zeros_like(arr)

def get_vectorized_func(function,signature="()->()"):
    return np.vectorize(function,otypes=[np.ndarray],signature=signature)

def define_slices(arr_len,window_size=1):
    print("defining slices...")
    slices = list(map(lambda start: slice(start, start + window_size),tqdm(range(0, arr_len - window_size, window_size))))
    return slices

def get_global_func(func,input_mmap,output_mmap,vectorized=True,transpose=False):

    def atomic_vec(slice,func,input_mmap,output_mmap):
        if not transpose:
            output_mmap[slice,...] = func(input_mmap[slice])
        else:
            #scores,indices = func(input_mmap[slice,...])
            scores,indices = func(*input_mmap(slice))
            output_mmap[slice,0] = scores
            output_mmap[slice,1] = indices
        return None

    def atomic_unvec(slice,func,input_mmap,output_mmap):
        #a = func(*input_mmap[slice])
        a = func(*input_mmap(slice))
        #a = np.resize(a,(1,2*(100+1),42,4))        
        output_mmap[slice,:a.shape[0],:,:] = a
        return None

    if vectorized:
        atomic = atomic_vec
    else:
        atomic = atomic_unvec

    global global_func     
    def global_func(slice):
        return atomic(slice,func=func,input_mmap=input_mmap,output_mmap=output_mmap)
    return global_func

def map_chunks(func,iterable,chunksize=1,N_CORES=1):
    chunks = list(batched_list(iterable,chunksize))
    total = np.ceil(len(iterable)/chunksize).astype(int)
    print(total)
    results = []
    #with Pool(N_CORES) as p:    
    #with ProcessPoolExecutor() as p: 
    with Pool(N_CORES) as p:
        tasks = p.imap(func,chunks)
    #tasks = Parallel(n_jobs=N_CORES,backend='multiprocessing',batch_size=1,pre_dispatch=N_CORES*4)(delayed(func)(chunk) for chunk in tqdm(chunks))
    for r in tqdm(tasks,total=total):
        results.extend(r)
    print('done.')
    return results 

### SOURCE: https://superfastpython.com/multiprocessing-pool-shared-global-variables/
# initialize worker processes
def init_worker(data):
    # declare scope of a new global variable
    global shared_data
    # store argument in the global variable for this process
    shared_data = data

# def map_chunks(func,iterable,p=None,chunksize=100000):      
# 	tasks = []
# 	for chunk in batched_list(iterable,chunksize):
# 		tasks.append(p.apply_async(func,[chunk]))
# 	results = []
# 	for r in tqdm(tasks):
# 		r.wait()
# 		results.extend(r.get())
# 	return results 