import multiprocessing, sys, os, glob
from collections import deque
import tensorflow as tf
from load_config import CONFIG
OUTPUT_DIR = CONFIG['RESULTS_DIR']

import pandas as pd
import numpy as np
from score_utils import calc_ions, scoring,get_matching_intensities_vectorized,post_process_scores, match_score_vectorized
from sharedMemory import sharedMemory
from cntxt import cntxt
from tqdm import tqdm
from utils import batched_list,unbatched_list,get_global_func,define_slices,get_vectorized_func, join_list_of_indices,split_array_of_indices
from proteomics_utils import normalize_intensities,theoretical_peptide_mass,trim_peaks_list_v2, remove_precursor

MAX_N_PEAKS = CONFIG['MAX_N_PEAKS']#500
BATCH_SIZE=CONFIG['BATCH_SIZE']#64
NUMBER_OF_THREADS=CONFIG['NUMBER_OF_THREADS']#64
K=CONFIG['K']#50
OPEN_SEARCH=CONFIG['OPEN_SEARCH']#50
TOPK=1
VERBOSE = False
SUBSET = None

def trim_peaks_list_(x): 
    mzs, intensities, precursor_mz = x
    mzs, intensities = remove_precursor(mzs, intensities, float(precursor_mz))
    mzs, intensities = mzs, normalize_intensities(intensities)
    return trim_peaks_list_v2(mzs, intensities,MAX_N_PEAKS=MAX_N_PEAKS,pad=False)

def post_process_psms(i,row,scores,db_index2local_index,local_peptides):
    foo = row['topk_distances']>0.0
    if len(row['topk_distances'][foo])==0:
        return ['AAAAAAA'], [True], [-1.0], [-1.0], [None]
    #topk_peptides = row['topk_peptides'][foo]
    topk_distances = row['topk_distances'][foo]
    top_peptide_is_decoy = row['is_decoy'][foo]
    topk_db_indices = row['topk_db_indices'][foo]


    best_score, best_score_index = scores[i]
    best_score, best_score_index = best_score, int(best_score_index)

    best_db_indices = topk_db_indices[best_score_index]
    top_peptide_index = db_index2local_index[best_db_indices]
    
    return [local_peptides[top_peptide_index]], [top_peptide_is_decoy[best_score_index]], [topk_distances[best_score_index]], [best_score], [0.0]

#if __name__ == '__main__':
def search_score(OUTPUT_DIR=OUTPUT_DIR):
    
    runs = glob.glob(os.path.join(OUTPUT_DIR,'*.search_results.pkl'))
    #db_indices2peptides = {}
    #db_indices2peptides = multiprocessing.Manager().dict()

    with cntxt('load search results') as t:
        fwd_peptides = np.load(os.path.join(CONFIG['RESULTS_DIR']+'/forward/db',"peptides.npy"))
        rev_peptides = np.load(os.path.join(CONFIG['RESULTS_DIR']+'/rev/db',"peptides.npy"))
        peptides = np.concatenate([fwd_peptides,rev_peptides])
        db_size=len(peptides)
        print(len(peptides))

        db_indices = np.array([]).astype(np.int32)
        for run in runs:
            search_results = pd.read_pickle(run)
            search_results = search_results[:SUBSET]
            #peptides = np.concatenate(search_results['topk_peptides'])
            #print(len(peptides))
            tmp_indices = np.concatenate(search_results['topk_db_indices'])
            db_indices = np.concatenate([db_indices,tmp_indices])
        del search_results

    with cntxt('unique indices') as t:
        u = np.unique(db_indices)
        print(len(u))
        db_index2local_index = np.zeros(u.max()+1,dtype=np.int32)
        peptides = peptides[u]            
        db_index2local_index[u]=np.arange(len(u))
        #db_index2local_index = dict(zip(u,range(len(u))))

    #     with cntxt('add to global dictionary...') as t:        
    #         db_indices2peptides.update(tqdm(zip(db_indices,peptides)))

    # with cntxt('build local index map...') as t:
    #     db_index2local_index = dict(tqdm(zip(db_indices2peptides.keys(),range(len(db_indices2peptides)))))

    # peptides = list(db_indices2peptides.values())
    slices = define_slices(len(peptides),100)
    calc_ions_vec = get_vectorized_func(calc_ions,signature="()->(f,t)")
    #memmap_ions = np.memmap('/tmp/ions.memmap',dtype=np.float32,mode='w+',shape=(len(peptides),42,4))
    memmap_ions_handle = sharedMemory(type=np.float32, shape=(len(peptides),42,4),name='memmap_ions')
    memmap_ions = memmap_ions_handle.array
    #global memmap_ions
    #memmap_ions = np.zeros(dtype=np.float32,shape=(len(peptides),42,4))
    calc_ions_global = get_global_func(calc_ions_vec,input_mmap=peptides,output_mmap=memmap_ions,vectorized=True)

    with cntxt('calculating ions...') as t:
        with multiprocessing.Pool(None,maxtasksperchild=300) as p:
            p.map(calc_ions_global,tqdm(slices),chunksize=100)

    for run in runs:
        search_results = pd.read_pickle(run)
        #search_results = search_results.drop(columns=['topk_peptides'])
        search_results = search_results[:SUBSET]
        raw_file = search_results['raw_file'][0]

        with cntxt('assign local indices...') as t:
            
            ind = list(map(lambda x:x['topk_db_indices'][x['topk_distances']>0.0],tqdm(pd.DataFrame.to_dict(search_results[['topk_db_indices','topk_distances']],orient='records'))))
            joined_ind,splits_ind = join_list_of_indices(ind)
            #lookup_map = lambda indices: list(map(lambda x: db_index2local_index[x],indices))
            #lookup_map = lambda x: db_index2local_index[x]
            #db_index2local_index_vec = get_vectorized_func(lookup_map)
            #ind = np.array(list(map(lookup_map,tqdm(joined_ind))))
            #ind = np.vectorize(db_index2local_index.__getitem__)(joined_ind)
            ind = db_index2local_index[joined_ind]
            ind = split_array_of_indices(ind,splits_ind)
            #ind = list(map(lambda x:lookup_map(x['topk_db_indices'][x['topk_distances']>0.0]),tqdm(pd.DataFrame.to_dict(search_results[['topk_db_indices','topk_distances']],orient='records'))))

        with cntxt('preprocess spectra...') as t:
            mzs, intensities,precursorMZs  = search_results.mzs, search_results.intensities, search_results.precursorMZ
            mzs, intensities = zip(*list(map(trim_peaks_list_,zip(mzs, intensities, precursorMZs))))

        def psms_func(slice):
            return mzs[slice].astype(np.float32),intensities[slice].astype(np.float32),memmap_ions[ind[slice]]

        print(psms_func(1)[0].shape,psms_func(1)[1].shape,psms_func(1)[2].shape)
        get_matching_intensities_vectorized(*psms_func(1))

        with cntxt('prepare scoring...') as t:
            #slices = define_slices(len(mzs),10)
            slices = range(len(mzs))
            #memmap_scores = np.memmap('/tmp/scores.memmap',dtype=np.float32,mode='w+',shape=(len(mzs),2))
            memmap_scores_handle = sharedMemory(type=np.float32, shape=(len(mzs),2),name='memmap_scores')
            memmap_scores = memmap_scores_handle.array
            match_score_vectorized_global = get_global_func(match_score_vectorized,input_mmap=psms_func,output_mmap=memmap_scores,vectorized=True,transpose=True)

        with cntxt('scoring...') as t:
            with multiprocessing.Pool(None,maxtasksperchild=300) as p:
                p.map(match_score_vectorized_global,tqdm(slices),chunksize=1)

        #memmap_ions = np.memmap('/tmp/ions.memmap',dtype=np.float32,mode='w+',shape=(1,))
        memmap_ions_handle.release()


        # with cntxt('prepare matching...') as t:
        #     slices = range(len(mzs))
        #     memmap_matches = np.memmap('/tmp/matches.memmap',dtype=np.float32,mode='w+',shape=(len(mzs),(OPEN_SEARCH+1)*(K+1),42,4))
        #     #global memmap_matches
        #     #memmap_matches = np.zeros(dtype=np.float32,shape=(len(mzs),(OPEN_SEARCH+1)*(K+1),42,4))
        #     get_matching_intensities_vectorized_global = get_global_func(get_matching_intensities_vectorized,input_mmap=psms_func,output_mmap=memmap_matches,vectorized=False)
        
        # with cntxt('matching...') as t:
        #     with multiprocessing.Pool(None,maxtasksperchild=300) as p:
        #         p.map(get_matching_intensities_vectorized_global,tqdm(slices),chunksize=1)

        # memmap_ions = np.memmap('/tmp/ions.memmap',dtype=np.float32,mode='w+',shape=(1,))

        # with cntxt('prepare scoring...') as t:
        #     slices = define_slices(len(mzs),100)
        #     memmap_scores = np.memmap('/tmp/scores.memmap',dtype=np.float32,mode='w+',shape=(len(mzs),2))
        #     #global memmap_scores
        #     #memmap_scores = np.zeros(dtype=np.float32,shape=(len(mzs),2))
        #     post_process_scores_global = get_global_func(post_process_scores,input_mmap=memmap_matches,output_mmap=memmap_scores,vectorized=True,transpose=True)

        # with cntxt('scoring...') as t:
        #     with multiprocessing.Pool(None,maxtasksperchild=300) as p:
        #         p.map(post_process_scores_global,tqdm(slices),chunksize=10)
        #         #deque(map(post_process_scores_global,tqdm(slices)))

        # memmap_matches = np.memmap('/tmp/matches.memmap',dtype=np.float32,mode='w+',shape=(1,))

        top_peptides = []
        top_peptide_is_decoys = []
        top_peptide_distances = []
        best_scores = []
        all_scores = []

        with cntxt('postprocess...') as t:
            for i,row in tqdm(enumerate(pd.DataFrame.to_dict(search_results[['topk_db_indices','is_decoy','topk_distances']],orient='records'))):
                top_peptide, top_peptide_is_decoy, top_peptide_distance, best_score, all_scores = post_process_psms(i,row,memmap_scores,db_index2local_index,peptides)
                top_peptides.extend(top_peptide)
                top_peptide_is_decoys.extend(top_peptide_is_decoy)
                top_peptide_distances.extend(top_peptide_distance)
                best_scores.extend(best_score)
                all_scores.extend(all_scores)
        
    
        #memmap_scores = np.memmap('/tmp/scores.memmap',dtype=np.float32,mode='w+',shape=(1,))
        memmap_scores_handle.release()

        #list(map(lambda i,row: post_process_psms(i,row,scores),tqdm(enumerate(pd.DataFrame.to_dict(search_results[['topk_db_indices','topk_distances']],orient='records')))))

        #     raw_files = store.keys()
        #     search_results_scored = pd.DataFrame()

        # with pd.HDFStore(os.path.join(OUTPUT_DIR,'search_results.h5')) as store, pd.HDFStore(os.path.join(OUTPUT_DIR,'search_results_scored.h5')) as store_out:
        #     raw_files = store.keys()
        #     search_results_scored = pd.DataFrame()
        if False:
            with multiprocessing.Pool(NUMBER_OF_THREADS) as p:
                
                peptide_charge = set()
                for key in raw_files:
                    search_results = store[key]
                    search_results = search_results[:SUBSET]
                    print('explode...' )
                    tmp = search_results[['topk_peptides','charge']].explode('topk_peptides')
                    additional_peptide_charge = list(zip(tmp.topk_peptides,tmp.charge))
                    additional_peptide_charge = set(additional_peptide_charge)
                    peptide_charge = peptide_charge.union(additional_peptide_charge)

                print('calculate ions...')
                ions = list(p.map(calc_ions,tqdm(peptide_charge)))
                peptide_charge_2_ions = dict(zip(peptide_charge,ions))
            
                for key in raw_files:
                    search_results = store[key]

                    top_peptides = []
                    top_peptide_is_decoys = []
                    top_peptide_distances = []
                    best_scores = []
                    all_scores = []


                    search_results = search_results[:SUBSET]
                    #for i,row in enumerate(search_results.iterrows()):
                    for i in tqdm(range(0,len(search_results),BATCH_SIZE)):     
                        rows = search_results.iloc[i:i+BATCH_SIZE] # TODO: fix last batch
                        true_peptide = rows['peptide'].to_numpy()
                        batched_topk_peptides = rows['topk_peptides'].to_numpy()
                        charges = rows['charge'].to_numpy()
                        
                        apparent_batch_size = len(rows)

                        topk_peptides = np.array(unbatched_list(batched_topk_peptides))

                        # isoforms = list(parser.isoforms(peptideSequence,variable_mods=VARIABLE_MODS)) 
                        # print(isoforms)
                        # for isoform in isoforms:
                        #     isoform = isoform.replace('oxM','m')
                        apparent_K = int(len(topk_peptides)/apparent_batch_size)
                        charges_tiled = np.repeat(charges,apparent_K)
                        topk_peptides_charge = list(zip(topk_peptides,charges_tiled))

                        ions = [peptide_charge_2_ions[key] for key in topk_peptides_charge]

                        # for _ in range(1):
                        #     if VERBOSE:
                        #         print('calc ions...')
                        #     ions = list(p.map(calc_ions,topk_peptides_charge))
                        
                        ions = np.reshape(ions,(apparent_batch_size,apparent_K,-1))
                        #ions = np.zeros((apparent_batch_size,k,200))
                        #print(len(ions))
                        #ions = list(batched_list(ions,batch_size))
                        #mzs = np.array(rows['mzs'])
                        #intensities = np.array(rows['intensities'])
                        mzs = rows['mzs'].tolist()
                        intensities = rows['intensities'].tolist()
                        for _ in range(1):
                            if VERBOSE:
                                print('trim peaks...')
                            mzs, intensities = zip(*list(p.map(trim_peaks_list_,list(zip(mzs,intensities)))))
                        #mzs, intensities = list(map(trim_ions,rows['mzs'])),list(map(trim_ions,rows['intensities']))
                        #mzs, intensities = np.array(rows['mzs']),np.array(rows['intensities'])
                        #mzs, intensities = np.expand_dims(mzs,0), np.expand_dims(intensities,0)

                        mzs = np.array(mzs)
                        intensities = np.array(intensities)
                        #ions = np.array(ions)
                        #print(ions.shape)
                        #ions = np.transpose(ions,(1,0,2))

                        # print(mzs.shape)
                        # print(intensities.shape)
                        for _ in range(1):
                            if VERBOSE:
                                print('scoring...')
                            best_score_index, best_score, pos_score = scoring(mzs, intensities, ions)        
                        #print(pos_score.shape)
                        #replace zeros with -1
                        #pos_score = np.where(pos_score==0.0, -100.0, pos_score)
                        sorted_index = np.argsort(pos_score,axis=-1)[:,::-1]
                        sorted_index = sorted_index[:,:TOPK]
                        #sorted_index = np.reshape(best_score_index,(len(best_score_index),1))
                        
                        #print(sorted_index)
                        top_peptide_ = batched_topk_peptides
                        top_peptide_is_decoy_ = rows['is_decoy'].to_numpy()
                        top_peptide_distance_ = rows['topk_distances'].to_numpy()
                        best_score_ = np.array(pos_score)
                        
                        top_peptide = [top_peptide_[i][sorted_index[i,:]] for i in range(top_peptide_.shape[0])]
                        top_peptide_is_decoy = [top_peptide_is_decoy_[i][sorted_index[i,:]] for i in range(top_peptide_is_decoy_.shape[0])]
                        top_peptide_distance = [top_peptide_distance_[i][sorted_index[i,:]] for i in range(top_peptide_distance_.shape[0])]
                        best_score = [best_score_[i][sorted_index[i,:]] for i in range(best_score_.shape[0])]

                        #top_peptide = [batched_topk_peptides[id][b] for id,b in enumerate(best_score_index)]
                        #top_peptide_is_decoy = [rows['is_decoy'].to_numpy()[id][b] for id,b in enumerate(best_score_index)]
                        #top_peptide_distance = [rows['topk_distances'].to_numpy()[id][b] for id,b in enumerate(best_score_index)]
                        if VERBOSE:
                            print(sum(top_peptide==true_peptide))
                        #print(sum(top_peptide==true_peptide),top_peptide,true_peptide,best_score)
                        # if i > 10:
                        #     quit()

                        top_peptides.extend(top_peptide)
                        top_peptide_is_decoys.extend(top_peptide_is_decoy)
                        top_peptide_distances.extend(top_peptide_distance)
                        best_scores.extend(best_score)
                        all_scores.extend(np.reshape(pos_score,-1))

                #search_results = search_results[:SUBSET]
                
                search_results= search_results.iloc[np.repeat(np.arange(len(search_results)), TOPK)]


        search_results['best_is_decoy']=top_peptide_is_decoys
        search_results['best_distance']=top_peptide_distances
        search_results['best_score']=best_scores
        search_results['best_peptide']=top_peptides
        all_peptides = top_peptides
        search_results['peptide_mass']= list(map(lambda x: theoretical_peptide_mass(*x),zip(all_peptides,np.zeros_like(all_peptides))))
        search_results['delta_mass']=search_results['pepmass'] - search_results['peptide_mass']
        print(len(search_results))
        search_results=search_results.drop(columns=['mzs', 'intensities','is_decoy'])
        print(sum(search_results['best_peptide']==search_results['peptide'])/len(search_results))

        print(search_results)
        print(search_results.columns)

        print(os.path.join(OUTPUT_DIR,'%s.search_results_scored.pkl'%raw_file))
        search_results.to_pickle(os.path.join(OUTPUT_DIR,'%s.search_results_scored.pkl'%raw_file))
        
        #search_results.to_csv('search_results_scored.csv',index=False)
        #search_results.to_hdf(os.path.join(OUTPUT_DIR,'search_results_scored.h5'),key='search_results_scored', mode='w')

        #search_results_scored = pd.concat([search_results_scored,search_results],ignore_index=True)
        #with pd.HDFStore(os.path.join(OUTPUT_DIR,'search_results_scored.h5')) as store_out:
        #store_out.put(key,search_results)
        #search_results_scored.to_hdf(os.path.join(OUTPUT_DIR,'search_results_scored.h5'),key='search_results_scored', mode='w')