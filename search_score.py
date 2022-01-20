import multiprocessing, sys, os
import tensorflow as tf
from load_config import CONFIG
OUTPUT_DIR = CONFIG['RESULTS_DIR']

import pandas as pd
import numpy as np
from score_utils import calc_ions, scoring
from tqdm import tqdm
from utils import batched_list,unbatched_list
from proteomics_utils import normalize_intensities,theoretical_peptide_mass,trim_peaks_list

MAX_N_PEAKS = CONFIG['MAX_N_PEAKS']#500
BATCH_SIZE=CONFIG['BATCH_SIZE']#64
NUMBER_OF_THREADS=CONFIG['NUMBER_OF_THREADS']#64
K=CONFIG['K']#50
TOPK=1
VERBOSE = False
SUBSET=None

def trim_peaks_list_(x): 
    mzs, intensities = x
    mzs, intensities = mzs, normalize_intensities(intensities)
    return trim_peaks_list(mzs, intensities,MAX_N_PEAKS=MAX_N_PEAKS)

#if __name__ == '__main__':
def search_score(OUTPUT_DIR=OUTPUT_DIR):
    
    with pd.HDFStore(os.path.join(OUTPUT_DIR,'search_results.h5')) as store, pd.HDFStore(os.path.join(OUTPUT_DIR,'search_results_scored.h5')) as store_out:
        raw_files = store.keys()
        search_results_scored = pd.DataFrame()

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


                search_results['best_is_decoy']=list(unbatched_list(top_peptide_is_decoys))
                search_results['best_distance']=list(unbatched_list(top_peptide_distances))
                search_results['best_score']=list(unbatched_list(best_scores))
                search_results['best_peptide']=list(unbatched_list(top_peptides))
                all_peptides = list(unbatched_list(top_peptides))
                search_results['peptide_mass']= list(map(lambda x: theoretical_peptide_mass(*x),zip(all_peptides,np.zeros_like(all_peptides))))
                search_results['delta_mass']=search_results['pepmass'] - search_results['peptide_mass']
                print(len(search_results))
                #search_results=search_results.drop(columns=['mzs', 'intensities'])
                print(sum(search_results['best_peptide']==search_results['peptide'])/len(search_results))

                print(search_results)
                print(search_results.columns)

                #search_results.to_csv('search_results_scored.csv',index=False)
                #search_results.to_hdf(os.path.join(OUTPUT_DIR,'search_results_scored.h5'),key='search_results_scored', mode='w')

                #search_results_scored = pd.concat([search_results_scored,search_results],ignore_index=True)
                #with pd.HDFStore(os.path.join(OUTPUT_DIR,'search_results_scored.h5')) as store_out:
                store_out.put(key,search_results)
    #search_results_scored.to_hdf(os.path.join(OUTPUT_DIR,'search_results_scored.h5'),key='search_results_scored', mode='w')