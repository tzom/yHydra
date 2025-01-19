import pandas as pd
import numpy as np
#from score import calc_ions, scoring
from tqdm import tqdm
from pyteomics import auxiliary as aux
import os, glob
from load_config import CONFIG

#OUTPUT_DIR = args.OUTPUT_DIR
#REV_OUTPUT_DIR = args.REV_OUTPUT_DIR

FDR = CONFIG['FDR']
MIN_DELTA_MASS = CONFIG['MIN_DELTA_MASS']
MAX_DELTA_MASS = CONFIG['MAX_DELTA_MASS']
SAVE_DB_AS_JSON = False
PLOT_SCOREDIST = False

#search_results = pd.read_hdf(os.path.join(OUTPUT_DIR,'search_results_scored.h5'),'search_results_scored')
# rev_search_results = pd.read_hdf(os.path.join(REV_OUTPUT_DIR,'search_results_scored.h5'),'search_results_scored')
def fdr_filter():

    OUTPUT_DIR = CONFIG['RESULTS_DIR']

    # with pd.HDFStore(os.path.join(OUTPUT_DIR,'search_results_scored.h5')) as store:
    #     raw_files = store.keys()
    #     search_results = pd.concat([store[key] for key in raw_files])
    
    runs = glob.glob(os.path.join(OUTPUT_DIR,'*.search_results_scored.pkl'))
    search_results = pd.concat([pd.read_pickle(run) for run in runs])

    # search_results['is_decoy'] = False
    # rev_search_results['is_decoy'] = True

    df = search_results#pd.concat([search_results,rev_search_results])

    df = df[df.delta_mass<MAX_DELTA_MASS]
    df = df[df.delta_mass>MIN_DELTA_MASS]

    #df.best_score = np.log(df.best_score+1.)
    #index = df.groupby('id')['best_score'].nlargest(1).reset_index(drop=True).index
    #df = df.iloc[index]

    df.best_score = -df.best_score
    df_filtered = aux.filter(df, key='best_score', is_decoy='best_is_decoy', fdr=FDR)
    df_filtered = df_filtered[~df_filtered.best_is_decoy]

    cutoff = -df_filtered.best_score.max()

    # df_filtered = df[df.best_score>5.0]
    # df_filtered = df_filtered[~df_filtered.best_is_decoy]

    #yhydra_ident_peptides=set(df_filtered.best_peptide.unique())
    
    print('Identified PSMs (yHydra):',len(df_filtered))
    df = df.sort_values(by=['best_score'])[::-1]
    df_filtered = df_filtered.drop_duplicates(subset=['best_peptide'],keep='first')
    print('Identified peptides (yHydra):',len(df_filtered))
    print(df_filtered)

    if PLOT_SCOREDIST:
        import matplotlib.pyplot as plt
        plt.title("N identified peptides: %s"%len(df_filtered))
        x = df
        x.best_score = -x.best_score
        x = x[x.best_score > 0.0]
        #bins = np.linspace(min(search_results.best_score),100.,100)
        bins = np.linspace(0.,100.,100)
        plt.hist(x.best_score,label='targets+decoys',bins=bins,alpha=0.1)         
        plt.hist(x[~x.best_is_decoy].best_score,label='targets',bins=bins,alpha=0.5)
        plt.hist(x[x.best_is_decoy].best_score,label='decoys',color='black',bins=bins,alpha=1.0)   
        #plt.axvline(cutoff,color='r',alpha=0.3,label='cutoff (1% FDR)')
        plt.xlabel('N PSMs')
        plt.xlabel('log yHydra score')
        plt.legend()
        plt.savefig('./figures/scoredist.png')


    if SAVE_DB_AS_JSON:
        import json
        with open(os.path.join(OUTPUT_DIR+'/forward/db','db.json')) as fp:
            ncbi_peptide_protein = json.load(fp)
        df_filtered['accession'] = list(map(lambda x: ncbi_peptide_protein[x],df_filtered.best_peptide))



    #df_filtered.to_hdf(os.path.join(OUTPUT_DIR,'search_results_scored_filtered.h5'),key='search_results_scored_filtered', mode='w')
    df_filtered.to_pickle(os.path.join(OUTPUT_DIR,'search_results_scored_filtered.pkl'))
