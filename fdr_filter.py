import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
#from score import calc_ions, scoring
from tqdm import tqdm
from pyteomics import auxiliary as aux
import os
from load_config import CONFIG

#OUTPUT_DIR = args.OUTPUT_DIR
#REV_OUTPUT_DIR = args.REV_OUTPUT_DIR

FDR = CONFIG['FDR']
MIN_DELTA_MASS = CONFIG['MIN_DELTA_MASS']
MAX_DELTA_MASS = CONFIG['MAX_DELTA_MASS']
SAVE_DB_AS_JSON = True

#search_results = pd.read_hdf(os.path.join(OUTPUT_DIR,'search_results_scored.h5'),'search_results_scored')
# rev_search_results = pd.read_hdf(os.path.join(REV_OUTPUT_DIR,'search_results_scored.h5'),'search_results_scored')
def fdr_filter():

    OUTPUT_DIR = CONFIG['RESULTS_DIR']

    with pd.HDFStore(os.path.join(OUTPUT_DIR,'search_results_scored.h5')) as store:
        raw_files = store.keys()
        search_results = pd.concat([store[key] for key in raw_files])

    # search_results['is_decoy'] = False
    # rev_search_results['is_decoy'] = True

    df = search_results#pd.concat([search_results,rev_search_results])

    df = df[df.delta_mass<MAX_DELTA_MASS]
    df = df[df.delta_mass>MIN_DELTA_MASS]

    df.best_score = -np.log(df.best_score+1.)
    #index = df.groupby('id')['best_score'].nlargest(1).reset_index(drop=True).index
    #df = df.iloc[index]

    df_filtered = aux.filter(df, key='best_score', is_decoy='best_is_decoy', fdr=FDR)
    df_filtered = df_filtered[~df_filtered.best_is_decoy]

    if SAVE_DB_AS_JSON:
        import json
        with open(os.path.join(OUTPUT_DIR+'/forward/db','db.json')) as fp:
            ncbi_peptide_protein = json.load(fp)
        df_filtered['accession'] = list(map(lambda x: ncbi_peptide_protein[x],df_filtered.best_peptide))

    print(sum(df_filtered['best_peptide']==df_filtered['peptide'])/len(df_filtered))
    ground_truth_ident_peptides=set(df.peptide.unique())
    yhydra_ident_peptides=set(df_filtered.best_peptide.unique())
    print('Identified peptides (true):',len(ground_truth_ident_peptides))
    print('Identified peptides (yHydra):',len(yhydra_ident_peptides))

    df_filtered.to_hdf(os.path.join(OUTPUT_DIR,'search_results_scored_filtered.h5'),key='search_results_scored_filtered', mode='w')
