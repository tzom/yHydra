import sys,os 
os.environ['YHYDRA_CONFIG'] = sys.argv[1]
from load_config import CONFIG

RESULTS_DIR = CONFIG['RESULTS_DIR']

#%%
import pandas as pd
import os


# with pd.HDFStore(os.path.join('%s/'%(RESULTS_DIR),'search_results_scored_filtered.h5')) as store:
#     raw_files = store.keys()
#     search_results = pd.concat([store[key] for key in raw_files],ignore_index=True)

search_results = pd.read_pickle(os.path.join('%s/'%(RESULTS_DIR),'search_results_scored_filtered.pkl'))

selected_columns = ['raw_file','scan','index','precursorMZ','pepmass','charge','best_is_decoy','best_distance','best_score','best_peptide','peptide_mass','delta_mass']

print(search_results[selected_columns])
search_results[selected_columns].to_csv(os.path.join('%s/'%(RESULTS_DIR),'search_results_scored_filtered.csv'))
N_peptides=len(set(search_results.best_peptide.unique()))
N_PSMs=len(search_results)

print('Identified PSMs (yHydra): %s'%(N_PSMs))
print('Identified peptides (yHydra): %s'%(N_peptides))