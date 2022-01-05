import sys,os 
os.environ['YHYDRA_CONFIG'] = sys.argv[1]
from load_config import CONFIG

RESULTS_DIR = CONFIG['RESULTS_DIR']

#%%
import pandas as pd
import os


with pd.HDFStore(os.path.join('%s/forward/'%(RESULTS_DIR),'search_results_scored_filtered.h5')) as store:
    raw_files = store.keys()
    search_results = pd.concat([store[key] for key in raw_files],ignore_index=True)

print(search_results)