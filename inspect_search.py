import pandas as pd
import os


with pd.HDFStore(os.path.join('example/search/forward/','search_results_scored_filtered.h5')) as store:
    raw_files = store.keys()
    search_results = pd.concat([store[key] for key in raw_files],ignore_index=True)

print(search_results)