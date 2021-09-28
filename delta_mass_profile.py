import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

search_results = pd.read_hdf('./tmp/forward/search_results_scored_filtered.h5','search_results_scored_filtered')
# search_results = search_results[:10]
# search_results = search_results.drop(columns=['mzs','intensities'])
# search_results = search_results.explode(['topk_peptides','topk_distances'])
# search_results.to_csv('./tmp/tmp.csv',sep='\t',index=False)
# exit()
#search_results = search_results[search_results.peptide==search_results.best_peptide]

print(search_results)


delta_mass = search_results['delta_mass'].to_numpy()

delta_mass = delta_mass[np.abs(delta_mass)-200.<0.]

smallest = np.min(delta_mass)
largest = np.max(delta_mass)

print(smallest,largest)

linewidth = 0.3

hist, bin_edges = np.histogram(delta_mass,bins=10000)#np.linspace(start=min(delta_mass),stop=max(delta_mass),num=1000000),)
all_diffs = (bin_edges[:-1]+bin_edges[1:])/2
all_values = hist

topk= np.argsort(all_values)[::-1][1:6]

print(all_diffs[topk],all_values[topk])

plt.figure(figsize=(9,5))

ma,st,ba=plt.stem(all_diffs,all_values,basefmt=' ',markerfmt=' ',linefmt='green',use_line_collection=True,label='delta masses')
plt.setp(st, 'linewidth', linewidth)
#plt.yscale('log')
plt.xlim(-150,300)
plt.ylim(0,5000)
plt.xlabel('mass difference')

for x,y in list(zip(all_diffs[topk],all_values[topk])):
    plt.gca().text(x-2.5, y+20, '%s Da'%np.round(x,2),rotation=70)
plt.tightlayout()
plt.savefig('./figures/delta_mass_profile',dpi=600)