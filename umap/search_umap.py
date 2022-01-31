import sys,os 
sys.path.append('.')

os.environ['YHYDRA_CONFIG'] = sys.argv[1]
import setup_device 

import pandas as pd
pd.set_option('display.max_rows', 100)
import numpy as np
#from score import calc_ions, scoring
from tqdm import tqdm
from pyteomics import auxiliary as aux
import os
from load_config import CONFIG

from proteomics_utils import normalize_intensities,trim_peaks_list_v2,MAX_N_PEAKS,NORMALIZATION_METHOD
from load_model import spectrum_embedder,sequence_embedder
from proteomics_utils import theoretical_peptide_mass,precursor2peptide_mass

FDR = CONFIG['FDR']
MIN_DELTA_MASS = CONFIG['MIN_DELTA_MASS']
MAX_DELTA_MASS = CONFIG['MAX_DELTA_MASS']
BATCH_SIZE = CONFIG['BATCH_SIZE']
SAVE_DB_AS_JSON = True

OUTPUT_DIR = CONFIG['RESULTS_DIR']

FRAC = 1.0
FIT = True
DPI = 300

with pd.HDFStore(os.path.join(OUTPUT_DIR,'search_results_scored.h5')) as store:
    raw_files = store.keys()
    search_results = pd.concat([store[key] for key in raw_files])

# search_results['is_decoy'] = False
# rev_search_results['is_decoy'] = True

df = search_results#pd.concat([search_results,rev_search_results])

df = df[df.delta_mass<MAX_DELTA_MASS]
df = df[df.delta_mass>MIN_DELTA_MASS]

df.best_score = -np.log(df.best_score+1.)
df = aux.filter(df, key='best_score', is_decoy='best_is_decoy', fdr=1.0)

print(df.columns)
print(df)

df = df.sample(frac=FRAC)

preprocessed_spectra = []
for i,psm in tqdm(df.iterrows()):
    #for i,spectrum in enumerate(tqdm(parse_hdf_npy(MGF,calibrate_fragments=True,database_filename='/hpi/fs00/home/tom.altenburg/projects/test_alphapept/bruker_example/test_database.hdf'))):        
        mzs = psm['mzs']
        intensities = psm['intensities']

        mzs = np.array(mzs)
        intensities = np.array(intensities)
        #mzs, intensities = mzs,normalize_intensities(intensities,method=NORMALIZATION_METHOD)
        #mzs, intensities = trim_peaks_list_v2(mzs, intensities, MAX_N_PEAKS=MAX_N_PEAKS, PAD_N_PEAKS=500)
        preprocessed_spectrum = np.stack((mzs, intensities),axis=-1)
        preprocessed_spectra.append(preprocessed_spectrum)        

print('embedding spectra...')
for _ in tqdm(range(1)): 
    ds_spectra = np.array(preprocessed_spectra)
    embedded_spectra = spectrum_embedder.predict(ds_spectra,batch_size=BATCH_SIZE)
    
print('embedding peptides...')
import multiprocessing
from embed_db import p_b_map, trim_sequence, get_sequence_of_indices, BATCH_SIZE_PEPTIDES
from pyteomics import electrochem, achrom

def pI(peptide): return electrochem.pI(peptide, precision_pI=0.001)
def RT(peptide): return achrom.calculate_RT(peptide, achrom.RCs_guo_ph7_0)
def feature(peptide):
    if 'K' in peptide or 'R' in peptide:
        return 1.0
    else:
        return 0.0



for _ in tqdm(range(1)): 
    with multiprocessing.pool.ThreadPool() as p:
        peptides = df.best_peptide.to_list()
        pI_peptides = p_b_map(pI,p,peptides,batch_size=BATCH_SIZE_PEPTIDES)
        RT_peptides = p_b_map(RT,p,peptides,batch_size=BATCH_SIZE_PEPTIDES)
        feature_peptides = p_b_map(feature,p,peptides,batch_size=BATCH_SIZE_PEPTIDES)
        peptides = p_b_map(trim_sequence,p,peptides,batch_size=BATCH_SIZE_PEPTIDES)
        peptides = p_b_map(get_sequence_of_indices,p,peptides,batch_size=BATCH_SIZE_PEPTIDES)        
        embedded_peptides = sequence_embedder.predict(peptides,batch_size=BATCH_SIZE_PEPTIDES)

def append_dim(X,new_dim,axis=1):
    return np.concatenate((X, np.expand_dims(new_dim,axis=axis)), axis=axis)

embedded_spectra = embedded_spectra

print(embedded_spectra.shape)
print(embedded_peptides.shape)

#import umap.umap_ as umap
import umap
import umap.plot
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

_,peptide_index = np.unique(df.best_peptide, return_inverse=True)

df['peptide_index'] = peptide_index
df['embedded_spectra'] = embedded_spectra.tolist()
df['pI'] = pI_peptides
df['RT'] = RT_peptides
df['feature'] = feature_peptides
# df = pd.DataFrame({
#     'pepmass':df.pepmass,
#     'charge':df.charge,
#     'q':np.log(df.q),
#     'peptide':peptide_index,
#     'best_peptide':df.best_peptide,
#     'embedded_spectra':embedded_spectra.tolist(),
#     'run':df.raw_file,
#     })#, index=[0])

print('run umap...')

df_concat = pd.concat([df])
embedded_spectra = np.concatenate([embedded_spectra], axis=0)

np.random.seed(42)
import pickle
def umap_shuffled(df: pd.DataFrame, features: np.ndarray):        
    N = len(df)
    print('N elements to embed, shuffling...' )
    shuffled_indices = np.arange(N)
    np.random.shuffle(shuffled_indices)
    features = features[shuffled_indices,:]
    df = df.iloc[shuffled_indices]
    print('umap fitting...' )
    
    if FIT:
        mapper = umap.UMAP(n_neighbors=500,min_dist=0.5).fit(features)
        with open("umap/umap.pickle", "wb") as f:
            pickle.dump(mapper,f)
    else:
        with open("umap/umap.pickle", "rb") as f:
            mapper = pickle.load(f)
    print('umap fitting done.' )       
    return df, features, mapper, shuffled_indices

df,embedded_spectra,mapper,shuffled_indices = umap_shuffled(df, embedded_spectra)
embedded_peptides = embedded_peptides[shuffled_indices,:]

if FIT:
    mapped_all = mapper.transform(embedded_spectra)
    with open("umap/mapped_all.pickle", "wb") as f:
        pickle.dump(mapped_all,f)
    mapped_all_peptides = mapper.transform(embedded_peptides)
    with open("umap/mapped_all_peptides.pickle", "wb") as f:
        pickle.dump(mapped_all_peptides,f)
else:
    with open("umap/mapped_all.pickle", "rb") as f:
        mapped_all = pickle.load(f)
    with open("umap/mapped_all_peptides.pickle", "rb") as f:
        mapped_all_peptides = pickle.load(f)


def get_colorbar(x,ax,cmap='viridis',discrete = False, orientation='horizontal'):
    #cmap = mpl.cm[cmap]
    cmap = plt.get_cmap(cmap)
    
    if discrete:
        boundaries = np.arange(min(x)-0.5,max(x)+1.5,1.0)
        norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)
        cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                        norm=norm,
                                        boundaries=boundaries,
                                        ticks=boundaries+0.5,
                                        orientation=orientation)
    else:
        norm = mpl.colors.Normalize(vmin=min(x), vmax=max(x))
        cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                        norm=norm,
                                        orientation=orientation)

    # if discrete:
    #     labels = x
    #     loc    = x + 0.5
    #     cb.set_ticks(loc)
    #     cb.set_ticklabels(labels)
    return cmap, norm, cb

args = {'s':0.5}

counts_dict = df['best_peptide'].value_counts().to_dict()
df['freq'] = df['best_peptide'].map(counts_dict)
most_frequent_peptide_index = df['freq'].argmax()
most_frequent_peptide = str(df.best_peptide.values[most_frequent_peptide_index])
print(df[:100][['freq','best_peptide','delta_mass','charge']],most_frequent_peptide)

selected_indices = df.best_peptide == most_frequent_peptide
#mapped_selected = mapped_all[selected_indices,:]#mapper.transform(embedded_spectra[selected_indices,:])
#df_keep = df[selected_indices]

plt.figure(figsize=(4,4))
plt.scatter(mapped_all[:, 0], mapped_all[:, 1], s = args['s'], c='grey', alpha=0.5)
plt.scatter(mapped_all[selected_indices, 0], mapped_all[selected_indices, 1], s = args['s'], c=df[selected_indices].charge, alpha=0.9)
plt.axis('off')
plt.savefig('figures/umap_common_peptide.png',dpi=DPI)

plt.figure(figsize=(4,4))
plt.scatter(mapped_all[:, 0], mapped_all[:, 1], s = args['s'], c='grey', alpha=0.5)
plt.scatter(mapped_all_peptides[:, 0], mapped_all_peptides[:, 1], s = args['s'], c='green', alpha=0.5)
plt.axis('off')
plt.savefig('figures/umap_peptide_spectra.png',dpi=DPI)

plt.figure(figsize=(4,4))
plt.scatter(mapped_all_peptides[:, 0], mapped_all_peptides[:, 1], s = args['s'], c=df.pI, cmap='plasma', alpha=0.5)
plt.axis('off')
plt.savefig('figures/umap_peptide_pI.png',dpi=DPI)

plt.figure(figsize=(4,0.2))
cmap, norm, cb = get_colorbar(df.pI.values,plt.gca(),'plasma') 
cb.set_label('pI')
plt.savefig('figures/umap_peptide_pI_colorbar.png', bbox_inches="tight", dpi=DPI)

plt.figure(figsize=(4,4))
plt.scatter(mapped_all_peptides[:, 0], mapped_all_peptides[:, 1], s = args['s'], c=df.RT, cmap='plasma', alpha=0.5)
plt.axis('off')
plt.savefig('figures/umap_peptide_RT.png',dpi=DPI)

plt.figure(figsize=(4,0.2))
cmap, norm, cb = get_colorbar(df.RT.values,plt.gca(),'plasma') 
cb.set_label('RT')
plt.savefig('figures/umap_peptide_RT_colorbar.png', bbox_inches="tight", dpi=DPI)

plt.figure(figsize=(4,4))
plt.scatter(mapped_all_peptides[:, 0], mapped_all_peptides[:, 1], s = args['s'], c=df.feature, cmap='cool', alpha=0.5)
plt.axis('off')
plt.savefig('figures/umap_peptide_feature.png',dpi=DPI)

plt.figure(figsize=(4,0.2))
cmap, norm, cb = get_colorbar(df.feature.values,plt.gca(),'cool',discrete=True) 
cb.set_label('missing K/R')
plt.savefig('figures/umap_peptide_feature_colorbar.png', bbox_inches="tight", dpi=DPI)

import copy 
tmp = copy.copy(df.sort_values(by='q',ascending=True))
tmp = tmp.drop_duplicates(subset=['best_peptide','charge'],keep='first')

tmp = tmp[tmp.q < FDR]
tmp = tmp[~tmp.best_is_decoy]
filtered_indices = tmp.index
mask = df.index.isin(filtered_indices)
identified = df[mask]

#identified, mapped_identified, m = umap_shuffled(identified,embedded_spectra[mask,:])
#mapped_identified
#unidentified = df[~mask]
#mapped_unidientified = m.transform(embedded_spectra[~mask,:])


plt.figure(figsize=(4,4))
plt.scatter(mapped_all[mask, 0], mapped_all[mask, 1], s = args['s'], c='red', alpha=0.3)
plt.scatter(mapped_all[~mask, 0], mapped_all[~mask, 1], s = args['s'], c='blue', alpha=0.3)    
plt.axis('off')
plt.savefig('figures/umap_q.png',dpi=DPI)

plt.figure(figsize=(4,4))
c = df.charge.values
plt.scatter(mapped_all[:, 0], mapped_all[:, 1], s = args['s'], c=c, alpha=0.3)
plt.axis('off')
plt.savefig('figures/umap_charge.png',dpi=DPI)

plt.figure(figsize=(4,0.2))
cmap, norm, cb = get_colorbar(c,plt.gca(),discrete=True) 
cb.set_label('Postive Charge')
plt.savefig('figures/umap_charge_colorbar.png', bbox_inches="tight", dpi=DPI)


plt.figure(figsize=(4,4))
c = df['pepmass'].values
plt.scatter(mapped_all[:, 0], mapped_all[:, 1], s = args['s'], c=c, cmap='jet', alpha=0.3)
plt.axis('off')
plt.savefig('figures/umap_pepmass.png',dpi=DPI)

plt.figure(figsize=(4,0.2))
cmap, norm, cb = get_colorbar(c,plt.gca(),cmap='jet',discrete=False) 
cb.set_label('Precursor mass')
plt.savefig('figures/umap_pepmass_colorbar.png', bbox_inches="tight", dpi=DPI)

# plt.figure()
# ax = umap.plot.points(mapper, labels=df_concat.pepmass)
# plt.title("pepmass")
# ax.get_legend().remove()
# plt.savefig('figures/umap_pepmass.png',dpi=DPI)

# plt.figure()
# ax = umap.plot.points(mapper, labels=df_concat.peptide)
# plt.title("peptide")
# ax.get_legend().remove()
# plt.savefig('figures/umap_peptide.png',dpi=DPI)

# plt.figure()
# ax = umap.plot.points(mapper, labels=df_concat.raw_file)
# plt.title("raw_file")
# ax.get_legend().remove()
# plt.savefig('figures/umap_raw_file.png',dpi=DPI)




# plt.figure()
# ax = umap.plot.points(mapper, labels=df_concat.precursorMZ)
# ax.get_legend().remove()
# plt.savefig('figures/umap_precursorMZ.png',dpi=600)

# plt.figure()
# ax = umap.plot.points(mapper, labels=df_concat.modified)
# ax.get_legend().set_title("modified")
# plt.savefig('figures/umap_modified.png',dpi=600)


# plt.figure()
# N_spectra = embedded_spectra.shape[0]
# print('%s elements to embed, shuffling...'%N )
# shuffled_indices = np.arange(N_spectra)
# np.random.shuffle(shuffled_indices)
# embedded_spectra = embedded_spectra[shuffled_indices,:]
# df_shuffled = df.iloc[shuffled_indices]
# mapper = umap.UMAP().fit(embedded_spectra)
# mapped_spectra = mapper.transform(embedded_spectra)
# mapped_peptides = mapper.transform(embedded_peptides)

# plt.scatter(mapped_peptides[:, 0], mapped_peptides[:, 1], s = 5, c=df_shuffled.charge, cmap='jet',label='peptides')
# plt.scatter(mapped_spectra[:, 0], mapped_spectra[:, 1], s = 5, c=df_shuffled.charge, cmap='Spectral',label='spectra')
# plt.legend()
# ax.get_legend().set_title("charge")
# plt.savefig('figures/umap_spectra_peptides_charge.png',dpi=600)