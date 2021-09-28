import sys

sys.path.append("../dnovo3")
sys.path.append("../Neonomicon")
import glob, os
import tensorflow as tf

if True:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    device = '/CPU:0'
    use_gpu=False
else:
    device = '/GPU:0'
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0],True)
    use_gpu=True



import random
from tf_data_json import USIs,parse_json_npy
from usi_magic import parse_usi
#from tf_data_mgf import MGF
from load_model import spectrum_embedder,sequence_embedder
from proteomics_utils import theoretical_peptide_mass,precursor2peptide_mass

from tqdm import tqdm
import numpy as np
import multiprocessing

AUTOTUNE=tf.data.AUTOTUNE

print('fire up datasets...')
N = 5000
MSMS_OUTPUT_IN_RESULTS = False


if True:
    print('globbing files...')
    files = glob.glob(os.path.join('../../scratch/USI_files/PXD007963/**/','*.json'))
    #files = glob.glob(os.path.join('../../scratch/USI_files/delete_me/PXD003916/Michelle-Experimental-Sample6.mzid_Michelle-Experimental-Sample6.MGF','*.json'))
    #files = glob.glob(os.path.join('../../Neonomicon/files/test/**/','*.json'))
    #files = glob.glob(os.path.join('../../Neonomicon/dump','*.json'))
    random.seed(0)
    random.shuffle(files)
    files = files[:N]

    ds = USIs(files,batch_size=1,buffer_size=1).get_dataset().unbatch()
    ds_spectra = ds.map(lambda x,y: x).batch(64)
    ds = USIs(files,batch_size=1,buffer_size=1).get_dataset().unbatch()
    ds_peptides = ds.map(lambda x,y: y).batch(256)

true_peptides = []
true_precursorMZs = []
true_pepmasses = []
true_charges = []
true_modified = []
true_ID = []
true_mzs = []
true_intensities = []

input_specs_npy = {
    "mzs": np.float32,
    "intensities": np.float32,
    "usi": str,
    #"charge": float,
    "precursorMZ": float,
}

def parse_json_npy_(file_location): return parse_json_npy(file_location,specs=input_specs_npy)

if __name__ == '__main__':

    if True:
        with multiprocessing.Pool(64) as p:
            print('getting true peptides...')
            for psm in tqdm(list(p.imap(parse_json_npy_, files,1))):
            #for psm in tqdm(list(map(lambda file_location: parse_json_npy(file_location,specs=input_specs_npy), files))):
                if MSMS_OUTPUT_IN_RESULTS:
                    true_mzs.append(psm['mzs'])
                    true_intensities.append(psm['intensities'])
                #charge=psm['charge']
                precursorMZ=float(psm['precursorMZ'])
                usi=str(psm['usi'])
                collection_identifier, run_identifier, index, charge, peptideSequence, positions = parse_usi(usi)
                if positions is not None and len(positions)>0:
                    if type(positions[0])==tuple:
                        true_modified.append(positions[0][1])
                    else:
                        true_modified.append(positions[1]) 
                else:
                    true_modified.append('unmod.')
                true_peptides.append(peptideSequence)
                true_precursorMZs.append(precursorMZ)
                true_pepmasses.append(precursor2peptide_mass(precursorMZ,int(charge)))
                true_charges.append(int(charge))
                true_ID.append(usi)

    with tf.device(device):
        print('embedding spectra...')
        for _ in tqdm(range(1)):        
            embedded_spectra = spectrum_embedder.predict(ds_spectra)

        print('embedding peptides...')
        for _ in tqdm(range(1)):        
            embedded_peptides = sequence_embedder.predict(ds_peptides)


    def append_dim(X,new_dim,axis=1):
        return np.concatenate((X, np.expand_dims(new_dim,axis=axis)), axis=axis)

    embedded_spectra = embedded_spectra

    print(embedded_spectra.shape)

    import umap.umap_ as umap
    import umap.plot
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame({
        'precursorMZ':true_precursorMZs,
        'charge':true_charges,
        'modified':true_modified,
        'embedded_spectra':embedded_spectra.tolist(),
        })#, index=[0])

    print('run umap...')


    df_concat = pd.concat([df,df])
    embedded_spectra_peptides = np.concatenate([embedded_peptides,embedded_spectra], axis=0)
    
    N = len(df_concat)
    print('N elements to embed, shuffling...' )
    shuffled_indices = np.arange(N)
    np.random.shuffle(shuffled_indices)
    embedded_spectra_peptides = embedded_spectra_peptides[shuffled_indices,:]
    df_concat = df_concat.iloc[shuffled_indices]
    print('umap fitting...' )
    mapper = umap.UMAP().fit(embedded_spectra_peptides)

    plt.figure()
    ax = umap.plot.points(mapper, labels=df_concat.charge)
    ax.get_legend().set_title("charge")
    plt.savefig('figures/umap_charge.png',dpi=600)

    plt.figure()
    ax = umap.plot.points(mapper, labels=df_concat.precursorMZ)
    ax.get_legend().remove()
    plt.savefig('figures/umap_precursorMZ.png',dpi=600)
    
    plt.figure()
    ax = umap.plot.points(mapper, labels=df_concat.modified)
    ax.get_legend().set_title("modified")
    plt.savefig('figures/umap_modified.png',dpi=600)


    plt.figure()
    N_spectra = embedded_spectra.shape[0]
    print('%s elements to embed, shuffling...'%N )
    shuffled_indices = np.arange(N_spectra)
    np.random.shuffle(shuffled_indices)
    embedded_spectra = embedded_spectra[shuffled_indices,:]
    df_shuffled = df.iloc[shuffled_indices]
    mapper = umap.UMAP().fit(embedded_spectra)
    mapped_spectra = mapper.transform(embedded_spectra)
    mapped_peptides = mapper.transform(embedded_peptides)

    plt.scatter(mapped_peptides[:, 0], mapped_peptides[:, 1], s = 5, c=df_shuffled.charge, cmap='jet',label='peptides')
    plt.scatter(mapped_spectra[:, 0], mapped_spectra[:, 1], s = 5, c=df_shuffled.charge, cmap='Spectral',label='spectra')
    plt.legend()
    ax.get_legend().set_title("charge")
    plt.savefig('figures/umap_spectra_peptides_charge.png',dpi=600)