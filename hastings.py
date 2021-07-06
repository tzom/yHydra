import sys

from numpy.core.fromnumeric import squeeze 
sys.path.append("..")
import glob, os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
import copy
from tf_data_json import USIs,parse_json_npy,parse_usi, parse_peptide
from preprocessing import get_sequence_from_indices, get_sequence_of_indices
from check_embedding import spectrum_embedder,sequence_embedder
from fasta2db import theoretical_peptide_mass

from tqdm import tqdm
import numpy as np
import tensorflow as tf

AUTOTUNE=tf.data.AUTOTUNE

DB_DIR = './db'

db_embedded_peptides = np.load(os.path.join(DB_DIR,"embedded_peptides.npy"))
db_peptides = np.load(os.path.join(DB_DIR,"peptides.npy"))
db_pepmasses = np.load(os.path.join(DB_DIR,"pepmasses.npy"))

sorted_indices = np.argsort(db_pepmasses)

db_embedded_peptides=db_embedded_peptides[sorted_indices]
db_peptides=db_peptides[sorted_indices]
db_pepmasses=db_pepmasses[sorted_indices]

N = 1

files = glob.glob(os.path.join('../../Neonomicon/files/test/**/','*.json'))
# #files = glob.glob(os.path.join('../../Neonomicon/dump','*.json'))
# random.seed(0)
# random.shuffle(files)
files = files[-N:]


#print(files[0])
#files = ['../../Neonomicon/files/test/S_venezuelae_GYM_2_21Mar16_Arwen_16-01-03_msgfplus/000454f47977d07980a077b850004743.json']

ds = USIs(files,batch_size=1,buffer_size=1).get_dataset().unbatch()
ds_spectra = ds.map(lambda x,y: x).batch(1)
ds_peptides = ds.map(lambda x,y: y).batch(1)

true_peptides = []
true_pepmasses = []

input_specs_npy = {
    "mzs": np.float32,
    "intensities": np.float32,
    "usi": str,
    #"charge": float,
    "precursorMZ": float,
}

for psm in tqdm(list(map(lambda file_location: parse_json_npy(file_location,specs=input_specs_npy), files[-N:]))):

    #charge=psm['charge']
    precursorMZ=float(psm['precursorMZ'])
    usi=str(psm['usi'])
    collection_identifier, run_identifier, index, charge, peptideSequence, positions = parse_usi(usi)
    true_peptides.append(peptideSequence)
    true_pepmasses.append(float(charge)*precursorMZ)

thoretical_pepmasses = np.array(list(map(theoretical_peptide_mass,true_peptides)))

embedded_spectra = spectrum_embedder.predict(ds_spectra)
embedded_peptides = sequence_embedder.predict(ds_peptides)


embedded_spectrum = embedded_spectra[0]
embedded_peptide = embedded_peptides[0]
true_peptide = true_peptides[0]

#peptide_as_indices = np.array(get_sequence_of_indices(true_peptide))
#print(peptide_as_indices)

#from Bio.SubsMat.MatrixInfo import pam90 as pam
from Bio.Align import substitution_matrices
from Bio import pairwise2

pam = substitution_matrices.load("PAM70")
pam = pam
print(substitution_matrices.load())


def change_single_aa(aa,new_aa,substitution_matrix=pam):
    logodd_to_prob = lambda x: np.exp(x*np.log(2))
    m_ = logodd_to_prob(substitution_matrix)
    return m_[aa,new_aa]/(np.sum(m_[aa,:]))
    

def mutate_single_aa(aa,substitution_matrix=pam,method='random'):
    logodd_to_prob = lambda x: np.exp(x*np.log(2))
    m_ = logodd_to_prob(substitution_matrix)
    n = m_.shape[0]
    probabilities_aa = m_[aa,:]/(np.sum(m_[aa,:]))

    if True:
        index_of_aa = m_.alphabet.index(aa)
        probabilities_aa[index_of_aa] = 0.0
        softmax = lambda x : np.exp(x)/sum(np.exp(x))
        probabilities_aa = softmax(probabilities_aa)

    if method == 'random':
        selected = np.random.choice(range(n), 1, p=probabilities_aa)
    elif method== 'max':
        selected = [np.argmax(probabilities_aa)]
    mutation = [m_.alphabet[s] for s in selected]
    mutation = [aa if x=='*' else x for x in mutation]
    return mutation

def mutate(sequence,position=None):
    if position == None:
        selected_position = random.randint(0,len(sequence)-1)
        method='random'
    else:
        selected_position = position
        method='max'
    char_at_position = str(sequence[selected_position])
    mutation = mutate_single_aa(char_at_position,pam,method)[0]
    return sequence[:selected_position] + mutation + sequence[selected_position + 1:]


def random_process(sequence):
    x_ = mutate(sequence)
    x = parse_peptide(x_)
    x = np.expand_dims(x,0)
    return x_,sequence_embedder(x)

diff = lambda x,y : np.squeeze(np.linalg.norm(x-y,2))
#diff = lambda x,y : np.squeeze(np.linalg.norm(x-y,1))
#diff = lambda x,y : np.squeeze(np.matmul(x,np.array(y).T))

#norm = lambda x : np.squeeze(np.sqrt(np.inner(x,x)))
#diff = lambda x,y : np.squeeze(norm(x-y))

sigmoid = lambda x: 1 / (1 + np.exp(-x))

def deterministic_process_matrix(sequence,embedded_spectrum,subsitution_matrix=pam):    
    p,q = len(sequence),len(subsitution_matrix.alphabet[:-1])
    dist_probs = np.zeros((p,q))
    subsitution_probs = np.zeros((p,q))


    x = parse_peptide(sequence)
    x = np.expand_dims(x,0)
    initial_dist = diff(sequence_embedder(x),embedded_spectrum)

    for pos in tqdm(range(p)):
        for i,mutation in enumerate(subsitution_matrix.alphabet[:-1]):
            x_ = sequence[:pos] + mutation + sequence[pos + 1:]
            x = parse_peptide(x_)
            x = np.expand_dims(x,0)
            distance = initial_dist - diff(sequence_embedder(x),embedded_spectrum)
            distance_ = sigmoid(distance)
            dist_probs[pos,i] = distance_
            subsitution_probs[pos,i] = change_single_aa(sequence[pos],mutation,subsitution_matrix)
            #print(sigmoid(distance),np.exp(change_single_aa(sequence[pos],mutation,subsitution_matrix)))
        #quit()

    return dist_probs,subsitution_probs


def deterministic_process(sequence,embedded_spectrum):    
    for pos in tqdm(range(len(sequence))):
        distances = []
        for mutation in pam.alphabet[:-1]:
            x_ = sequence[:pos] + mutation + sequence[pos + 1:]
            #x = mutate(sequence,pos)
            x = parse_peptide(x_)
            x = np.expand_dims(x,0)
            distance = diff(sequence_embedder(x),embedded_spectrum)
            distance_ = -np.log(change_single_aa(sequence[pos],mutation)+0.001) * distance
            distances.append(distance_)
            print(mutation,distance,distance_,change_single_aa(sequence[pos],mutation))
        lowest = np.argmax(-np.array(distances))
        sequence = x_[:pos] + pam.alphabet[:-1][lowest] + x_[pos + 1:]
    x_ = parse_peptide(sequence)
    x_ = np.expand_dims(x_,0)
    return sequence,diff(sequence_embedder(x_),embedded_spectrum)

#mutated_peptide = "DLEHVVLDEADQMLBMBFIHDYKK"

x = copy.copy(true_peptide)
for _ in range(1):
    x = mutate(x)
    print(x)


mutated_peptide = copy.copy(x)

for _ in range(1):

    dist_probs,subsitution_probs = deterministic_process_matrix(mutated_peptide,embedded_spectrum,pam.T)

    standardize = lambda A: A / np.std(A)

    dist_probs = standardize(dist_probs)
    subsitution_probs = standardize(subsitution_probs)

    print(dist_probs[:4,:4])
    print(subsitution_probs[:4,:4])

    #matrix = subsitution_probs
    matrix = dist_probs * subsitution_probs

    from beamsearch import beam_search_decoder

    indices = beam_search_decoder(matrix, 3)

    indices_greedy = beam_search_decoder(matrix, 1)

    print('greedy:',indices[0][0] == indices_greedy[0])

    for i in range(3):
        decoded_sequence = get_sequence_from_indices(indices[i][0],pam.alphabet[:-1])
        alignments = pairwise2.align.localxx(true_peptide, decoded_sequence)
        print(pairwise2.format_alignment(*alignments[0]))
        alignments = pairwise2.align.localxx(true_peptide, mutated_peptide)
        print(pairwise2.format_alignment(*alignments[0]))

    mutated_peptide = get_sequence_from_indices(indices[i][0],pam.alphabet[:-1])

quit()

keep, final_dist = deterministic_process(mutated_peptide,embedded_spectrum)
print(final_dist,diff(embedded_spectrum,embedded_peptide))
alignments = pairwise2.align.localxx(true_peptide, mutated_peptide)
print(pairwise2.format_alignment(*alignments[0]))

alignments = pairwise2.align.localxx(true_peptide, keep)
print(pairwise2.format_alignment(*alignments[0]))

quit()

lowest_dist = 10.

for _ in range(2):
    #x = true_peptide
    x = mutated_peptide
    y = parse_peptide(x)
    y = np.expand_dims(y,0)

    initial_dist = diff(sequence_embedder(y),embedded_spectrum)
    dist = initial_dist
    delta = -0.5
    k = 0
    for i in tqdm(range(1000)):
        if k>50:
            break
        candidate,embedded_candidate = random_process(x)
        update = diff(embedded_spectrum,embedded_candidate) - dist    
        if update < delta:
            x = candidate
            dist = update + dist
            if dist < lowest_dist:
                lowest_dist = dist
                keep = candidate
            print(x,dist)
            print(diff(embedded_peptide,embedded_candidate))         
        else:
            k+=1
            pass

chain = [mutate(true_peptide) for _ in tqdm(range(1000))]
#print(true_peptide,mutated_peptide)

x = list(map(parse_peptide,[true_peptide]))
x = np.array(x)

x_ = list(map(parse_peptide,tqdm(chain)))
x_ = np.array(x_)

embedded_peptide = sequence_embedder(x)
embedded_peptide_ = sequence_embedder(x_)



a = [diff(embedded_peptide,x) for x in tqdm(embedded_peptide_)]
b = diff(embedded_spectrum,embedded_peptide)
c = [diff(embedded_spectrum,x) for x in tqdm(embedded_peptide_)]

acc = np.argmax(c-b)

print(true_peptide)
print(chain[acc])

import matplotlib.pyplot as plt
import seaborn as sns


sns.jointplot(x=c-b,y=np.array(a))

#plt.hist(c-b,alpha=0.5)
#plt.hist(a,alpha=0.5)
#plt.yscale('log')
plt.savefig('figures/mutation_effecton_dists.png')