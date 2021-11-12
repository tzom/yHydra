import Bio,gzip
from Bio import SeqIO
import pyteomics
from pyteomics import mass,fasta
import pyteomics.parser as pyt_parser
import pandas as pd
import numpy as np
import json,os
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='convert')
parser.add_argument('--FASTA_FILE', type=str, help='path to fasta file')
parser.add_argument('--fasta_type', default='generic', type=str, help='uniprot/ncbi/generic')
parser.add_argument('--MAX_MISSED_CLEAVAGES', default=1, type=int, help='maximum number of miscleavages')
parser.add_argument('--DB_DIR', default='./DB', type=str, help='path to db file')
parser.add_argument('--REVERSE_DECOY', default=False, type=bool, help='path to db file')


args = parser.parse_args()

from load_config import CONFIG

MAX_DATABASE_SIZE=100000000
PEPTIDE_MINIMUM_LENGTH=CONFIG['PEPTIDE_MINIMUM_LENGTH']#7
PEPTIDE_MAXIMUM_LENGTH=CONFIG['PEPTIDE_MAXIMUM_LENGTH']#42
MAX_MISSED_CLEAVAGES=CONFIG['MAX_MISSED_CLEAVAGES']#args.MAX_MISSED_CLEAVAGES
SEMI_SPECIFIC_CLEAVAGE=CONFIG['SEMI_SPECIFIC_CLEAVAGE']
EMBED=True
SAVE_DB_AS_JSON=False

FASTA_FILE = args.FASTA_FILE
fasta_type = args.fasta_type
DB_DIR = args.DB_DIR

# FASTA_FILE = 'uniprot_sprot.fasta.gz'
# fasta_type = 'uniprot'
# DB_DIR = './db_miscleav_1'

if not os.path.exists(DB_DIR):
    os.mkdir(DB_DIR)

def add_check_keys_exising(key,dictionary,element):
    if key in dictionary:
        dictionary[key].add(element)
    else: 
        dictionary[key] = set([element])
    return dictionary   

def cleave_peptide(protein_sequence):
    return pyt_parser.cleave(protein_sequence, pyt_parser.expasy_rules['trypsin'],min_length=PEPTIDE_MINIMUM_LENGTH,missed_cleavages=MAX_MISSED_CLEAVAGES, semi=SEMI_SPECIFIC_CLEAVAGE)

def digest_seq_record(seq_record):
    ID=None
    HEADER = seq_record[0]
    SEQ = seq_record[1]

    if fasta_type=='generic':
        accesion_id = ID
        speciesName = None
        protName = HEADER
    if fasta_type=='uniprot':               
        accesion_id = ID
        speciesName = HEADER.split("OS=")[1].split("OX=")[0]
        prot = HEADER.split("|")[1]
        protName = HEADER.split("|")[2].split("OX=")[0]
    elif fasta_type=='ncbi':
        accesion_id = ID
        speciesName = HEADER.split("[")[1][:-1]
        prot = HEADER.split("[")[0]
        protName = " ".join(prot.split(" ")[1:])
    #SEQ = str(seq_record.seq)
    
    cleaved_peptides = cleave_peptide(SEQ)

    LENGTH_CONDITION = lambda x: not (len(x) > PEPTIDE_MAXIMUM_LENGTH or len(x) < PEPTIDE_MINIMUM_LENGTH)
    cleaved_peptides = list(filter(LENGTH_CONDITION,cleaved_peptides))

    ODD_AMINOACIDS_CONDITION = lambda x: not (len(set(x).intersection(set(['X','U','J','Z','B','O'])))>0)
    cleaved_peptides = list(filter(ODD_AMINOACIDS_CONDITION,cleaved_peptides))

    accesion_id = HEADER.split()[0]
    return HEADER, accesion_id, cleaved_peptides

from collections import defaultdict

if __name__ == '__main__':

    ncbi_peptide_protein = defaultdict(set)
    ncbi_peptide_meta = {}

    all_peptides = []
    all_proteins = []

    print('Digesting peptides...')

    from multiprocessing.pool import Pool, ThreadPool

    with Pool() as p, ThreadPool() as tp:

        with gzip.open(FASTA_FILE, "rt") as FASTA_FILE:
            if args.REVERSE_DECOY:
                FASTA_FILE = fasta.decoy_db(FASTA_FILE,decoy_only=True)
            else:
                FASTA_FILE = fasta.read(FASTA_FILE)
            #seqio = SeqIO.parse(FASTA_FILE, "fasta")
            for seq_record in tqdm(p.map(digest_seq_record,FASTA_FILE)):
                #ID = seq_record.id
                #HEADER = seq_record.description
                #SEQ = str(seq_record.seq)


                HEADER, accesion_id, cleaved_peptides = seq_record

                list(map(lambda peptide: add_check_keys_exising(peptide,ncbi_peptide_protein,accesion_id),cleaved_peptides))

                # for peptide in cleaved_peptides:

                # #     peptide_protein_entry={'accesion_id':accesion_id,'speciesName':speciesName,'protName':protName}

                #     add_check_keys_exising(peptide,ncbi_peptide_protein,accesion_id)

                if len(ncbi_peptide_protein) > MAX_DATABASE_SIZE:
                    print('exceeding maximum number of allowd peptides %s'%MAX_DATABASE_SIZE)
                    break

                
    #ncbi_peptide_protein = dict(zip(all_peptides,all_proteins))        

    print('Done.')
    print(len(ncbi_peptide_protein))
    if SAVE_DB_AS_JSON:
        print('saving db as db.json... ')
        import json
        with open(os.path.join(DB_DIR,'db.json'), 'w') as fp:
            json.dump(ncbi_peptide_protein, fp)

    if EMBED:
        print('Embed peptides... ')
        peptides = list(ncbi_peptide_protein.keys())
        #pepmasses = list(map(theoretical_peptide_mass,tqdm(peptides)))
        np.save(os.path.join(DB_DIR,"peptides.npy"),np.array(peptides))
        #np.save(os.path.join(DB_DIR,"pepmasses.npy"),np.array(pepmasses))
        #embeddings = list(map(seq_embedder,tqdm(peptides)))
        print('Done.')

