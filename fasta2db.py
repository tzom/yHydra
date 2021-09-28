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

MAX_DATABASE_SIZE=100000000
PEPTIDE_MINIMUM_LENGTH=7
PEPTIDE_MAXIMUM_LENGTH=42
MAX_MISSED_CLEAVAGES=args.MAX_MISSED_CLEAVAGES
EMBED=True

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
        tmp = dictionary[peptide]['proteins']
        tmp.append(element)
    else: 
        dictionary[peptide] = {'proteins':[element]}
    return dictionary

if __name__ == '__main__':

    ncbi_peptide_protein = {}
    ncbi_peptide_meta = {}

    print('Digesting peptides...')

    with gzip.open(FASTA_FILE, "rt") as FASTA_FILE:
        if args.REVERSE_DECOY:
            FASTA_FILE = fasta.decoy_db(FASTA_FILE,decoy_only=True)
        else:
            FASTA_FILE = fasta.read(FASTA_FILE)
        #seqio = SeqIO.parse(FASTA_FILE, "fasta")
        for seq_record in tqdm(FASTA_FILE):
            #ID = seq_record.id
            #HEADER = seq_record.description
            #SEQ = str(seq_record.seq)

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
            cleaved_peptides = pyt_parser.cleave(SEQ, pyt_parser.expasy_rules['trypsin'],min_length=PEPTIDE_MINIMUM_LENGTH,missed_cleavages=MAX_MISSED_CLEAVAGES)
            for peptide in cleaved_peptides:
                if len(peptide) > PEPTIDE_MAXIMUM_LENGTH or len(peptide) < PEPTIDE_MINIMUM_LENGTH:
                    continue
                
                # if peptide[-1] not in set(['K','R']):
                #     print(peptide,SEQ)


                if len(set(peptide).intersection(set(['X','U','J','Z','B','O'])))>0:
                    continue

                peptide_protein_entry={'accesion_id':accesion_id,'speciesName':speciesName,'protName':protName}

                add_check_keys_exising(peptide,ncbi_peptide_protein,peptide_protein_entry)

            if len(ncbi_peptide_protein) > MAX_DATABASE_SIZE:
                print('exceeding maximum number of allowd peptides %s'%MAX_DATABASE_SIZE)
                break

    print('Done.')
    print(len(ncbi_peptide_protein))

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

