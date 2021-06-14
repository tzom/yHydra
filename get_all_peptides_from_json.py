import sys 
sys.path.append('..')
from tf_data_json import parse_json_,parse_json_npy,parse_usi
import glob, os
import json
from tqdm import tqdm

#files = glob.glob(os.path.join('../../Neonomicon/files/**/','*.json'))[:10000]
files = glob.glob(os.path.join('../../Neonomicon/dump/','*.json'))[:1000]

peptides = []

for psm in tqdm(list(map(lambda file_location: parse_json_npy(file_location), files))):
    usi = str(psm['usi'])
    collection_identifier, run_identifier, index, charge, peptideSequence, positions = parse_usi(usi)
    peptides.append(peptideSequence)

print(len(peptides))

with open('db.json','r') as f:
    db = json.load(f)

peptides_in_db = set(db.keys())

set_peptides = set(peptides)

print(len(set_peptides.intersection(peptides_in_db)))

# for peptide in list(map(lambda file_location: parse_json_(file_location)[1], files)):
#     print(peptide)

