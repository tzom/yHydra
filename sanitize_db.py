import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='convert')
parser.add_argument('--DB_DIR', default='./DB', type=str, help='path to db file')
parser.add_argument('--DECOY_DB_DIR', default='./DB', type=str, help='path to db file')

args = parser.parse_args()

DB_DIR = args.DB_DIR
DECOY_DB_DIR = args.DECOY_DB_DIR

print('loading peptides..')
db_peptides = np.load(os.path.join(DB_DIR,"peptides.npy"))
decoy_db_peptides = np.load(os.path.join(DECOY_DB_DIR,"peptides.npy"))

print('computing/removing intersection..')
decoy_db_peptides_set = set(decoy_db_peptides)
intersection = set(db_peptides).intersection(decoy_db_peptides_set)

sanitized_decoy_db_peptides = decoy_db_peptides_set - intersection

print('N peptides in DB: %s'%len(db_peptides))
print('N peptides in decoy DB: %s'%len(decoy_db_peptides_set))
print('N peptides in decoy DB: %s'%len(decoy_db_peptides))
print('N shared peptides removed from decoy DB: %s'%len(intersection))

print('saving sanitized decoy peptides...')
sanitized_decoy_db_peptides = np.array(list(sanitized_decoy_db_peptides))
np.save(os.path.join(DECOY_DB_DIR,"peptides.npy"),sanitized_decoy_db_peptides)