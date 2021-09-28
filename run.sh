#!/bin/bash

source activate 
conda activate yhydra_env

#wget https://www.uniprot.org/uniprot/?query=taxonomy:63366&format=fasta -O taxid_63366.fasta

OUTPUT_DIR='test/forward'
DB_DIR=$OUTPUT_DIR'/db_miscleav_1'

DECOY_OUTPUT_DIR='test/rev'
DECOY_DB_DIR=$DECOY_OUTPUT_DIR'/db_miscleav_1'

FASTA='test/SynPCC7002_Cbase.fasta.gz'
JSON_DIR='../../scratch/USI_files/PXD007963/**/'
DEBUG_N=10000

echo $FASTA
echo $OUTPUT_DIR
echo $DB_DIR
echo $JSON_DIR

python fasta2db.py --FASTA_FILE=$FASTA --MAX_MISSED_CLEAVAGES=1 --DB_DIR=$DB_DIR
python embed_db.py --DB_DIR=$DB_DIR
python search.py --DB_DIR=$DB_DIR --JSON_DIR=${JSON_DIR} --OUTPUT_DIR=$OUTPUT_DIR --DEBUG_N=$DEBUG_N
python search_score.py --OUTPUT_DIR=$OUTPUT_DIR

python fasta2db.py --FASTA_FILE=$FASTA --MAX_MISSED_CLEAVAGES=1 --DB_DIR=$DECOY_DB_DIR --REVERSE_DECOY=TRUE
python embed_db.py --DB_DIR=$DECOY_DB_DIR
python search.py --DB_DIR=$DECOY_DB_DIR --JSON_DIR=${JSON_DIR} --OUTPUT_DIR=$DECOY_OUTPUT_DIR --DEBUG_N=$DEBUG_N
python search_score.py --OUTPUT_DIR=$DECOY_OUTPUT_DIR

python fdr_filter.py --OUTPUT_DIR=$OUTPUT_DIR --REV_OUTPUT_DIR=$DECOY_OUTPUT_DIR