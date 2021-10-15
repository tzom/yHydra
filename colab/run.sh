#!/bin/bash

mkdir -p "./data"
cp /content/drive/MyDrive/datasets/PXD026566/* ./data

OUTPUT_DIR='data/forward'
DB_DIR=$OUTPUT_DIR'/db_miscleav_1'

DECOY_OUTPUT_DIR='data/rev'
DECOY_DB_DIR=$DECOY_OUTPUT_DIR'/db_miscleav_1'

FASTA='data/10090.fasta.gz'
MGF='data/20210311_HW4_001.mgf'

GPU='0'

mkdir -p $OUTPUT_DIR
mkdir -p $DB_DIR

mkdir -p $DECOY_OUTPUT_DIR
mkdir -p $DECOY_DB_DIR

python ../fasta2db.py --FASTA_FILE=$FASTA --MAX_MISSED_CLEAVAGES=1 --DB_DIR=$DB_DIR
python ../fasta2db.py --FASTA_FILE=$FASTA --MAX_MISSED_CLEAVAGES=1 --DB_DIR=$DECOY_DB_DIR --REVERSE_DECOY=TRUE

# python ../embed_db.py --DB_DIR=$DB_DIR --GPU=$GPU
# python ../embed_db.py --DB_DIR=$DECOY_DB_DIR --GPU=$GPU

# python ../search.py --DB_DIR=$DB_DIR --DECOY_DB_DIR=$DECOY_DB_DIR --MGF="$MGF" --OUTPUT_DIR=$OUTPUT_DIR --GPU=$GPU

# python ../search_score.py --OUTPUT_DIR=$OUTPUT_DIR --GPU=$GPU
# python ../fdr_filter.py --OUTPUT_DIR=$OUTPUT_DIR 