#!/bin/bash

#source /hpi/fs00/home/tom.altenburg/conda/bin/activate 
#conda activate yhydra_env

source /hpi/fs00/home/tom.altenburg/conda/bin/activate 
conda activate yhydra_gpu_env

RESULTS_DIR=$(cat config.yaml | yq -r .RESULTS_DIR)

OUTPUT_DIR=$RESULTS_DIR'/forward'
DB_DIR=$OUTPUT_DIR'/db'

DECOY_OUTPUT_DIR=$RESULTS_DIR'/rev'
DECOY_DB_DIR=$DECOY_OUTPUT_DIR'/db'

FASTA=$(cat config.yaml | yq -r .FASTA)
MGFs=$(cat config.yaml | yq -r .MGFs)

MGFs=$(ls ${MGFs})

GPU=$(cat config.yaml | yq .GPU)
MAX_MISSED_CLEAVAGES=$(cat config.yaml | yq .MAX_MISSED_CLEAVAGES)

mkdir -p $OUTPUT_DIR
mkdir -p $DECOY_OUTPUT_DIR

python fasta2db.py --FASTA_FILE=$FASTA --MAX_MISSED_CLEAVAGES=$MAX_MISSED_CLEAVAGES --DB_DIR=$DB_DIR
python fasta2db.py --FASTA_FILE=$FASTA --MAX_MISSED_CLEAVAGES=$MAX_MISSED_CLEAVAGES --DB_DIR=$DECOY_DB_DIR --REVERSE_DECOY=TRUE
python sanitize_db.py --DB_DIR=$DB_DIR --DECOY_DB_DIR=$DECOY_DB_DIR

python embed_db.py --DB_DIR=$DB_DIR --GPU=$GPU
python embed_db.py --DB_DIR=$DECOY_DB_DIR --GPU=$GPU

for MGF in $MGFs
do
   #mono ~/tools/ThermoRawFileParser/ThermoRawFileParser.exe -L=2 -f=0 -s -i=$MGF | python search.py --DB_DIR=$DB_DIR --DECOY_DB_DIR=$DECOY_DB_DIR --MGF=$MGF --OUTPUT_DIR=$OUTPUT_DIR --GPU=$GPU
   python search.py --DB_DIR=$DB_DIR --DECOY_DB_DIR=$DECOY_DB_DIR --MGF=$MGF --OUTPUT_DIR=$OUTPUT_DIR --GPU=$GPU
done
python search_score.py --OUTPUT_DIR=$OUTPUT_DIR --GPU=$GPU
python fdr_filter.py --OUTPUT_DIR=$OUTPUT_DIR