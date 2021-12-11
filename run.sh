#!/bin/bash
CONFIG="$1"
#source /hpi/fs00/home/tom.altenburg/conda/bin/activate 
#conda activate yhydra_env

#source /hpi/fs00/home/tom.altenburg/conda/bin/activate 
#conda activate yhydra_gpu

conda activate yhydra_env

RESULTS_DIR=$(cat $CONFIG | yq -r .RESULTS_DIR)

OUTPUT_DIR=$RESULTS_DIR'/forward'
DB_DIR=$OUTPUT_DIR'/db'

DECOY_OUTPUT_DIR=$RESULTS_DIR'/rev'
DECOY_DB_DIR=$DECOY_OUTPUT_DIR'/db'

FASTA=$(cat $CONFIG | yq -r .FASTA)
FASTA=$(ls ${FASTA})

#MGFs=$(cat $CONFIG | yq -r .MGFs)
#MGFs=$(ls ${MGFs})

RAWs=$(cat $CONFIG | yq -r .RAWs)
RAWs=$(ls ${RAWs})

GPU=$(cat $CONFIG | yq .GPU)
MAX_MISSED_CLEAVAGES=$(cat $CONFIG | yq .MAX_MISSED_CLEAVAGES)

mkdir -p $OUTPUT_DIR
mkdir -p $DECOY_OUTPUT_DIR

python fasta2db.py --FASTA_FILE=$FASTA --MAX_MISSED_CLEAVAGES=$MAX_MISSED_CLEAVAGES --DB_DIR=$DB_DIR
python fasta2db.py --FASTA_FILE=$FASTA --MAX_MISSED_CLEAVAGES=$MAX_MISSED_CLEAVAGES --DB_DIR=$DECOY_DB_DIR --REVERSE_DECOY=TRUE
python sanitize_db.py --DB_DIR=$DB_DIR --DECOY_DB_DIR=$DECOY_DB_DIR

python embed_db.py --DB_DIR=$DB_DIR --GPU=$GPU
python embed_db.py --DB_DIR=$DECOY_DB_DIR --GPU=$GPU

for RAW in $RAWs
do
   mono ext/ThermoRawFileParser/ThermoRawFileParser.exe -L=2 -f=0 -i=$RAW &
done

wait

for RAW in $RAWs
do
   MGF=${RAW%.raw}.mgf
   #mono ~/tools/ThermoRawFileParser/ThermoRawFileParser.exe -L=2 -f=0 -s -i=$MGF | python search.py --DB_DIR=$DB_DIR --DECOY_DB_DIR=$DECOY_DB_DIR --MGF=$MGF --OUTPUT_DIR=$OUTPUT_DIR --GPU=$GPU
   python search.py --DB_DIR=$DB_DIR --DECOY_DB_DIR=$DECOY_DB_DIR --MGF=$MGF --OUTPUT_DIR=$OUTPUT_DIR --GPU=$GPU
done
python search_score.py --OUTPUT_DIR=$OUTPUT_DIR --GPU=$GPU
python fdr_filter.py --OUTPUT_DIR=$OUTPUT_DIR
