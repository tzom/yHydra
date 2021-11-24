#!/bin/bash

# source /hpi/fs00/home/tom.altenburg/conda/bin/activate 
# conda activate yhydra_env

source /hpi/fs00/home/tom.altenburg/conda/bin/activate 
conda activate yhydra_gpu_env

# source /hpi/fs00/home/tom.altenburg/scratch/powerconda/bin/activate 
# conda activate yhydra_power9_env

#wget https://www.uniprot.org/uniprot/?query=taxonomy:63366&format=fasta -O taxid_63366.fasta

OUTPUT_DIR='/hpi/fs00/home/tom.altenburg/scratch/yHydra_testing/PXD007963_search/forward'
DB_DIR=$OUTPUT_DIR'/db_miscleav_1'

DECOY_OUTPUT_DIR='/hpi/fs00/home/tom.altenburg/scratch/yHydra_testing/PXD007963_search/rev'
DECOY_DB_DIR=$DECOY_OUTPUT_DIR'/db_miscleav_1'

#FASTA='/hpi/fs00/home/tom.altenburg/projects/yHydra/uniprot_sprot.fasta.gz'

#FASTA='test/SynPCC7002_Cbase.fasta.gz'
#JSON_DIR='../../scratch/USI_files/PXD007963/**/'

#FASTA='/hpi/fs00/home/tom.altenburg/scratch/yHydra_testing/PXD003916/UP000000559_237561.fasta.gz'
#MGF='/hpi/fs00/home/tom.altenburg/scratch/yHydra_testing/PXD003916/raw/Michelle-Experimental-Sample3.mgf'

FASTA='test/SynPCC7002_Cbase.fasta.gz'
#MGF='/hpi/fs00/home/tom.altenburg/scratch/yHydra_testing/PXD007963/raw/qe2_03132014_1WT-1.mgf'
MGFs=$(ls /hpi/fs00/home/tom.altenburg/scratch/yHydra_testing/PXD007963/raw/*WT-*.mgf)
DEBUG_N=10000

GPU='0'

echo $FASTA
echo $OUTPUT_DIR
echo $DB_DIR

mkdir -p $OUTPUT_DIR
mkdir -p $DECOY_OUTPUT_DIR

start=`date +%s`

python fasta2db.py --FASTA_FILE=$FASTA --MAX_MISSED_CLEAVAGES=1 --DB_DIR=$DB_DIR
python fasta2db.py --FASTA_FILE=$FASTA --MAX_MISSED_CLEAVAGES=1 --DB_DIR=$DECOY_DB_DIR --REVERSE_DECOY=TRUE
python sanitize_db.py --DB_DIR=$DB_DIR --DECOY_DB_DIR=$DECOY_DB_DIR

python embed_db.py --DB_DIR=$DB_DIR --GPU=$GPU
python embed_db.py --DB_DIR=$DECOY_DB_DIR --GPU=$GPU

for MGF in $MGFs
do
    python search.py --DB_DIR=$DB_DIR --DECOY_DB_DIR=$DECOY_DB_DIR --MGF=$MGF --OUTPUT_DIR=$OUTPUT_DIR --GPU=$GPU
done
python search_score.py --OUTPUT_DIR=$OUTPUT_DIR --GPU=$GPU
python fdr_filter.py --OUTPUT_DIR=$OUTPUT_DIR

end=`date +%s`

echo "$((end-start)) seconds" >> "$OUTPUT_DIR/run.log"