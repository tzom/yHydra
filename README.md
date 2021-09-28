# yHydra

python fasta2db.py --FASTA_FILE uniprot_sprot.fasta.gz --fasta_type=uniprot --MAX_MISSED_CLEAVAGES=1 --DB_DIR=./db_miscleav_1
python embed_db.py --DB_DIR db_miscleav_1/
python search.py --DB_DIR db_miscleav_1/ --JSON_DIR='../../scratch/USI_files/PXD007963/**/' --OUTPUT_DIR=./output
python search_score.py --OUTPUT_DIR=./output
python fdr_filter.py --OUTPUT_DIR=./tmp/forward --REV_OUTPUT_DIR=./tmp/rev