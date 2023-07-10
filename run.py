import sys,os 
os.environ['YHYDRA_CONFIG'] = sys.argv[1]
os.environ['LD_LIBRARY_PATH'] = ""
import setup_device 

from load_config import CONFIG
import glob

RAWs = glob.glob(CONFIG['RAWs'])
FASTA = glob.glob(CONFIG['FASTA'])[0] 

# from fasta2db import digest_fasta
# digest_fasta(FASTA,REVERSE_DECOY=False)
# digest_fasta(FASTA,REVERSE_DECOY=True)

# from sanitize_db import sanitize_db
# sanitize_db()

# from embed_db import embed_db
# embed_db(REVERSE_DECOY=False)
# embed_db(REVERSE_DECOY=True)

# from calc_masses_db import calc_masses
# calc_masses(REVERSE_DECOY=False)
# calc_masses(REVERSE_DECOY=True)
# import gc
# gc.collect()

# # # from pyThermoRawFileParser import parse_rawfiles
# # # raw = parse_rawfiles(RAWs)

from search import search
[search(RAW.replace('.raw','.mgf')) for RAW in RAWs]
#s = [search(RAW) for RAW in RAWs]
import gc
gc.collect()

from search_score import search_score
search_score()

from fdr_filter import fdr_filter
fdr_filter()