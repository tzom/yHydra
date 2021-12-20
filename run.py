import sys,os 
os.environ['YHYDRA_CONFIG'] = sys.argv[1]
import setup_device 

from load_config import CONFIG
import glob

RAWs = glob.glob(CONFIG['RAWs'])
FASTA = glob.glob(CONFIG['FASTA'])[0] 

from fasta2db import digest_fasta
f = digest_fasta(FASTA,REVERSE_DECOY=False)
r = digest_fasta(FASTA,REVERSE_DECOY=True)

from pyThermoRawFileParser import parse_rawfiles
raw = parse_rawfiles(RAWs)

from sanitize_db import sanitize_db
s = sanitize_db()

from embed_db import embed_db
e = embed_db(REVERSE_DECOY=False)
e = embed_db(REVERSE_DECOY=True)

from search import search
s = [search(RAW.replace('.raw','.mgf')) for RAW in RAWs]

from search_score import search_score
search_score()

from fdr_filter import fdr_filter
fdr_filter()