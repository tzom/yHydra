import sys,os 
os.environ['YHYDRA_CONFIG'] = sys.argv[1]#'config.yaml'

from fasta2db import digest_fasta
from pyThermoRawFileParser import parse_rawfiles
from sanitize_db import sanitize_db
import setup_device 
from embed_db import embed_db
from search import search
from search_score import search_score
from fdr_filter import fdr_filter
from load_config import CONFIG
import glob

RAWs = glob.glob(CONFIG['RAWs'])
FASTA = glob.glob(CONFIG['FASTA'])[0] 

f = digest_fasta(FASTA,REVERSE_DECOY=False)
r = digest_fasta(FASTA,REVERSE_DECOY=True)
raw = parse_rawfiles(RAWs)
s = sanitize_db()
e = embed_db(REVERSE_DECOY=False)
e = embed_db(REVERSE_DECOY=True)
s = [search(RAW.replace('.raw','.mgf')) for RAW in RAWs]
search_score()
fdr_filter()