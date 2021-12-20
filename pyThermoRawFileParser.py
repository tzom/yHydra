import subprocess
from load_config import CONFIG
NUMBER_OF_THREADS = CONFIG['NUMBER_OF_THREADS']

def create_CMD(RAW):
    return ' '.join(['mono','ext/ThermoRawFileParser/ThermoRawFileParser.exe', "-L", "2", "-f", "0", "-i", RAW])

def parse_rawfile(RAW):
    p = subprocess.Popen(create_CMD(RAW)
    #stdout=subprocess.DEVNULL,
    )
    errors = p.communicate()
    return None

def parse_rawfiles(RAWs):
    from functools import partial
    from multiprocessing.dummy import Pool
    from subprocess import call

    pool = Pool(NUMBER_OF_THREADS) # two concurrent commands at a time
    commands = list(map(create_CMD,RAWs))
    for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands)):
        if returncode != 0:
            print("%d command failed: %d" % (i, returncode))
    return None


if __name__ == '__main__':
    import glob,os
    os.environ['YHYDRA_CONFIG'] = 'config.yaml'

    from load_config import CONFIG
    RAWs = glob.glob(CONFIG['RAWs'])   
    
    parse_rawfiles(RAWs)
