### SOURCE: https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
from time import perf_counter
import os,psutil

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

class cntxt:

    def __init__(self, txt_msg):
        print(txt_msg)

    def __enter__(self):
        self.time = perf_counter()
        self.memory = get_process_memory()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.delta_memory = (get_process_memory() - self.memory)/1e9
        self.memory = get_process_memory()/1e9        
        self.time_readout = f'Time: {self.time:.2f}s'
        #self.delta_memory_readout = f'DeltaMemory:  GBs'
        self.memory_readout = f'Memory: {self.memory:.2f}[{self.delta_memory:.2f}]GBs'
        print(self.time_readout,self.memory_readout)