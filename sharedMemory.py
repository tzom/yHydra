from multiprocessing import shared_memory
import numpy as np
import atexit

class sharedMemory():
    def __init__(self,type=np.float32, shape=(1,), name='shared_memory',):
    #def create(NP_DATA_TYPE=np.float32, ARRAY_SHAPE=(1,), NP_SHARED_NAME='shared_memory',):
        NP_DATA_TYPE = type
        ARRAY_SHAPE = shape
        NP_SHARED_NAME = name
        d_size = np.dtype(NP_DATA_TYPE).itemsize * np.prod(ARRAY_SHAPE)
        shm = shared_memory.SharedMemory(create=True, size=d_size, name=NP_SHARED_NAME)
        self.shm = shm
        self.array = np.ndarray(shape=ARRAY_SHAPE, dtype=NP_DATA_TYPE, buffer=shm.buf)   
        atexit.register(self.release) 

    def assign(self,data):
        self.array[:] = data[:]

    def release(self,):
        try:
            self.shm.close()
            self.shm.unlink()
        except:
            pass

if __name__ == '__main__':
    from cntxt import cntxt
    N=10000000
    with cntxt('reserve shared mem'):
        shm = sharedMemory(type=np.float32, shape=(N,42,4))
        print(shm.array[0])   
    
    shm.release()