from load_config import CONFIG
GPU = str(CONFIG['GPU'])
import os
# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.eager.context import device 

if GPU == '-1':
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    import tensorflow as tf
    device = '/CPU:0'
    use_gpu=False    
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    import tensorflow as tf
    device = '/GPU:0'
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0],True)
    use_gpu=True

