import os
import tensorflow as tf
from load_config import CONFIG

USE_CHARGE=True
#model_dir = './saved_27_06_2021'
model_dir = "/hpi/fs00/home/tom.altenburg/projects/yHydra_train/non_tryptic_weigths/saved_23_03_2022"
if "USE_TIMSTOF" in CONFIG.keys():
    model_dir = "/hpi/fs00/home/tom.altenburg/projects/yHydra_train/non_tryptic_weigths/yhydra_charge_nontryptic"

if not os.path.exists(model_dir):
    os.system("tar -xvf %s"%model_dir+'.tar.gz')

loaded_model = tf.keras.models.load_model(model_dir,custom_objects={'metric_acc':lambda x:x})
#loaded_model.summary()

def get_indidividual_embedder(model):

    spec_input = model.get_layer('spectrum_input').input
    seq_input = model.get_layer('sequence_input').input
    if USE_CHARGE:
        charge_input = model.get_layer('charge_input').input

    spec_embedding = model.get_layer("spec_emb").output
    seq_embedding = model.get_layer("seq_emb").output

    if USE_CHARGE:
        spectrum_embedder = tf.keras.Model(inputs=[spec_input,charge_input],outputs=spec_embedding)
    else: 
        spectrum_embedder = tf.keras.Model(inputs=spec_input,outputs=spec_embedding)
    sequence_embedder = tf.keras.Model(inputs=seq_input,outputs=seq_embedding)    

    return spectrum_embedder,sequence_embedder

spectrum_embedder,sequence_embedder = get_indidividual_embedder(loaded_model)