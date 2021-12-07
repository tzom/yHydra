import os
import tensorflow as tf

model_dir = './saved_27_06_2021'

if not os.path.exists(model_dir):
    os.system("tar -xvf %s"%model_dir+'.tar.gz')

loaded_model = tf.keras.models.load_model(model_dir,custom_objects={'metric_acc':lambda x:x})
#loaded_model.summary()

def get_indidividual_embedder(model):

    spec_input = model.get_layer('spectrum_input').input
    seq_input = model.get_layer('sequence_input').input

    spec_embedding = model.get_layer("spec_emb").output
    seq_embedding = model.get_layer("seq_emb").output

    spectrum_embedder = tf.keras.Model(inputs=spec_input,outputs=spec_embedding)
    sequence_embedder = tf.keras.Model(inputs=seq_input,outputs=seq_embedding)    

    return spectrum_embedder,sequence_embedder

spectrum_embedder,sequence_embedder = get_indidividual_embedder(loaded_model)