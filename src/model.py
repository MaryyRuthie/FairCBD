import tensorflow.keras as keras
from tensorflow.keras import layers,Input

from configs import config as cf


def get_dense_model():
    '''
    define dense model architecture
    '''
    # model = keras.Sequential()
    # model.add(layers.Dense(512, activation='relu',
    #                        input=(bert_embed_dim,)))
    # model.add(layers.Dense(32, activation='relu'))
    # model.add(layers.Dense(1))
    input_layer = Input(shape=(cf.bert_embed_dim,))

# Define the model
    model = keras.Sequential([
        input_layer,
        layers.Dense(512, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    return model