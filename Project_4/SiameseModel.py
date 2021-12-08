import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM 
from keras.layers import SimpleRNN as RNN
from keras.layers import Dropout
from keras.layers import ReLU
from keras.optimizers import Adam
from keras.layers import Lambda
from keras import Input
import tensorflow.keras.backend as K
from keras.models import Model
import numpy as np


def euclidean_distance(outputs):
    (traj1_feats, traj2_feats) = outputs
    return K.sqrt(K.sum(K.square(traj1_feats - traj2_feats), axis=1, keepdims=True))

def manhattan_distance(outputs):
    (traj1_feats, traj2_feats) = outputs
    return K.sum(K.abs(traj1_feats - traj2_feats),axis=1, keepdims=True)

# def create_rnn(input_shape):

#     inputs = Input(input_shape)

#     x = RNN(16, activation='relu', return_sequences=True)(inputs)
#     x = Dropout(0.2)(x)
#     x = RNN(16, activation='relu')(x)
#     x = Dropout(0.2)(x)
#     x = Dense(16)(x)
#     x = ReLU()(x)
#     x = Dropout(0.2)(x)
#     x = Dense(16)(x)
#     x = ReLU()(x)
#     out = Dropout(0.2)(x)

#     rnn = Model(inputs, out)
#     print(rnn.summary())
    
    
#     return rnn

def create_rnn(input_shape):

    inputs = Input(input_shape)

    x = LSTM(16, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(16)(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)
    x = Dense(16)(x)
    x = ReLU()(x)
    out = Dropout(0.3)(x)

    rnn = Model(inputs, out)
    print(rnn.summary())
    
    
    return rnn