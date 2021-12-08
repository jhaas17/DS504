import tensorflow as tf
import numpy as np 
import keras
import pickle
import random


from SiameseModel import *
from train import *
from plotting import *
from utils import *

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

print("Running on GPU: ", tf.test.is_gpu_available(cuda_only=True))

config = tf.ConfigProto( device_count = {'GPU': 1 },log_device_placement=False) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess) 


#DEFINE MODEL NAME FOR OUTPUT
modelname = 'Test' 

# Define parameters 
num_epoch = 20
subtraj_len = 64
batchsize = 32
lr = 0.0015

print("MODELNAME: " + modelname)
print("Batchsize: " + str(batchsize))
print("Subtrajectory Length: " + str(subtraj_len))
print("Learning Rate: " + str(lr))

print("Model: ")
traj1 = Input((subtraj_len,4))
traj2 = Input((subtraj_len,4))
rnn = create_rnn(input_shape=(subtraj_len,4))
traj1_feats = rnn(traj1)
traj2_feats = rnn(traj2)
distance_layer = Lambda(euclidean_distance)([traj1_feats, traj2_feats])
out = Dense(1, activation='sigmoid')(distance_layer)

model = Model(inputs=[traj1, traj2],outputs=out)

model.compile(loss="binary_crossentropy",optimizer=Adam(lr), metrics=["accuracy"])
print(model.summary())

train_data = pd.read_pickle('data/project_4_train.pkl')

train_data, train_labels = make_pairs(train_data)

val_data = pd.read_pickle('data/validate_set.pkl')
val_labels = pd.read_pickle('data/validate_label.pkl')

train_data, train_labels = process_data_train(train_data, train_labels, subtraj_len)

print(train_data.shape)
val_data, val_labels = process_data_train(val_data, val_labels, subtraj_len)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= 'Out/' + modelname + '.h5',
    save_weights_only=False,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

history = model.fit([np.array(train_data[:,0]),np.array(train_data[:,1])],
                train_labels, 
                validation_data=([np.array(val_data[:,0]),np.array(val_data[:,1])],val_labels), 
                batch_size=batchsize, epochs=num_epoch, verbose=2,
                callbacks=[model_checkpoint_callback])

plot_losses(history.history["loss"], modelname)
model.save('Out/' + modelname + 'Last.h5')