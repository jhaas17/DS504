# import tensorflow as tf
import numpy as np 
import keras

from GAN import *
from train import *
from train import *
from plotting import *
import keras.datasets.cifar10 as cifar
import keras.datasets.fashion_mnist as fmnist

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

print("Running on GPU: ", tf.test.is_gpu_available(cuda_only=True))

config = tf.ConfigProto( device_count = {'GPU': 1 },log_device_placement=False) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess) 

#DEFINE MODEL NAME FOR OUTPUT
print("Final")
modelname = 'Final' 

# Define parameters 
z_dim = 100
num_epoch = 100
batchsize = 32
lr = 0.0002

print("MODELNAME: " + modelname)

print("Noise dimensions: " + str(z_dim))
print("Batchsize: " + str(batchsize))
print("Learning Rate: " + str(lr))

###UNCOMMENT AND COMMENT THE MNIST PART TO USE FASHION MNIST DATASET

# # Fashion Mnist data

# (trainX, _), (_, _) = fmnist.load_data()
# trainX = trainX.reshape(60000, 28, 28, 1)
# trainX = trainX.astype('float32')

# #Normalize between -1 to 1 
# data = (trainX - 127.5) / 127.5


# MNIST DATASET

(trainX, _), (_, _) = load_data()
trainX = trainX.reshape(60000, 28, 28, 1)
trainX = trainX.astype('float32')

#Normalize between -1 to 1 
data = (trainX - 127.5) / 127.5

# CIFAR10
# (trainX, _), (_, _) = cifar.load_data()
# trainX = trainX.reshape(50000, 32, 32, 3)
# trainX = trainX.astype('float32')
# data = (trainX - 127.5) / 127.5

adam = keras.optimizers.Adam(lr=lr, beta_1=0.5)

g = create_generator(z_dim)
d = create_discriminator(opt=adam)
gan = create_gan(g, d, adam)

print("Generator: ")
print(g.summary())

print("Discriminator: ")
print(d.summary())

train(g, d, gan, data, modelname, batchsize, num_epoch, z_dim)