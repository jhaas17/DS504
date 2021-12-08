import tensorflow as tf
import keras
import numpy as np 

from utils import *
from plotting import *


# Training
def train(g, d, gan, data, modelname, batchsize=32, num_epochs=100, z_dim=100):
    d_losses = []
    g_losses = []

    batchCount = int(data.shape[0]/batchsize)
    n = batchsize
    
    for i in range(num_epochs):
        for j in range(batchCount):  
        
            # Create a batch by drawing random index numbers from the training set
            X_real = get_real_samples(data, n)
            
            # Generate the images from the generator and random noise
            X_fake = get_fake_samples(g, z_dim, n)

            # Create labels
            y_real = np.random.uniform(low=0.75, high=1.2, size=(n,1))
            y_fake = np.random.uniform(low=0.0, high=0.4, size=(n,1))

            # y_real = np.ones((n,1))
            # y_fake = np.zeros((n,1))

            # Combine fake and real samples and label
            X = np.concatenate((X_real, X_fake), axis=0)
            y = np.concatenate((y_real, y_fake), axis=0)

            # Train discriminator on generated images

            # Switch between using just real/fake in discriminator training, or both 
            # if j%2==0: 
            #     d_loss,_ = d.train_on_batch(X_real,y_real)
            # else:
            #     d_loss,_ = d.train_on_batch(X_fake,y_fake)

            d_loss, _ = d.train_on_batch(X,y)

            # Generate random noise samples to train generator 
            noise_samples = get_noise_samples(z_dim, batchsize)
            #Create real labels for the fake noise samples for generator training
            y_hat = np.ones((batchsize,1))

            # y_hat = np.random.uniform(low=0.75, high=1.15, size=(n,1))

            # Train generator
            g_loss, g_acc = gan.train_on_batch(noise_samples, y_hat)

        d_losses.append(d_loss)
        g_losses.append(g_loss)
          
        print('epoch>%d, d=%.3f, g=%.3f, g_acc=%.3f' % (i+1, d_loss, g_loss, g_acc))
        

        if (i%10==0 or (i+1)==100):
            test(i, g, d, data, z_dim)

            plot_samples(g, z_dim, i+1, modelname)
            plot_losses(d_losses, g_losses, i+1, modelname)

            save_model(g, i, modelname)