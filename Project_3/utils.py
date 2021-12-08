from keras.datasets.mnist import load_data
import numpy as np
import keras


def get_real_samples(data, n):
    sample_indexes = np.random.randint(0,data.shape[0],n)
    X_real = data[sample_indexes]
    y_real = np.ones((n,1))
    return X_real

def get_noise_samples(z_dim, n):
    X_fake_noise = np.random.normal(size=[n, z_dim])
    return X_fake_noise

def get_fake_samples(g, z_dim, n):
    X_fake_noise = get_noise_samples(z_dim, n)
    X_fake = g.predict(X_fake_noise)
    return X_fake

def test(epoch, g, d, data, z_dim, n=100):
    # Create a batch by drawing random index numbers from the training set
    X_real = get_real_samples(data, n)
    y_real = np.ones((n,1))
    X_fake = get_fake_samples(g, z_dim, n)
    y_fake = np.zeros((n,1))

    X = np.concatenate((X_real, X_fake), axis=0)
    y = np.concatenate((y_real, y_fake), axis=0)

    # test descriminator on the samples samples
    _, acc= d.evaluate(X, y, verbose=0)
    # summarize discriminator performance
    print('Testing accuracy ==  %.0f%%' % (acc*100))
    return 
    
def save_model(g, epoch, modelname):    
    # serialize model to JSON
    model_json = g.to_json()
    with open("generator.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    g.save_weights(modelname + "generator" + "-" + str(epoch) + ".h5")

