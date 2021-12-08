import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D


##BEST GENERATOR - with -1 to 1 input!
def create_generator(z_dim):
	# Leaky ReLU with 2 UpSampling 2d
	g = Sequential()
	g.add(Dense(7*7*256, input_dim=z_dim))
	g.add(LeakyReLU())
	g.add(Reshape((7,7,256)))
	g.add(UpSampling2D(input_shape=(7,7,256)))
	g.add(LeakyReLU())
	g.add(UpSampling2D())
	g.add(LeakyReLU())
	g.add(Conv2D(1, 7, activation='tanh', padding='same'))
	return g

def create_discriminator(opt, input_shape=(28,28,1)):
	# Leaky ReLU with 2 Conv Layers (32,64) Kernel 5, stride 2x2
	d = Sequential()
	d.add(Conv2D(32, 5, strides=2, padding='same', input_shape=input_shape))
	d.add(LeakyReLU())
	d.add(Dropout(0.3))
	d.add(Conv2D(64, 5, strides=2, padding='same'))
	d.add(LeakyReLU())
	d.add(Dropout(0.3))
	d.add(Flatten())
	d.add(Dense(1, activation='sigmoid'))
	d.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

	return d

def create_gan(g, d, opt):
	d.trainable = False
	gan = Sequential()
	gan.add(g)
	gan.add(d)
	gan.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return gan
