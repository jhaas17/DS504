import numpy as np 
import matplotlib.pyplot as plt
from utils import *

def plot_losses(losses, modelname):
    #Plot Loss plot for generator and discriminator (Maximizing Generator, minimizing discriminator)
    plt.plot(losses)
    plt.title('Loss Plot')
    plt.savefig('Out/' + modelname + 'Loss' + '.png')
    plt.clf()

def plot_accuracies(accuracies, epoch, modelname):
    #Plot Loss plot for generator and discriminator (Maximizing Generator, minimizing discriminator)
    plt.scatter(range(epoch),accuracies,label='Accuracies Plot')
    plt.title('Accuracies Plot')
    plt.legend()
    plt.savefig('Out/' + modelname + 'Accuracies' + str(epoch) + '.png')
    plt.clf()
