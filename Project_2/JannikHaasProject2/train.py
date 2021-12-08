from model import NeuralNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler

import pandas as pd
import datetime as dt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def weights_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform(layer.weight.data)
        layer.bias.data.fill_(0)


def test(test_errors, model, testing):

    y_pred = []
    
    for i,x in enumerate(testing):
        features = torch.from_numpy(x).to(device)
        model.eval()
        out = model(features)
        y = torch.argmax(out).item()

        y_pred.append(y)

    test_accuracy.append(sum(1 for x,y in zip(y_pred,testing_label) if x == y) / len(y_pred))
   
    return test_errors
desc = "model description"
modelname = "model name"
num_epochs = 10000
learning_rate = 0.00015

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
torch.manual_seed(2020)

data = pd.read_pickle('TaxiStats.pkl').drop(['date'],axis=1)
data = data[data['num_trips']!=0]



data = shuffle(data)

training, testing = train_test_split(data, test_size=0.2, random_state=2021)

train_label = np.array(training['plate'])
training = training.drop(['plate'],axis=1)
training = StandardScaler().fit_transform(training)

testing_label = np.array(testing['plate'])
testing = testing.drop(['plate'],axis=1)
testing = StandardScaler().fit_transform(testing)

model = NeuralNet().to(device)
model.double()
model.apply(weights_init)
print(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

losses = []
lossesavg = []
test_accuracy = []
iters = []
train_accuracy = []
train_accuracies = []
print(desc)
print(modelname)
print('Epochs: ' + str(num_epochs))
print("Begin Training")
for epoch in range(num_epochs):
    print("Epoch number: " + str(epoch))
    if (epoch < 1000 and epoch%50 ==0) or epoch%1000 == 0:
        torch.save(model, modelname + str(epoch) + 'pth.tar')
        plt.scatter(range(len(lossesavg)),lossesavg)
        plt.savefig('Loss' + modelname + str(epoch) + '.png')
        plt.clf()

        plt.scatter(iters,test_accuracy,label='testing')
        plt.scatter(iters,train_accuracy,label='training')
        plt.legend()
        plt.savefig("Accuracies" + modelname + str(epoch) + ".png")
        plt.clf()
    for i, x in enumerate(training):
        model.train()
        
        features = torch.from_numpy(x).to(device)

        out = model(features)
        out = out.reshape(1,out.size()[0])
        loss = criterion(out,torch.LongTensor(np.array([train_label[i]])).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        train_accuracies.append(1 if torch.argmax(out).item() == train_label[i] else 0)

        if len(losses)%10000 == 0:
            print("Loss avg: " + str(sum(losses[-100:])/100))
            lossesavg.append(sum(losses[-100:])/100)
            test_accuracy = test(test_accuracy, model, testing)
            iters.append(len(losses))
            train_accuracy.append(sum(train_accuracies[-100:])/100)
            

torch.save(model, modelname + str(epoch) + 'pth.tar')
plt.scatter(range(len(lossesavg)),lossesavg)
plt.savefig('Loss' + modelname + str(epoch) + '.png')
plt.clf()

plt.scatter(iters,test_accuracy,label='testing')
plt.scatter(iters,train_accuracy,label='training')
plt.legend()
plt.savefig("Accuracies" + modelname + str(epoch) + ".png")
plt.clf()
