import pandas as pd
import datetime as dt
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
# Preprocess data --> Day of week, time with passenger, time without passenger
#RNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


num_drivers = 5
num_epochs = 2
batch_size = 1
learning_rate = 0.0015

rnn_input_size = 3
rnn_hidden_size = 64
rnn_num_layers=2

fc_input_size = 9

class RNN(nn.Module):
    def __init__(self, rnn_input_size, rnn_hidden_size, rnn_num_layers, num_drivers,fc_input_size):
        super(RNN, self).__init__()
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_empty = nn.RNN(rnn_input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
        self.rnn_occupied = nn.RNN(rnn_input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
        self.fce = nn.Linear(rnn_hidden_size,2)
        self.fco = nn.Linear(rnn_hidden_size,2)
        self.fc1 = nn.Linear(fc_input_size,64)
        self.fc2 = nn.Linear(64,num_drivers)
        # -> x needs to be: (batch_size, seq, input_size)
        
    def forward(self, x, y, z):
        # Set initial hidden states (and cell states for LSTM)
        # h0 = torch.zeros(self.rnn_num_layers, x.size(0), self.rnn_hidden_size).double().to(device)
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        x, _ = self.rnn_empty(x)  
        y, _ = self.rnn_occupied(y)
        
        
        x = x[:, -1, :]
        y = y[:, -1, :]
         
        x = self.fce(x)
        y = self.fco(y)

        # print(x[0])
        # print(y)
        # print(z)
    

        z = torch.cat((x[0],y[0],z),0)

        z = F.relu(self.fc1(z))
        z = self.fc2(z)

        return z



data = pd.read_pickle('TaxiSubsFinal.pkl').drop(['date'],axis=1)
data = data[data['num_trips']!=0]

training, testing = train_test_split(data, test_size=0.2)

train_label = np.array(training['plate'])
training.drop(['plate'],axis=1)
training = training.to_numpy()

testing_label = np.array(testing['plate'])
testing.drop(['plate'],axis=1)
testing = testing.to_numpy()

model = RNN(rnn_input_size, rnn_hidden_size, rnn_num_layers, num_drivers,fc_input_size).to(device)
model.double()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
print(model)

#Empty trajectories = 1; occupied trajectories=5 

losses = []
print("Begin Training")
for epoch in range(num_epochs):
    print("Epoch number: " + str(epoch))
    for i, x in enumerate(training):
        empty_traj = torch.tensor(x[1]).reshape(1,x[1].shape[0],3).double().to(device)
        occupied_traj = torch.tensor(x[5]).reshape(1,x[5].shape[0],x[5].shape[1]).double().to(device)
        features = torch.from_numpy(np.array(list(x[[0,2,3,4,6]]),dtype=np.double)).to(device)
        
        out = model(empty_traj.double(), occupied_traj.double(), features.double())
        out = out.reshape(1,out.size()[0])
        
        loss = criterion(out,torch.LongTensor(np.array([train_label[i]])))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%10 == 0:
            print("Loss: " + str(loss.item()))
    
torch.save(model, 'TaxiModel2.pth.tar')

print("Begin Testing")
model = torch.load('TaxiModel2.pth.tar')

y_pred = []
for i,x in enumerate(testing):
    empty_traj = torch.tensor(x[1]).reshape(1,x[1].shape[0],3).double().to(device)
    occupied_traj = torch.tensor(x[5]).reshape(1,x[5].shape[0],x[5].shape[1]).double().to(device)
    features = torch.from_numpy(np.array(list(x[[0,2,3,4,6]]),dtype=np.double))
    
    out = model(empty_traj.double(), occupied_traj.double(), features.double())
    y = torch.argmax(out).item()

    y_pred.append(y)

print("Testing Accuracy: ")
print(sum(1 for x,y in zip(y_pred,testing_label) if x == y) / len(y_pred))