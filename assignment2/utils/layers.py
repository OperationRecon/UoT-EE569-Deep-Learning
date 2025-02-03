import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# This file defines the CNN model to be used in this assignment
# Architecture would be modifed directly here for now
# TODO: Add a way to modularly build architecture

cnn_architecture = [[4,16,8,4],[16,32,4,2],[2,2],[32,64,4,1],[64,64,2,1,],] # entries with 2 values are for pooling layers, other for conv layers
fcn_architecture = [64, 512] #in the middle are the hidden layer sizes, 64 is the input size
output_size = 5 # number of actions

class CNN(nn.Module):
    '''CNN model, current archtecture is hardcoded.'''
    def __init__(self, architecture=cnn_architecture):
        super(CNN, self).__init__()
        self.layers = [nn.Conv2d(*i) if len(i) > 2 else nn.MaxPool2d(*i)  for i in architecture]

    def forward(self, x):
        '''forward pass of the CNN model. automatically flattens the output of the last layer.'''
        for layer in self.layers:
            x = layer(x)
            if type(layer) == nn.Conv2d:
                x = F.relu(x)
        x = x.view(x.size(0), -1)
        return x

if __name__ == "__main__":
    model = CNN()
    x = torch.tensor(np.random.randn(1,1,96,96), dtype=torch.float32)
    x = model.forward(x)
    print(model.layers,x.shape)

class FCN(nn.Module):
    '''Fully connected layers, netwrk, currently also hardcoded'''
    def __init__(self, architecture=fcn_architecture,activation=F.relu):
        super(FCN, self).__init__()
        self.layers = [nn.Linear(architecture[i], architecture[i+1]) for i in range(len(architecture)-1)]
        self.activation = activation
    
    def forward(self, x):
        '''forward pass of the FCN model'''
        for layer in self.layers:
            x = self.activation(layer(x))
        return x
    
class Out_Layer(nn.Module):
    '''Output layer for the DQN model takes the input of the previous layer, converts it to the output size
    and calculates the Q-values for the actions.'''
    def __init__(self, input_size, output_size, loss=nn.MSELoss()):
        super(Out_Layer, self).__init__()
        self.layers = [nn.Linear(input_size, output_size)]
        
    def forward(self, x):
        '''forward pass of the output layer'''
        for layer in self.layers:
            x = layer(x)
        return x

class DQN(nn.Module):
    '''DQN model, combines the CNN, FCN and Out_Layer to form the complete model.'''
    def __init__(self, cnn_architecture=cnn_architecture, fcn_architecture=fcn_architecture, output_size=output_size):
        super(DQN, self).__init__()
        self.cnn = CNN(cnn_architecture)
        self.fcn = FCN(fcn_architecture)
        self.out_layer = Out_Layer(fcn_architecture[-1], output_size)
        self.network = nn.Sequential(self.cnn, self.fcn, self.out_layer)
        
    def forward(self, x):
        '''forward pass of the DQN model'''
        x = self.network.forward(x)
        return x
    
if __name__ == "__main__":
    model = DQN()
    x = torch.tensor(np.random.randn(1,1,96,96), dtype=torch.float32)
    x = model.forward(x)
    print(model.network,x.shape)