import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.buffer import Replay_Buffer

# This file defines the model to be used in this assignment
# Architecture would be modifed directly here for now
# TODO: Add a way to modularly build architecture

cnn_architecture = [[4,16,8,4],[16,32,4,2],[2,2],[32,64,4,1],[64,64,2,1,],] # entries with 2 values are for pooling layers, other for conv layers
fcn_architecture = [64, 512] #in the middle are the hidden layer sizes, 64 is the input size
output_size = 5 # number of actions
class DQN(nn.Module):
    '''DQN model. Architecture is hardcoded for now.'''
    def __init__(self, cnn_architecture=cnn_architecture, fcn_architecture=fcn_architecture, output_size=output_size, original=True, learning_rate=0.001):
        super(DQN, self).__init__()
        
        self.layers = nn.Sequential()
        
        # add convulotional layers
        for i in cnn_architecture:
            if len(i) > 2:
                self.layers.append(module=nn.Sequential(nn.Conv2d(*i), nn.ReLU()))
            else:
                self.layers.append(module=nn.MaxPool2d(*i))
        
        self.layers.append(module=nn.Flatten()) # flatten the output of the conv layers

        # add fully connected layers
        for i in range(len(fcn_architecture) - 1):
            self.layers.append(module=nn.Sequential(nn.Linear(fcn_architecture[i],fcn_architecture[i+1]), nn.ReLU()))
        
        # add the output layer
        self.layers.append(module=nn.Linear(fcn_architecture[-1], output_size))
        
        # initialize the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # initialize the loss function
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        '''forward pass of the DQN model'''
        x = self.layers(x)
        return x
    
    def learn(self, buffer, batch_size, target_model, discount=0.99):
        '''learn from the buffer'''
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        actions = actions.long()
        
        # Forward pass to get the predicted Q-values
        predicted_q_values = self.layers(states)
        predicted_q_values = predicted_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # calculate target values
        with torch.no_grad():
            next_q_values = target_model(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards.view(-1) + (1 - dones.view(-1)) * discount * max_next_q_values
        

        # calculate loss
        loss = self.loss_function(predicted_q_values, target_q_values)
        
        # backpropogate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self, target_model):
        '''update the target model'''
        target_model.load_state_dict(self.state_dict())

    
if __name__ == "__main__":
# Some testing to make sure things work
    model = DQN()
    target_model = DQN()
    buffer = Replay_Buffer()
    x = torch.tensor(np.random.randn(1,4,96,96), dtype=torch.float32)
    x = model.forward(x)

    print(model,x.shape)
    model.update_target(target_model)

    for i in range(1000):
        buffer.add(np.random.randn(1,4,96,96), np.random.randint(0,5), np.random.randn(1), np.random.randn(1,4,96,96), np.random.randn(1))

    print(buffer.sample(10))
    for i in range(10):
        model.learn(buffer, 10, target_model)
    
    model.update_target(target_model)

    def compare_state_dicts(model, target_model):
        model_state_dict = model.state_dict()
        target_model_state_dict = target_model.state_dict()

        for key in model_state_dict:
            if not torch.equal(model_state_dict[key], target_model_state_dict[key]):
                return False
        return True

    # Example Usage
    are_equal = compare_state_dicts(model, target_model)
    print("Are the state dictionaries equal?", are_equal)
