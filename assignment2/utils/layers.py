import torch
import torch.nn as nn
import numpy as np
from utils.buffer import Replay_Buffer

# This file defines the model to be used in this assignment
# Architecture would be modifed directly here for now
# TODO: Add a way to modularly build architecture

cnn_architecture = [
    [4, 32, 16, 4, 8],    # Conv2d: in_channels=4, out_channels=32, kernel_size=16, stride=8
    [2,2],                # MaxPool2d: kernel_size=2, stride=2
    [32, 64, 8, 2, 4],    # Conv2d: in_channels=32, out_channels=64, kernel_size=8, stride=4  
    [2,2],                # MaxPool2d: kernel_size=2, stride=2
    [64, 128, 4, 1, 2],   # Conv2d: in_channels=32, out_channels=64, kernel_size=4, stride=2
    [2,2],                # MaxPool2d: kernel_size=2, stride=2
    [128, 128, 2, 1, 1],  # Conv2d: in_channels=32, out_channels=64, kernel_size=2, stride=1
    [2,2],                # MaxPool2d: kernel_size=2, stride=2
]

fcn_architecture = [128, 512, 128] # Fully connected layers

output_size = 5 # number of actions

# moves model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class DQN(nn.Module):
    '''DQN model. Architecture is hardcoded for now.'''
    def __init__(self, cnn_architecture=cnn_architecture, fcn_architecture=fcn_architecture, output_size=output_size, original=True, learning_rate=0.0002):
        super(DQN, self).__init__()
        
        layers = []
        
        # Add convolutional layers
        for params in cnn_architecture:
            if len(params) > 2:
                layers.append(nn.Conv2d(*params))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.MaxPool2d(*params))
        
        layers.append(nn.Flatten())
        
        # Add fully connected layers
        input_dim = fcn_architecture[0]
        for output_dim in fcn_architecture[1:]:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = output_dim
        
        # Add the output layer
        layers.append(nn.Linear(fcn_architecture[-1], output_size))

        self.layers = nn.Sequential(*layers)
        
        # properly initialize weights
        self.init_weights()

        # initialize the optimizer
        self.optimizer = torch.optim.Adam(self.layers.parameters(), lr=learning_rate)

        # initialize the loss function
        self.loss_function = nn.SmoothL1Loss()

    def init_weights(self):
        '''ensures proper weight initialization'''
        for m in self.layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        '''forward pass of the DQN model'''
        return self.layers(x)
        
    
    def learn(self, buffer, sample_size, batch_size, target_model, discount=0.98):
        '''Takes a sample from the buffer, 
        then performs the forward and backwards pass that sample one batch at a time.'''

        # Take a sample of experiences
        states, actions, rewards, next_states, dones = buffer.sample(sample_size)
        
        # Move tensors to the device
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        
        total_size = states.shape[0]

        for i in range(0, total_size, batch_size):
            # Take a ini-batch out of the sampled experiences

            s = states[i:min(i+batch_size, total_size)]
            a = actions[i:min(i+batch_size, total_size)]
            ns = next_states[i:min(i+batch_size, total_size)]
            r = rewards[i:min(i+batch_size, total_size)]
            d = dones[i:min(i+batch_size, total_size)]

            # predicted Q-values for current states
            q_values = self.forward(s)
            predicted_q_values = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

            # target Q-values for next states
            with torch.no_grad():
                next_q_values = target_model.forward(ns)
                max_next_q_values, _ = next_q_values.max(dim=1)
                target_q_values = r + (1 - d) * discount * max_next_q_values
            
            # loss
            loss = self.loss_function(predicted_q_values, target_q_values)

            # Backwards pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()

    
    def update_target(self, target_model):
        '''update the target model'''
        target_model.layers.load_state_dict(self.layers.state_dict())

    
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
