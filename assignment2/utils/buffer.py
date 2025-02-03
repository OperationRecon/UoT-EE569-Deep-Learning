from collections import deque

import numpy as np
import torch

class Replay_Buffer():
    def __init__(self, capacity=int(10e4)):
        self.buffer = deque(maxlen= capacity)
        self.capacity = capacity
    
    def add(self, state, action, reward, next_state, done):
        '''add a new experience to the buffer'''
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        '''sample a batch of experiences from the buffer'''
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        experiences = [self.buffer[i] for i in idx]
        
        # Convert experiences to PyTorch tensors
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        states = states.view(batch_size, *states.shape[2:])
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        next_states = next_states.view(batch_size, *next_states.shape[2:])
        dones = torch.tensor(dones, dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    
if __name__ == "__main__":
    buffer = Replay_Buffer()
    for i in range(1000):
        buffer.add(np.random.randn(1,1,96,96), np.random.randn(5), np.random.randn(1), np.random.randn(1,1,96,96), np.random.randn(1))
    print(buffer.sample(10))

