from collections import deque
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Replay_Buffer():
    def __init__(self, state_shape=[4,96,96], capacity=int(1e5)):
        self.capacity = capacity
        self.position = 0
        self.full = False
        
        # Preallocate memory for experiences
        self.states = torch.zeros((capacity, *state_shape), dtype=torch.float32)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
    
    def add(self, state, action, reward, next_state, done):
        '''Add a new experience to the buffer'''
        idx = self.position % self.capacity
        
        self.states[idx] = torch.from_numpy(state)
        self.actions[idx] = action.clone().view(-1)
        self.rewards[idx] = torch.tensor(reward, dtype=torch.float32).view(-1)
        self.next_states[idx] = torch.from_numpy(next_state)
        self.dones[idx] = torch.tensor(done, dtype=torch.float32).view(-1)
        
        self.position += 1
        if self.position >= self.capacity:
            self.full = True
    
    def sample(self, batch_size):
        '''Take a sample of experiences from the buffer.'''
        max_index = self.capacity if self.full else self.position
        
        # Adjust batch size if buffer has fewer samples than batch_size
        if max_index < batch_size:
            batch_size = max_index
        
        idx = np.random.choice(max_index, batch_size, replace=False)
        states = self.states[idx].to(device)
        actions = self.actions[idx].to(device)
        rewards = self.rewards[idx].to(device)
        next_states = self.next_states[idx].to(device)
        dones = self.dones[idx].to(device)
        
        return states, actions.squeeze(-1), rewards.squeeze(-1), next_states, dones.squeeze(-1)

if __name__ == "__main__":
    buffer = Replay_Buffer()
    for i in range(1000):
        buffer.add(np.random.randn(1,1,96,96), np.random.randn(5), np.random.randn(1), np.random.randn(1,1,96,96), np.random.randn(1))
    print(buffer.sample(10))

