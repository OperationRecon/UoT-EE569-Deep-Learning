import numpy as np
import torch
import torch.optim as optim

# Initialize the DQN model
model = DQN()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example batch of experiences from the replay buffer
batch_size = 32
buffer = Replay_Buffer()
for _ in range(1000):
    buffer.add(np.random.randn(1, 1, 96, 96), np.random.randn(3), np.random.randn(1), np.random.randn(1, 1, 96, 96), np.random.randn(1))
states, actions, rewards, next_states, dones = buffer.sample(batch_size)

# Convert actions to long type for indexing
actions = actions.long()

# Forward pass to get the predicted Q-values
predicted_q_values = model(states)

# Compute the target Q-values
with torch.no_grad():
    next_q_values = model(next_states)
    max_next_q_values = next_q_values.max(dim=1)[0]
    target_q_values = rewards + (1 - dones) * 0.99 * max_next_q_values

# Gather the Q-values corresponding to the actions taken
predicted_q_values = predicted_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

# Compute the loss
loss_fn = nn.MSELoss()
loss = loss_fn(predicted_q_values, target_q_values)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")