from collections import deque
import cv2
import gym
from gym_multi_car_racing.multi_car_racing import MultiCarRacing
import numpy as np
import torch
from pyglet.window import key
from utils.layers import DQN


def discrete_to_action(action: list):
    '''converts the discrete action (likely to be retained as an array of the Q-values of all actions)
      to be produced by the DQNs to the actual actions to be taken by the cars'''

    action_map = {0: [-1, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 0.8], 4: [0, 0, 0]}
    for i in range(len(action)):
        a[i] = action_map[int(action[i].argmax())]

# moves model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# Enter the path of the saved model
# Load the saved data
checkpoint = torch.load('tweaked_reward.pth', weights_only=False)

# Initialize the models
model_1 = DQN().to(device)
model_2 = DQN().to(device)
target_model_1 = DQN().to(device)
target_model_2 = DQN().to(device)

NUM_CARS = 2  # Supports key control of two cars, but can simulate as many as needed
NUM_ACTIONS = 5

env = MultiCarRacing(NUM_CARS)

observation_frames = deque(maxlen=4)  # stores only last 4 frames to account for temporal info
f = env.render()

isopen = True
stopped = False

a = np.zeros((NUM_CARS, 3))

model_1.layers.load_state_dict(checkpoint['model_1_state_dict'])
model_2.layers.load_state_dict(checkpoint['model_2_state_dict'])

def key_press(k, mod):
    global restart, stopped
    if k==0xff1b: stopped = True # Terminate on esc.
    if k==key.ENTER: restart = True # Restart on Enter.
    print(restart)

while isopen and not stopped:
    env.reset()
    for viewer in env.viewer:
        viewer.window.on_key_press = key_press

    total_reward = np.zeros(NUM_CARS)
    steps = 0
    restart = False

    while True:
        s, r, done, info = env.step(a)
        total_reward += r

        if steps % 200 == 0 or done:
            print("\nActions: " + str.join(" ", [f"Car {x}: "+str(a[x]) for x in range(NUM_CARS)]))
            print(f"Step {steps} Total_reward "+str(total_reward))

        f = np.array([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis] for x in s[:]]) / 255 # convert to grayscale

        observation_frames.append(f)
        total_reward += r

        if steps % 4 == 0 and steps > 0:

            state1 = np.concatenate([i[0] for i in observation_frames], axis=-1).reshape(1, 96, 96, 4).transpose((0, 3, 1, 2))
            state2 = np.concatenate([i[1] for i in observation_frames], axis=-1).reshape(1, 96, 96, 4).transpose((0, 3, 1, 2))

            state1 = torch.tensor(state1, dtype=torch.float32).to(device)
            state2 = torch.tensor(state2, dtype=torch.float32).to(device)

            decision = [model_1.forward(state1), model_2.forward(state2)]
            
            discrete_to_action(decision)

        steps += 1

        if stopped or done or restart or isopen == False:
            break

        isopen = env.render().all()
        
    env.close()

