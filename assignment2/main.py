import numpy as np
import cv2
import torch
from pyglet.window import key
from collections import deque
from utils.policy import Epsilon_Greedy_Policy
from utils.layers import DQN
from utils.buffer import Replay_Buffer
from multi_car_racing.gym_multi_car_racing.multi_car_racing import MultiCarRacing


NUM_CARS = 2  # Supports key control of two cars, but can simulate as many as needed
USE_KEYBOARD = False  # Set to False to use random actions instead of keyboard

# Specify key controls for cars
CAR_CONTROL_KEYS = [[key.LEFT, key.RIGHT, key.UP, key.DOWN],
                    [key.A, key.D, key.W, key.S]]

POLICY = Epsilon_Greedy_Policy(epsilon=0.8, decay=0.99998)

model_1 = DQN()
model_2 = DQN()

a = np.zeros((NUM_CARS,3))
def key_press(k, mod):
    global restart, stopped, CAR_CONTROL_KEYS
    if k==key.ESCAPE: stopped = True # Terminate on esc.
    if k==key.RETURN or k==key.ENTER: restart = True # Restart on Enter.
    

    if USE_KEYBOARD:
        # Iterate through cars and assign them control keys (mod num controllers)
        for i in range(min(len(CAR_CONTROL_KEYS), NUM_CARS)):
            if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][0]:  a[i][0] = -1.0
            if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][1]: a[i][0] = +1.0
            if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][2]:    a[i][1] = +1.0
            if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][3]:  a[i][2] = +0.8   # set 1.0 for wheels to block to zero rotation

def discrete_to_action(action: list):
    '''converts the discrete action (likely to be retained as an array of the Q-values of all actions)
      to be produced by the DQNs to the actual actions to be taken by the cars'''

    action_map = {0: [-1,0,0], 1:[1,0,0], 2:[0,1,0], 3:[0,0,0.8], 4:[0,0,0]}
    for i in range(len(action)):
        a[i] = action_map[int(action[i].argmax())]
        

def key_release(k, mod):
    global CAR_CONTROL_KEYS

    if USE_KEYBOARD:
        # Iterate through cars and assign them control keys (mod num controllers)
        for i in range(min(len(CAR_CONTROL_KEYS), NUM_CARS)):
            if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][0]  and a[i][0]==-1.0: a[i][0] = 0
            if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][1] and a[i][0]==+1.0: a[i][0] = 0
            if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][2]:    a[i][1] = 0
            if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][3]:  a[i][2] = 0

# Explore with random actions
def random_action():
    return np.random.randn(*a.shape)

env = MultiCarRacing(NUM_CARS)
env.render()

for viewer in env.viewer:
    # key_press and key_release function are not conditional by USE_KEYBOARD because they contain other functionality
    viewer.window.on_key_press = key_press
    viewer.window.on_key_release = key_release

record_video = False
if record_video:
    from gym.wrappers.monitor import Monitor
    env = Monitor(env, '/tmp/video-test', force=True)
isopen = True
stopped = False
replay_buffer_car1 = Replay_Buffer(10000)
replay_buffer_car2 = Replay_Buffer(10000)
observation_frames = deque(maxlen=4) # sotres only last 4 frames to account for temporal info
prev_observation_frames = deque(maxlen=4)
prev_action = None
prev_reward = None
prev_done = None
state1 = np.random.rand(1,4,96,96)
while isopen and not stopped:
    env.reset()
    total_reward = np.zeros(NUM_CARS)
    steps = 0
    restart = False
    while True:
        if steps % 4 == 0:
            s, r, done, info = env.step(a)

            f = np.array([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis] for x in s[:]]) # convert to grayscale

            observation_frames.append(f)

            total_reward += r


        if steps % 16 == 0 and steps > 0:
            state1 = np.concatenate([i[0] for i in observation_frames], axis=-1).reshape(1,96,96,4).transpose((0,3,1,2))
            state2 = np.concatenate([i[1] for i in observation_frames], axis=-1).reshape(1,96,96,4).transpose((0,3,1,2))
            
            if steps > 16:
                replay_buffer_car1.add(prev_state1, prev_action[0], prev_reward[0], state1, prev_done)
                replay_buffer_car2.add(prev_state2, prev_action[1], prev_reward[1], state2, prev_done)
            
            prev_state1 = state1[:]
            prev_state2 = state2[:]

            prev_action = a
            prev_done = done
            prev_reward = r

            if not USE_KEYBOARD:
                if POLICY.select_action():
                    discrete_to_action(
                            [model_1.forward(torch.tensor(state1, dtype=torch.float32)), model_2.forward(torch.tensor(state2, dtype=torch.float32))]
                             )
                else:
                    discrete_to_action(random_action())
                    

            
        
        if steps % 200 == 0 or done:
            
            # cv2.imshow("frame", f[0])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            print("\nActions: " + str.join(" ", [f"Car {x}: "+str(a[x]) for x in range(NUM_CARS)]))
            print(f"Step {steps} Total_reward {total_reward} state {f.shape} temporal {state1[:,:,:,:,].shape}")
            
            for i in range(4):
                cv2.imshow("frame", state1.transpose((1,2,3,0))[i])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            #import matplotlib.pyplot as plt
            #plt.imshow(s)
            #plt.savefig("test.jpeg")
        steps += 1
        isopen = env.render().all()
        if stopped or done or restart or isopen == False:
            break
env.close()
