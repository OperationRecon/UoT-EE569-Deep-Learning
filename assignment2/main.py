import numpy as np
import cv2
from pyglet.window import key
from collections import deque

from utils.buffer import Replay_Buffer
from multi_car_racing.gym_multi_car_racing.multi_car_racing import MultiCarRacing
NUM_CARS = 2  # Supports key control of two cars, but can simulate as many as needed
USE_KEYBOARD = True  # Set to False to use random actions instead of keyboard

# Specify key controls for cars
CAR_CONTROL_KEYS = [[key.LEFT, key.RIGHT, key.UP, key.DOWN],
                    [key.A, key.D, key.W, key.S]]

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

def discrete_to_action(action):
    '''converts the discrete action (likely to be retained as an array of the Q-values of all actions)
      to be produced by the DQNs to the actual actions to be taken by the cars'''

    action_map = {0: [-1,0,0], 1:[1,0,0], 2:[0,1,0], 3:[0,0,0.8], 4:[0,0,0]}
    for i in range(action.shape[0]):
        a[i] = action_map[np.argmax(action[i])]
        

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
    return np.clip(np.random.randn(*a.shape),-1,1)

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
is_next_state = False
replay_buffer_car1 = Replay_Buffer(10000)
replay_buffer_car2 = Replay_Buffer(10000)
observation_frames = deque(maxlen=4) # sotres only last 4 frames to account for temporal info

while isopen and not stopped:
    env.reset()
    total_reward = np.zeros(NUM_CARS)
    steps = 0
    restart = False
    while True:
        if not USE_KEYBOARD:
            a = random_action()
        
        if steps % 4 == 0:
            s, r, done, info = env.step(a)

            f = np.array([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis] for x in s[:]]) # convert to grayscale

            observation_frames.append(f)

            total_reward += r
        
        if steps % 16 == 0 and steps > 0:
            is_next_state = True
            
        if is_next_state:
            is_next_state = False

            s, r, done, info = env.step(a)

            f = np.array([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis] for x in s[:]]) # convert to grayscale

            replay_buffer_car1.add(np.array(observation_frames[0]), a[0], r[0], f[0], done)
            replay_buffer_car2.add(np.array(observation_frames[1]), a[1], r[1], f[1], done)

        
        
        if steps % 200 == 0 or done:
            
            cv2.imshow("frame", f[0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print("\nActions: " + str.join(" ", [f"Car {x}: "+str(a[x]) for x in range(NUM_CARS)]))
            print(f"Step {steps} Total_reward {total_reward} state {f.shape}")
            #import matplotlib.pyplot as plt
            #plt.imshow(s)
            #plt.savefig("test.jpeg")
        steps += 1
        isopen = env.render().all()
        if stopped or done or restart or isopen == False:
            break
env.close()
