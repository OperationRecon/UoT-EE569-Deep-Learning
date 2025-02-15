import numpy as np
import cv2
import torch
from collections import deque
from utils.reward_calculation import calculate_reward
from utils.policy import Epsilon_Greedy_Policy
from utils.layers import DQN
from utils.buffer import Replay_Buffer
import gym
from multi_car_racing.gym_multi_car_racing.multi_car_racing import MultiCarRacing
import imageio
from torch.utils.tensorboard import SummaryWriter

# moves model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

EPOCHS = 800

TARGET_UPDATE_FREQUENCY = 14

NUM_CARS = 2  # Supports key control of two cars, but can simulate as many as needed
NUM_ACTIONS = 5

SAMPLE_SIZE = 800
BATCH_SIZE = 16
BUFFER_SIZE = 12000

OBSERVATION_CHANNELS = 4

LEARNING_RATE = 0.0001

REWARD_DROP_TOLERANCE = -40 # Determines how far can the reward drop before an episode is stopeed prematurely

early_stopping_patience = 200  # Stop training if no improvement for this many epochs
learning_iterations = 1

POLICY = Epsilon_Greedy_Policy(epsilon=1, decay=0.994)

model_1 = DQN(learning_rate=LEARNING_RATE).to(device)
target_model_1 = DQN().to(device)
model_2 = DQN(learning_rate=LEARNING_RATE).to(device)
target_model_2 = DQN().to(device)

a = np.zeros((NUM_CARS, 3))

def discrete_to_action(action: list):
    '''converts the discrete action (likely to be retained as an array of the Q-values of all actions)
      to be produced by the DQNs to the actual actions to be taken by the cars'''

    action_map = {0: [-1, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 0.8], 4: [0, 0, 0]}
    for i in range(len(action)):
        a[i] = action_map[int(action[i].argmax())]

# Explore with random actions
def random_action():
    '''Takes random action with higher chance of it being forward movement to encourage track progress.'''
    x =  torch.rand(NUM_CARS, NUM_ACTIONS)  * torch.tensor([1,1,1.2,1,0.8])[torch.newaxis, :]
    return x

env = gym.make("MultiCarRacing-v0", num_agents=2, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False, verbose=False)

f = env.render()

isopen = True
stopped = False

replay_buffer_car1 = Replay_Buffer(capacity=BUFFER_SIZE)
replay_buffer_car2 = Replay_Buffer(capacity=BUFFER_SIZE)

observation_frames = deque(maxlen=OBSERVATION_CHANNELS)  # stores only last 4 frames to account for temporal info
prev_observation_frames = deque(maxlen=OBSERVATION_CHANNELS)

state1 = np.zeros((1, 4, 96, 96))

# Lists to store performance metrics
total_rewards = []

# Early stopping variables
best_reward = -float('inf')
patience_counter = 0

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir='assignment2/runs/experiment1')

epoch = 0

while epoch <= EPOCHS and not stopped:
    env.reset()

    total_reward = np.zeros(NUM_CARS)
    episode_reward = np.zeros(NUM_CARS)
    high_episode_reward = np.zeros_like(total_reward)
    steps = 0
    restart = False

    if epoch % max(EPOCHS // 100, 1) == 0:
        # Initialize video recording
        video_filename = f'imgs/a2/epoch{epoch}.mp4'
        video_writer = imageio.get_writer(video_filename, fps=30)
    
    while True:

        s, r, done, info = env.step(a)
        
        f = np.array([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis] for x in s[:]]) / 255 # convert to grayscale amd normalise

        observation_frames.append(f)

        r = calculate_reward(env, r)
        total_reward += r
        episode_reward += r       


        for i in range(total_reward.shape[0]):
            high_episode_reward[i] = max(high_episode_reward[i], episode_reward[i])

        if steps % 4 == 0 and steps > 0:
            
            # create the state from the kast four frames stacked on the channels axis
            state1 = np.concatenate([i[0] for i in observation_frames], axis=-1).reshape(1, 96, 96, 4).transpose((0, 3, 1, 2))
            state2 = np.concatenate([i[1] for i in observation_frames], axis=-1).reshape(1, 96, 96, 4).transpose((0, 3, 1, 2))

            state1 = torch.tensor(state1, dtype=torch.float32).to(device)
            state2 = torch.tensor(state2, dtype=torch.float32).to(device)
            
            # episode if reward starts dropping in order to not fill the replay buffer with useless experiences
            done =  False not in [episode_reward[i] - high_episode_reward[i] < REWARD_DROP_TOLERANCE for i in range(NUM_CARS)]

            if steps > 4:
                replay_buffer_car1.add(prev_state1.cpu().numpy(), torch.argmax(decision[0]), r[0], state1.cpu().numpy(), done or restart)

                replay_buffer_car2.add(prev_state2.cpu().numpy(), torch.argmax(decision[1]), r[1], state2.cpu().numpy(), done or restart)
            
            prev_state1 = state1.clone()
            prev_state2 = state2.clone()
            
            # Using e-greedy policy, if true then eploit, else explore
            if POLICY.select_action():
                decision = [model_1.forward(state1), model_2.forward(state2)]
                
            else:
                decision = random_action()
            
            discrete_to_action(decision)

        if  done:
            # record diagnostics and proceed with learning

            print(f'\r Learning...',end='', flush=True)
            for _ in range(learning_iterations):
                loss_1 = model_1.learn(replay_buffer_car1, sample_size=SAMPLE_SIZE, batch_size=BATCH_SIZE, target_model=target_model_1)
                loss_2 = model_2.learn(replay_buffer_car2, sample_size=SAMPLE_SIZE, batch_size=BATCH_SIZE, target_model=target_model_2)
                writer.add_scalar('Loss/model_1', loss_1, epoch)
                writer.add_scalar('Loss/model_2', loss_2, epoch)

        steps += 1

        if stopped or done:
            # output reward for monitoring

            total_rewards.append(total_reward.mean()/steps)
            writer.add_scalar('Reward/total_reward', total_reward.mean()/steps, epoch)
            print(f"Epoch {epoch} - Total Reward: {total_reward.mean()/steps}")
            break

        if epoch % max(EPOCHS // 100, 1) == 0:
            # Capture frame for recording

            frame = env.render(mode='rgb_array')
            frame = np.block([[frame[0]], [frame[1]]])
            video_writer.append_data(frame)

    epoch += 1

    video_writer.close()

    # Ensure that there is some improvment with the model
    if total_reward.mean() > best_reward:
        best_reward = total_reward.mean()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter > early_stopping_patience:
        # No improvement in the model after too long, abort training
        print("Early stopping triggered.")
        break

    if epoch % TARGET_UPDATE_FREQUENCY == 0 and epoch > 0:
        model_1.update_target(target_model_1)
        model_2.update_target(target_model_2)

    POLICY.update_epsilon()

    env.close()

# Record model parameters and metrics
final_data = {
    'model_1_state_dict': model_1.layers.state_dict(),
    'model_2_state_dict': model_2.layers.state_dict(),
    'target_model_1_state_dict': target_model_1.layers.state_dict(),
    'target_model_2_state_dict': target_model_2.layers.state_dict(),
    'total_reward': total_reward,
    'high_total_reward': high_episode_reward,
    'epoch': epoch
}

torch.save(final_data, 'final_epoch_data.pth')
print("Final epoch data saved.")


# Close the TensorBoard writer
writer.close()

