# python related
import numpy as np
import random
from collections import deque

# training related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
# from torchinfo import summary

# gym related
import gym
import gym.wrappers as wrapper
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import cv2
import time
import wandb
import matplotlib
import matplotlib.pyplot as plt
import math

n_episode = 1000

# Environment and model hyperparameters
STATE_SIZE = [240, 256, 1]  # Dimensions of the game observation
ACTION_SIZE = 12  # Number of valid actions in the game
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 0.0001  # Learning rate
BATCH_SIZE = 32  # Batch size for training
MEMORY_SIZE = 20000  # Size of the replay memory buffer

LOG_FREQ = 100


class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class ReplayMemory_Per(object):
    # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity=MEMORY_SIZE, a=0.6, e=0.01):
        self.tree = SumTree(capacity)
        self.memory_size = capacity
        self.prio_max = 0.1
        self.a = a
        self.e = e

    def push(self, transition):
        p = (np.abs(self.prio_max) + self.e) ** self.a  # proportional priority
        self.tree.add(p, transition)

    def sample(self, batch_size):
        idxs = []
        priorities = []
        sample_datas = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            if not isinstance(data, tuple):
                print(idx, p, data, self.tree.write)
            idxs.append(idx)
            priorities.append(p)
            sample_datas.append(data)
        return idxs, priorities, sample_datas

    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries

class NoisyConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, sigma_init=0.017, bias=True, stride=1, padding=0, dilation=1, groups=1):
        super(NoisyConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_channels, in_channels, kernel_size, kernel_size), sigma_init))
        self.sigma_bias = nn.Parameter(torch.full((out_channels,), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.register_buffer("epsilon_bias", torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / (self.in_channels * self.kernel_size[0] ** 2))
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):

        # if self.training:
        weight = self.weight + self.sigma_weight * self.epsilon_weight.data.to(self.weight.device)
        bias = self.bias + self.sigma_bias * self.epsilon_bias.data.to(self.bias.device)
        # else:
        #     weight = self.weight
        #     bias = self.bias

        output = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return output

    def reset_noise(self):
        epsilon_weight = torch.randn(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        epsilon_bias = torch.randn(self.out_channels)
        self.epsilon_weight = nn.Parameter(epsilon_weight, requires_grad=False)
        self.epsilon_bias = nn.Parameter(epsilon_bias, requires_grad=False)

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        # if self.training:
        weight = self.weight + self.sigma_weight * self.epsilon_weight.data.to(self.weight.device)
        bias = self.bias + self.sigma_bias * self.epsilon_bias.data.to(self.bias.device)
        # else:
        #     weight = self.weight
        #     bias = self.bias

        output = F.linear(input, weight, bias)
        return output

    def reset_noise(self):
        epsilon_weight = torch.randn(self.out_features, self.in_features)
        epsilon_bias = torch.randn(self.out_features)
        self.epsilon_weight = nn.Parameter(epsilon_weight, requires_grad=False)
        self.epsilon_bias = nn.Parameter(epsilon_bias, requires_grad=False)

# Define the Double DQN model
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=4)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, stride=3)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=14, kernel_size=3, stride=2)
        self.fc1 = NoisyLinear(14 * 9 * 10, 256)
        self.value_stream = NoisyLinear(256, 1)
        self.advantage_stream = NoisyLinear(256, ACTION_SIZE)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # print(x.shape)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        value = self.value_stream(x)
        adv = self.advantage_stream(x)
        adv_average = torch.mean(adv, dim=1, keepdim=True)
        q_values = value + adv - adv_average
        return q_values

    def reset_noise(self):
        self.fc1.reset_noise()
        self.value_stream.reset_noise()
        self.advantage_stream.reset_noise()

    def save_mu_state_dict(self):
        state_dict = self.state_dict()
        mu_state_dict = {}
        for key, value in state_dict.items():
            if 'sigma' not in key and 'epsilon' not in key:
                mu_state_dict[key] = value
            else:
                mu_state_dict[key] = torch.zeros_like(value)
        return mu_state_dict
    
    def load_mu_state_dict(self, state_dict, strict=True):
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if 'sigma' not in key and 'epsilon' not in key:
                filtered_state_dict.update({key: value})
                print(value.type)
            else:
                filtered_state_dict.update({key: torch.empty(value.shape)})
        super().load_state_dict(filtered_state_dict, strict)

# Define the Double DQN agent
class DoubleDQNAgent:
    def __init__(self):
        # self.memory = deque(maxlen=MEMORY_SIZE)
        self.memory = ReplayMemory_Per(capacity=MEMORY_SIZE)
        self.model = DQN().to(device)
        self.target_model = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        # self.epsilon = EPSILON_START
        self.steps_counter = 0

        self.frames_counter = 0
        self.stacked_img = None
        self.stacked_img_buf = None
        self.prev_action = 0 # initialize as NOOP
        self.pick_action_flag = False

        self.debug_flag = True

    def init_target_model(self): # used only before training
        self.target_model = DQN().to(device)
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def choose_action(self, state):
        # grayscale the image
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = np.expand_dims(state, axis=2)

        if self.frames_counter != 12:
            
            # stack image
            if self.frames_counter == 0:
                self.stacked_img = state
            elif self.frames_counter % 4 == 0:
                self.stacked_img = np.concatenate((self.stacked_img, state), axis=2)

            # update member variables
            self.pick_action_flag = False

            # update frames counter
            self.frames_counter += 1

            return self.prev_action
        
        else: # self.frames_counter == 12
            
            # stack image
            self.stacked_img = np.concatenate((self.stacked_img, state), axis=2)
            self.stacked_img = np.int8(self.stacked_img)
            self.stacked_img = torch.from_numpy(self.stacked_img).float()
            self.stacked_img = self.stacked_img.permute(2, 0, 1)
            self.stacked_img = self.stacked_img.unsqueeze(0).to(device)
            
            # pick new action
            # if np.random.rand() <= self.epsilon:
            #     new_action = np.random.randint(0, 12)
            # else:
            q_values = self.model(self.stacked_img)
            new_action = q_values.max(1)[1].item()

            # update member variables
            self.stacked_img_buf = self.stacked_img.squeeze(0).to(torch.int8)
            self.stacked_img = None
            self.prev_action = new_action
            self.pick_action_flag = True

            # update frames counter
            self.frames_counter = 0

            return new_action

    def replay(self):
        if self.memory.size() < BATCH_SIZE:
            return

        if self.memory.size() == 20000 and self.debug_flag:
            self.debug_flag = False
            print("memory full")
        idxs, priorities, sample_datas = self.memory.sample(BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*sample_datas)

        # compute weights for loss update
        weights = np.power(np.array(priorities) + self.memory.e, -self.memory.a)
        weights /= weights.max()
        weights = torch.from_numpy(weights).float().to(device)

        states = torch.stack(states).float()
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).float().to(device)
        next_states = torch.stack(next_states).float()
        dones = torch.FloatTensor(dones).float().to(device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = (rewards + GAMMA * next_q_values * (1 - dones))
        loss = (weights * nn.MSELoss()(q_values, expected_q_values)).mean()
        
        wandb.log({"loss": loss.item()})
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # update PER
        td_errors = (q_values - expected_q_values).detach().squeeze().tolist()
        self.memory.update(idxs, td_errors)

        if self.steps_counter % 10000 == 0:
            print("copy!")
            self.target_model.load_state_dict(self.model.state_dict())

        self.steps_counter += 1
        self.model.reset_noise()
        self.target_model.reset_noise()

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        weights = self.model.state_dict()
        torch.save(weights, name)

# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    wandb.init(project="mario-with-dqn")#, mode="disabled")
    # Create the environment
    # env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0', stages=['1-1'])
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    # Create the agent
    agent = DoubleDQNAgent()
    agent.model.train()
    # agent.load("mario_model_noisy_per_step_20000.pt")
    agent.init_target_model()
    wandb.watch(agent.model, log_freq=LOG_FREQ)
    
    state_stack_temp = None
    action_temp = None
    reward_temp = None
    done_temp = None

    show_state = None
    show_next_state = None
    score_per_5_episode = 0
    # show_flag = True

    # Train the agent
    for episode in (range(n_episode)):
        state = env.reset()
        done = False
        score = 0
        first_action = True
        _4_frames_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            _4_frames_reward += reward

            if agent.pick_action_flag:
                if first_action:
                    
                    first_action = False

                    # just store
                    state_stack_temp = agent.stacked_img_buf
                    action_temp = action
                    reward_temp = _4_frames_reward
                    done_temp = done

                    _4_frames_reward = 0
                else:
                    # call remember() (to remember the last transition)
                    next_state_stack = agent.stacked_img_buf
                    action_temp = torch.tensor(action_temp).to(torch.int8)
                    reward_temp = torch.tensor(reward_temp).to(torch.int8)
                    done_temp = torch.tensor(done_temp).to(torch.int8)

                    agent.remember(state_stack_temp, \
                                    action_temp, \
                                    reward_temp, \
                                    next_state_stack, \
                                    done_temp)
                    
                    # for plotting
                    # if show_flag and np.random.rand() < 0.01:
                    #     show_flag = False
                    #     show_state = state_stack_temp.cpu().numpy()
                    #     show_next_state = next_state_stack.cpu().numpy()

                    # store
                    state_stack_temp = next_state_stack
                    action_temp = action
                    reward_temp = _4_frames_reward
                    done_temp = done


                    agent.replay()
                    _4_frames_reward = 0

            state = next_state
            score += reward
        print(f"Episode: {episode}, Score: {score}")

        
        # if agent.epsilon > EPSILON_END: # not used in noisy net
        #     agent.epsilon *= EPSILON_DECAY

        score_per_5_episode += score
        if episode % 5 == 0:
            wandb.log({"score per 5 epi": score_per_5_episode / 5})
            score_per_5_episode = 0

    # Save the trained model
    agent.save("noisy_per_step.pt")

    env.close()

    # matplotlib.use('TkAgg')
    # fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    # for i in range(2):
    #     for j in range(4):
    #         if i == 0:
    #             axes[i, j].imshow(show_state[j], cmap='gray')
    #         else:
    #             axes[i, j].imshow(show_next_state[j], cmap='gray')
    # plt.axis('off')  # Turn off axis
    # plt.tight_layout()
    # plt.show()
    # Close the environment