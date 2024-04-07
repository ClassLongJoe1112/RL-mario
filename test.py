# %%
# !pip install gym_super_mario_bros
# !pip install gym==0.23.1

# %%
from nes_py.wrappers import JoypadSpace
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym.wrappers as wrapper
import matplotlib.pyplot as plt
import cv2
import numpy as np
from train import DoubleDQNAgent
import torch

if __name__ == '__main__':
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    state = env.reset()

    agent = DoubleDQNAgent()
    agent.load("noisy_per_step.pt")
    agent.model.eval()
    done = False
    total_reward = 0
    for step in range(5000):
        if done:
            state = env.reset()
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            state = np.expand_dims(state, axis=2)
            print(total_reward)
            total_reward = 0
        action = agent.choose_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()

    env.close()
