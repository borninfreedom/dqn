#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 22:24:49 2021

@author: dell
"""

import gym
import numpy as np
import cartpole_swingup_envs

from stable_baselines3 import DQN,PPO
from stable_baselines3.dqn import MlpPolicy

env = gym.make('CartPoleSwingUpDiscrete-v0')

#model = DQN(MlpPolicy, env, verbose=1)
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000000, log_interval=4)
model.save("dqn_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

