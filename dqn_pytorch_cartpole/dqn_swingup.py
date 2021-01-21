#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 12:15:13 2021

@author: dell
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import cartpole_swingup_envs

BATCH_SIZE = 32            
LR = 0.01                  
EPSILON = 0.9               
GAMMA = 0.9                
TARGET_REPLACE_ITER = 100  
MEMORY_CAPACITY = 2000     
EPISODE=2000                

#env = gym.make('CartPoleSwingUpDiscrete-v0')
env = gym.make('CartPoleSwingUpDiscrete-v0')
#env = env.unwrapped
#env = gym.wrappers.Monitor(env, './video/',video_callable=lambda episode_id: True,force = True)

N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")
torch.FloatTensor=torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor #如果有GPU和cuda，数据将转移到GPU执行
torch.LongTensor=torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
    
class DQN:
    def __init__(self):
        self.net,self.target_net=Net().to(device),Net().to(device)
        
        self.learn_step_counter=0
        self.memory_counter=0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:  
            actions_value = self.net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:   
            action = np.random.randint(0, N_ACTIONS)
            
        return action
        
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_s = torch.FloatTensor(batch_memory[:, :N_STATES])
        batch_a = torch.LongTensor(batch_memory[:, N_STATES:N_STATES+1].astype(int))
        batch_r = torch.FloatTensor(batch_memory[:, N_STATES+1:N_STATES+2])
        batch_s_ = torch.FloatTensor(batch_memory[:, -N_STATES:])

        
        q = self.net(batch_s).gather(1, batch_a)  # shape (batch, 1)
        q_target = self.target_net(batch_s_).detach()     # detach from graph, don't backpropagate
        y = batch_r + GAMMA * q_target.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
dqn = DQN()
   
plot_x_data,plot_y_data=[],[]
for i_episode in range(10000):
    s = env.reset()
    episode_reward = 0
    while True:
        env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # modify the reward
        #x, x_dot, theta, theta_dot = s_[0,1,]
        x, x_dot, theta_cos, theta_sin, theta_dot=s_
# =============================================================================
#         r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
#         r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
#         r = r1 + r2
# =============================================================================

        dqn.store_transition(s, a, r, s_)

        episode_reward += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Episode: ', i_episode,
                      '| Episode_reward: ', round(episode_reward, 2))

        if done:
            break
        s = s_
    plot_x_data.append(i_episode)
    plot_y_data.append(episode_reward)
    plt.plot(plot_x_data,plot_y_data)