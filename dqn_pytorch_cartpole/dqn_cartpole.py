import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import random
from gym import wrappers
import torch.optim as optim
import math
import matplotlib.pyplot as plt

Env_Name='CartPole-v0'

Batch_Size=32
Replay_Memory_Size=10000
Target_Network_Update_Frequency=10000
Discount_Factor=0.99
Learning_Rate=0.00025
Initial_Exploration=1
Final_Exploration=0.1
Exploration_Decay=200
Episodes=500
Hidden_Layer_Size=256


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=",device)

env=gym.make(Env_Name)
#env=wrappers.Monitor(env,'./tmp/cartpole-v1-1',force=True)
#observation=env.reset()

Input_Shape=env.observation_space.shape[0]
Action_Shape=env.action_space.n
#print('Input Shape=',Input_Shape)
#print('Action Shape=',Action_Shape)


class ReplayMemory:
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory=[]
    def push(self,transition):
        self.memory.append(transition)
        if len(self.memory)>self.capacity:
            del self.memory[0]
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self,input_shape=4,hidden_layer_shape=Hidden_Layer_Size,action_shape=2):
        super(DQN,self).__init__()
        self.l1=nn.Linear(input_shape,hidden_layer_shape)
        self.l2=nn.Linear(hidden_layer_shape,action_shape)
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=self.l2(x)
        return x


model=DQN(input_shape=Input_Shape,hidden_layer_shape=Hidden_Layer_Size,action_shape=Action_Shape).to(device)


memory=ReplayMemory(Replay_Memory_Size)
optimizer=optim.Adam(model.parameters(),Learning_Rate)
steps_done=0
episode_durations=[]

def select_action(state):
    global steps_done
    sample=random.random()
    eps_threshold=Final_Exploration+(Initial_Exploration-Final_Exploration)*math.exp(-1.*steps_done/
                                                                                     Exploration_Decay)
    steps_done+=1
    if sample>eps_threshold:
        return  model(torch.tensor(state,dtype=torch.float32).to(device)).detach().max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(2)]]).to(device)

def run_episode(e,env):
    state=env.reset()
    steps=0
    while True:
        env.render()
        action=select_action(torch.tensor([state],dtype=torch.float32))
        next_state,reward,done,_=env.step(action[0,0].item())

        if done:
            reward=-1

        memory.push((torch.tensor([state],dtype=torch.float32)).to(device),
                    torch.tensor(action,dtype=torch.int64).to(device),
                    torch.tensor([next_state],dtype=torch.float32).to(device),
                    torch.tensor([reward],dtype=torch.float32).to(device))
        #learn()
        state=next_state
        steps+=1
        if done:
            print("{2} Episode {0} finished after {1} steps".format(e,steps,'\033[92m' if steps>=195
                                                                    else '\033[99m'))
            episode_durations.append(steps)
         #   plot_duration()
            break

def learn():
    if len(memory)<Batch_Size:
        return

    transitions=memory.sample(Batch_Size)
    batch_state,batch_action,batch_next_state,batch_reward=zip(*transitions)
    
    batch_state=torch.tensor(torch.cat(batch_state))
    batch_action=torch.tensor(torch.cat(batch_action))
    batch_reward=torch.tensor(torch.car(batch_reward))
    batch_next_state=torch.tensor(torch.cat(batch_next_state))

    current_q_values=model(batch_state).gather(1,batch_action)
    max_next_q_values=model(batch_next_state).detach().max(1)[0]
    expected_q_values=batch_reward+(Discount_Factor*max_next_q_values)

    loss=F.smooth_l1_loss(current_q_values,expected_q_values)

    optimizer.zero_grad()



























