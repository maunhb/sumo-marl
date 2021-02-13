from sumo_rl.agents.agent import Agent
from functools import reduce
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math
import os.path, csv, random

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.input_shape[0], 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, self.num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        x = self.fc3(x)

        return x

class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent(Agent):
    def __init__(self, config, action_space, state_space):
        self.rewards = []
        self.action_space = action_space
        self.observation_space = state_space
        self.action_log_frequency = config.ACTION_SELECTION_COUNT_FREQUENCY
        self.action_selections = [0 for _ in range(self.action_space.n)]
    
    # Define the DQN networks
    def declare_networks(self):
        self.model = DQN(self.num_feats, self.num_actions)
        # Create `self.target_model` with the same network architecture
        self.target_model = DQN(self.num_feats, self.num_actions)

    # Define the Replay Memory
    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)
    
    # Append the new transition to the Replay Memory
    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))
    
    # Sample transitions from the Replay Memory
    def sample_minibatch(self):
        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,)+self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        # Sometimes all next states are false
        try:
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values
    def new_episode(self):
        pass

    def observe(self, observation):
        ''' To override '''
        pass
    def learn(self, new_state, reward, done=False):
        pass

    # Sample action - was  get_action
    def act(self, s, eps=0.1):
        with torch.no_grad():
            # Epsilon-greedy
            if np.random.random() >= eps:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                while X.size()[1] < self.model.input_shape[0]:
                    zero = torch.tensor([[0.0]], device=self.device, dtype=torch.float)
                    X = torch.cat((X,zero), dim=1)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)


class Learning(DQNAgent):
    def __init__(self, config=None,  action_space=None, state_space=None):
        super().__init__(config=config, action_space=action_space, state_space=state_space)
    
    # Compute loss from the Bellman Optimality Equation
    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = batch_vars

        current_q_values = self.model(batch_state).gather(1, batch_action)

        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)

            expected_q_values = batch_reward + self.gamma*max_next_q_values
        
        diff = (expected_q_values - current_q_values)

        loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    # Update both networks (the agent and the target)
    def update(self, s, a, r, s_, sample_idx=0):
        self.append_to_replay(s, a, r, s_)
        
        # When not to update 
        if sample_idx < self.learn_start or sample_idx % self.update_freq != 0:
            return None

        batch_vars = self.sample_minibatch()
        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()

    def update_target_model(self):
        # Copy weights from model to target_model following `target_net_update_freq`.
        self.update_count+=1
        if self.update_count % self.target_net_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

class CoordDQNAgent(Learning):
    def __init__(self, config=None, action_space=None, state_space=None):
        super().__init__(config=config, action_space=action_space, state_space=state_space)
        self.device = config.device

        self.action_space = action_space
        self.observation_space = state_space

        # Hyperparameters
        self.gamma = config.GAMMA
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ

        # Environment specific parameters
        self.num_feats = self.observation_space.shape
        self.num_actions = self.action_space.n


        self.declare_networks()
        self.declare_memory()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LR)
        
        # Move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)
        
        self.model.train()
        self.target_model.train()
        
        self.update_count = 0

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)
    
    def MSE(self, x):
        return 0.5 * x.pow(2)

    def save_reward(self, reward):
        self.rewards.append(reward)


class Config():
    def __init__(self, sim_time=50000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Main agent variables
        self.GAMMA=0.99
        self.LR=1e-3
        
        # Epsilon variables
        self.epsilon_start    = 1.0
        self.epsilon_final    = 0.01
        self.epsilon_decay    = 10000
        self.epsilon_by_sample = lambda sample_idx: config.epsilon_final + (config.epsilon_start - config.epsilon_final) * math.exp(-1. * sample_idx / config.epsilon_decay)

        # Memory
        self.TARGET_NET_UPDATE_FREQ = 1000
        self.EXP_REPLAY_SIZE = 10000
        self.BATCH_SIZE = 64

        # Learning control variables
        self.LEARN_START = 1000
        self.MAX_SAMPLES = sim_time #50000
        self.UPDATE_FREQ = 1

        # Data logging parameters
        self.ACTION_SELECTION_COUNT_FREQUENCY = 1000
        
config = Config()
