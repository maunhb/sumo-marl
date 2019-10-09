from sumo_rl.agents.agent import Agent
from functools import reduce
import numpy as np

class CoordAgent(Agent):

    def __init__(self, joint_starting_state, joint_state_space, joint_action_space, alpha=0.5, gamma=0.95): 
        super(CoordAgent, self).__init__(joint_state_space, joint_action_space)
        self.state = joint_starting_state
        self.action_space = joint_action_space
        self.action = [1,1]
        self.alpha = alpha
        self.gamma = gamma
        # q table is a dict of states with a matrix of a1 rows a2 columns 
        self.q_table = {'{}'.format(self.state): [[0 for j in range(self.action_space[1].n)] for i in range(self.action_space[0].n)]} 
        self.cum_reward = 0

    def new_episode(self):
        pass

    def observe(self, observation):
        ''' To override '''
        pass

    def act(self):
        pass

    def learn(self, new_state, actions, reward, done=False):

        if '{}'.format(new_state) not in self.q_table.keys():
            self.q_table['{}'.format(new_state)] = [[0 for j in range(self.action_space[1].n)] for i in range(self.action_space[0].n)]

        s = self.state
        s1 = new_state
        self.action[0] = actions[0]
        self.action[1] = actions[1]
        
        self.q_table['{}'.format(s)][self.action[0]][self.action[1]] += self.alpha*(reward + self.gamma*max(map(max, self.q_table['{}'.format(s1)])) - self.q_table['{}'.format(s)][self.action[0]][self.action[1]])

        self.state = s1
        self.cum_reward += reward


# for deep : we will need w
# how to you find the grad??
# self.weight = self.q_table[s][a] + alpha (reward + gamma* max(self.q_table[s1]) - self.q_table[s][a] )
# self.w = self.q_table[s][a] + self.alpha*(reward + self.gamma*max(self.q_table[s1]) - self.q_table[s][a])* grad wrt w self.q_table[s][a] 

