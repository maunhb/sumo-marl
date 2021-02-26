from sumo_rl.agents.agent import Agent
from functools import reduce
import numpy as np

class CoordAgent(Agent):

    def __init__(self, joint_starting_state, joint_state_space, 
                       joint_action_space, alpha=0.5, gamma=0.95): 
        super(CoordAgent,self).__init__(joint_state_space, joint_action_space)
        self.state = joint_starting_state
        self.action_space = joint_action_space
        self.action = [1,1]
        self.alpha = alpha
        self.gamma = gamma
        # q table is a dict of states with a matrix of a1 rows a2 columns 
        self.q_table = {'{}'.format(self.state):
                         [[0 for j in range(self.action_space[1].n)] 
                             for i in range(self.action_space[0].n)]} 
        self.cum_reward = 0

    def new_episode(self):
        pass

    def observe(self, observation):
        ''' To override '''
        pass

    def act(self):
        pass

    def learn(self, next_s, actions, reward, done=False):

        if '{}'.format(next_s) not in self.q_table.keys():
            J = self.action_space[1].n
            I = self.action_space[0].n
            self.q_table['{}'.format(next_s)] = [[0 for j in range(J)] 
                                                    for i in range(I)]

        s = self.state
        s1 = next_s
        self.action[0] = actions[0]
        self.action[1] = actions[1]

        update = self.alpha*(reward + self.gamma*max(map(max, 
                 self.q_table['{}'.format(s1)])) - 
                 self.q_table['{}'.format(s)][self.action[0]][self.action[1]])
        
        self.q_table['{}'.format(s)][self.action[0]][self.action[1]] += update

        self.state = s1
        self.cum_reward += reward
