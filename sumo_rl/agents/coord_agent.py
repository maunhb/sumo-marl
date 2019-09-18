from sumo_rl.agents.agent import Agent
from functools import reduce
import numpy as np
from sumo_rl.agents.variable_elimination import VariableElimination

from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy


class CoordAgent(Agent):

    def __init__(self, starting_state, state_space, action_space, alpha, exploration_strategy=EpsilonGreedy(), alpha=0.5, gamma=0.95, coordination_graph, elim_ordering): 
        super(CoordAgent, self).__init__(state_space, action_space)
        self.state = starting_state
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}

        self.w_table = {self.state: [0 for _ in range(action_space.n)]}

        self.exploration = exploration_strategy
        self.coordination_graph = coordination_graph
        self.elimination_ordering = elim_ordering
        self.acc_reward = 0
        self.action_selection_ordering = list(reversed(elim_ordering)) 

    def new_episode(self):
        pass

    def observe(self, observation):
        ''' To override '''
        pass

    def act(self):
        # change this ???
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action


    def learn(self, new_state, reward, done=False):
        # updates the q table
        # maybe also need a w table?
        if new_state not in self.q_table:
            self.q_table[new_state] = [0 for _ in range(self.action_space.n)]
            self.w_table[new_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        s1 = new_state
        a = self.action
        # how to you find the grad??
        self.w_table[s][a] += self.alpha*(reward + self.gamma*max(self.q_table[s1]) - self.q_table[s][a])* grad wrt w_table self.q_table[s][a] 
        self.q_table[s][a] += self.alpha*(reward + self.gamma*max(self.q_table[s1]) - self.q_table[s][a])

        self.state = s1
        self.acc_reward += reward


