import numpy as np
from gym import spaces


class EpsilonGreedy:

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def choose(self, opt_actions, action_space):
        actions = opt_actions
        for i in opt_actions.keys():
            if np.random.rand() < self.epsilon:
                actions[i] = int(action_space.sample())
            else:
                actions[i] = opt_actions[i]

        self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)
        #print(self.epsilon)
        return actions

    def reset(self):
        self.epsilon = self.initial_epsilon
