# Code in this file is copied and adapted from
# https://github.com/modestyachts/ARS/blob/master/code/policies.py


import numpy as np
from Utils.filter import get_filter


class Policy(object):

    def __init__(self, policy_params):

        self.state_dim = policy_params['state_dim']
        self.action_dim = policy_params['action_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.state_filter = get_filter(policy_params['filter'], shape=(self.state_dim,))
        self.update_filter = True
        
    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.state_filter

    def act(self, state):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError


class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.weights = np.zeros((self.action_dim, self.state_dim), dtype=np.float64)

    def act(self, state):
        state = self.state_filter(state, update=self.update_filter)
        return np.dot(self.weights, state)

    def get_weights_plus_stats(self):
        
        mu, std = self.state_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux
        
