import numpy as np


class Rollouts(object):
    """
    This class stores rewards, observations, actions taken, terminal states, log probability of actions for each trail.
    This class calculates returns by return_t = r_t + gamma * return_(t+1)
    """
    def __init__(self, gamma, nprocess, nsteps, nactions, obs_shape):
        self.gamma = gamma
        self.nsteps = nsteps
        self.nprocess = nprocess
        self.obs_shape = obs_shape
        self.nactions = nactions
        self.use_rgb = True if len(obs_shape) == 3 else False
        # need one more state to evaluate return[i]
        self.returns = np.zeros((nsteps + 1, nprocess))
        if self.use_rgb:
            h, w, c = obs_shape
            self.observations = np.zeros((nsteps, nprocess, h, w, c))
        else:
            self.observations = np.zeros((nsteps, nprocess, obs_shape[0]))
        self.log_actions = np.zeros((nsteps, nprocess, nactions))
        self.actions = np.zeros((nsteps, nprocess))
        self.values = np.zeros((nsteps, nprocess))
        self.dones = np.zeros((nsteps, nprocess))
        self.rewards = np.zeros((nsteps, nprocess))

    def update(self, step, dones, rewards, log_actions, actions, values, observations):
        """
            update the current rollout buffer
        """
        # update observation, update from t1, t0 is reset observations
        self.observations[step] = observations
        self.log_actions[step] = log_actions
        self.actions[step] = actions
        self.values[step] = values
        self.dones[step] = dones
        self.rewards[step] = rewards

    def clear(self):
        self.returns = np.zeros((self.nsteps + 1, self.nprocess))
        if self.use_rgb:
            h, w, c = self.obs_shape
            self.observations = np.zeros((self.nsteps, self.nprocess, h, w, c))
        else:
            self.observations = np.zeros((self.nsteps, self.nprocess, self.obs_shape))
        self.log_actions = np.zeros((self.nsteps, self.nprocess, self.nactions))
        self.actions = np.zeros((self.nsteps, self.nprocess))
        self.values = np.zeros((self.nsteps, self.nprocess))
        self.dones = np.zeros((self.nsteps, self.nprocess))
        self.rewards = np.zeros((self.nsteps, self.nprocess))

    def calc_returns(self, value):
        self.returns[-1] = value
        self.dones = 1 - self.dones
        for step in reversed(range(self.nsteps)):
            self.returns[step] = self.gamma * self.returns[step+1] * self.dones[step] + self.rewards[step]

