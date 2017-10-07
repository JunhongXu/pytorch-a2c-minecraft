from policies import CNNPolicy, MLP
from torch.optim import RMSprop, Adam
from rollouts import Rollouts
from torch.nn import functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

class A2C(object):
    # TODO: implement the parllel version of A2C without using multi-processing
    # TODO: with multi-processing method to speed up training
    def __init__(self, envs, model, gamma, v_coeff, e_coeff, lr, nstep, nstack=None):
        """A parallel version of actor-critic method"""
        self.envs = envs
        self.nactions = self.envs.action_space.n
        self.nstack = nstack
        self.nstep = nstep
        self.obs_space = self.create_obs_space()

        # loss parameters
        self.v_coeff, self.e_coeff, self.lr = v_coeff, e_coeff, lr
        self.gamma = gamma
        self.nprocess = len(envs)

        # initialize model
        self.model = model(self.obs_space, self.nactions)

        # self.rollout = Rollouts()

    def run_episode(self, episode):
        """
        Run one episode of the environments. Episode in each environment finishes when either 1) done is True
        or 2) step == self.nstep

        Returns episode rewards , predicted values, observations, and actions
        """
        episode_rws, episode_values, episode_actions, episode_obs = [self._2d_list() for _ in range(4)]
        dones = np.array([False for _ in range(self.nprocess)])



    def create_obs_space(self):
        obs_shape = self.envs.observation_space.shape
        if len(obs_shape) > 1:
            self.use_rgb = True
            h, w, c = self.envs.observation_space.shape
            c *= self.nstack
            obs_shape = (h, w, c)
        else:
            c = self.envs.observation_space.shape[0]
            obs_shape = c
        return obs_shape

    def _2d_list(self):
        return [[] for _ in range(self.nprocess)]


class ActorCritic(object):
    def __init__(self, env, gamma, lr):
        """
        REINFORCE algorithm with advantage estimation
        """
        self.env = env
        self.nactions = env.action_space.n
        self.nobservation = env.observation_space.shape[0]
        self.model = MLP(num_actions=self.nactions, num_obs=self.nobservation)
        self.model.cuda()
        self.gamma = gamma
        self.optimizer = Adam(params=self.model.parameters(), lr=lr)

    def calculate_returns(self, rewards):
        nstep = rewards.shape[0]
        # print(nstep)
        returns = np.zeros(nstep)
        returns[-1] = rewards[-1]
        for i in reversed(range(0, nstep-1)):
            returns[i] = self.gamma * returns[i+1] + rewards[i]
        return returns

    def run_episode(self, episode):
        """
        Run through one episode of the game and collect episode_values, episode_rewards, episode_obs, episode_actions
        and calculate returns
        """
        episode_rewards = []
        episode_actions = []
        episode_obs = []
        episode_values = []
        step_obs = self.env.reset()
        total_reward = 0.0
        done = False
        step = 0
        while not done:
            if episode % 500 == 0:
                self.env.render()
            # add observation to buffer
            episode_obs.append(step_obs)  # act
            obs_tensor = Variable(torch.from_numpy(step_obs[np.newaxis, :]).float(), volatile=True).cuda()
            step_action, step_value = self.model.act(obs_tensor)
            step_action, step_value = step_action.cpu().numpy()[0][0], step_value.cpu().numpy()[0][0]

            # step
            step_obs, step_rewards, done, _ = self.env.step(step_action)
            r = np.sign(step_rewards)
            # add reward, action, value to buffer
            episode_rewards.append(r)
            episode_actions.append(step_action)
            episode_values.append(step_value)
            step += 1
            total_reward += step_rewards
        if done:
            episode_rewards.append(0)

        return np.asarray(episode_obs), np.asarray(episode_rewards), np.asarray(episode_actions), np.asarray(episode_values), total_reward

    def train(self, returns, obs, actions):
        returns = Variable(torch.from_numpy(returns).cuda()).float()
        obs = Variable(torch.from_numpy(obs).cuda()).float()
        actions = Variable(torch.LongTensor(actions)).cuda().view(-1, 1)

        # get action logits and values from the model
        logits, train_values = self.model.forward(obs)
        advantage = returns - train_values.view(-1)
        log_logits = F.log_softmax(logits)
        selected_logtis = log_logits.gather(1, actions).view(-1)
        policy_loss = -Variable(advantage.data) * selected_logtis
        # calculate policy loss
        policy_loss = torch.mean(policy_loss)

        # calculate entropy
        entropy = log_logits * F.softmax(logits)
        entropy = - entropy.sum(-1).mean()

        # value function
        mse = torch.mean(torch.pow(advantage, 2))
        loss = policy_loss + mse/2 - entropy * 0.02

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), 0.5)
        self.optimizer.step()
        return loss, policy_loss, mse, advantage, train_values, entropy


if __name__ == '__main__':
    import gym
    pg = ActorCritic(gym.make('LunarLander-v2'), 0.99, 4e-4)
    r = 0
    for i in range(0, 100000):
        obs, rws, acts, values, total_reward = pg.run_episode(i)
        returns = pg.calculate_returns(rws)
        loss, pl, mse, adv, tv, entropy = pg.train(returns[:-1], obs=obs, actions=acts)
        r += total_reward
        if i % 100 == 0:
            print('Update', i, r/100, mse.data[0], entropy.data[0], pl.data[0])
            r = 0

