import numpy as np
from torch.optim import RMSprop, Adam
from rollouts import Rollouts
from torch.nn import functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable


class A2C(object):
    def __init__(self, envs, model, gamma=0.99, v_coeff=0.5, e_coeff=0.02, lr=1e-3, nstep=50, nstack=None):
        """A parallel version of actor-critic method"""
        self.envs = envs
        self.nactions = self.envs.action_space.n
        self.nstack = nstack
        self.nstep = nstep
        self.obs_space = self.create_obs_space()
        if len(self.obs_space) > 1:
            self.use_rgb = True
        else:
            self.use_rgb = False
        # loss parameters
        self.v_coeff, self.e_coeff, self.lr = v_coeff, e_coeff, lr
        self.gamma = gamma
        self.nprocess = len(envs.remotes)

        # initialize model
        self.model = model(self.obs_space, self.nactions)
        self.model.cuda()
        # start environment
        self.step_obs = envs.reset()
        self.optimizer = Adam(params=self.model.parameters(), lr=lr)

    def run_episode(self, episode):
        """
        Run one episode of the environments. Episode in each environment finishes when either 1) done is True
        or 2) step == self.nstep

        Returns episode rewards , predicted values, observations, and actions
        """
        episode_rws, episode_values, episode_actions, episode_obs, episode_dones = [[] for _ in range(5)]
        final_reward = 0
        for step in range(self.nstep):
            self.envs.render()
            # reshape observations to desired shape
            self.step_obs = np.concatenate(self.step_obs, axis=0)
            self.step_obs = self.step_obs.reshape((-1,) + self.obs_space)
            # model forward (nprocess, *obs_shape)
            _step_obs = Variable(torch.from_numpy(self.step_obs).float(), volatile=True).cuda()
            step_action, step_value = self.model.act(_step_obs)
            step_action = step_action.flatten()
            step_value = step_value.flatten()
            # step
            step_obs, step_reward, step_done, _ = self.envs.step(step_action)
            final_reward += np.asarray(step_reward).mean()
            step_reward = np.sign(step_reward)
            for i, done in enumerate(step_done):
                if done:
                    step_obs[i] *= 0

            # store to buffer
            episode_obs.append(self.step_obs)
            episode_actions.append(step_action)
            episode_values.append(step_value)
            episode_dones.append(step_done)
            episode_rws.append(step_reward)
            self.step_obs = step_obs
        # bootstrap
        last_step_obs = np.concatenate(self.step_obs)
        last_step_obs = last_step_obs.reshape((-1,) + self.obs_space)
        # model forward (nprocess, *obs_shape)
        _step_obs = Variable(torch.from_numpy(last_step_obs).float(), volatile=True).cuda()
        _, last_step_value = self.model.act(_step_obs)
        episode_values.append(last_step_value.flatten())
        episode_obs = np.asarray(episode_obs).swapaxes(1, 0).reshape((-1, ) + self.obs_space)
        episode_rws = np.asarray(episode_rws).swapaxes(1, 0)
        episode_actions = np.asarray(episode_actions).swapaxes(1, 0)
        episode_dones = np.asarray(episode_dones).swapaxes(1, 0)
        episode_values = np.asarray(episode_values).swapaxes(1, 0)
        returns = self.calculate_returns(episode_values[:, -1], episode_rws, episode_dones)
        return episode_obs, episode_rws.flatten(), episode_values.flatten(), episode_actions.flatten(), episode_dones.flatten(), returns.flatten(), final_reward

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
        loss = policy_loss + mse * self.v_coeff - entropy * self.e_coeff

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), 0.5)
        self.optimizer.step()
        return loss, policy_loss, mse, advantage, train_values, entropy

    def create_obs_space(self):
        obs_shape = self.envs.observation_space.shape
        if len(obs_shape) > 1:
            self.use_rgb = True
            h, w, c = self.envs.observation_space.shape
            c *= self.nstack
            obs_shape = (c, h, w)
        else:
            c = self.envs.observation_space.shape
            obs_shape = c
        return obs_shape

    def calculate_returns(self, last_value, episode_rws, episode_dones):
        episode_dones = 1 - episode_dones
        last_value = last_value * episode_dones[:, -1]
        returns = []
        for env_idx, (value, reward, done) in enumerate(zip(last_value, episode_rws, episode_dones)):
            R = np.zeros(self.nstep+1)
            R[-1] = value
            for idx in reversed(range(self.nstep)):
                R[idx] = self.gamma * R[idx+1] * done[idx] + reward[idx]
            returns.append(R[:-1])
        return np.asarray(returns)


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
        for i in reversed(range(0, nstep - 1)):
            returns[i] = self.gamma * returns[i + 1] + rewards[i]
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
            step_action, step_value = step_action[0][0], step_value[0][0]

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

        return np.asarray(episode_obs), np.asarray(episode_rewards), np.asarray(episode_actions), np.asarray(
            episode_values), total_reward

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
        loss = policy_loss + mse / 2 - entropy * 0.02

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), 0.5)
        self.optimizer.step()
        return loss, policy_loss, mse, advantage, train_values, entropy


