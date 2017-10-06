from policies import CNNPolicy, MLP
from torch.optim import RMSprop, Adam
from rollouts import Rollouts
from torch.nn import functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

class A2C(object):
    def __init__(self, envs, model, gamma, v_coeff, e_coeff, lr, nstack, nprocess):
        h, w, c = envs.observation_space.shape
        c = nstack * c
        # self.rollout = Rollouts()


class PolicyGradient(object):
    def __init__(self, env, gamma, lr, max_step=1500):
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
        self.max_step = max_step

    def calculate_returns(self, rewards):
        nstep = rewards.shape[0]
        # print(nstep)
        returns = np.zeros(nstep)
        returns[-1] = rewards[-1]
        for i in reversed(range(0, nstep-1)):
            returns[i] = self.gamma * returns[i+1] + rewards[i]
        return returns

    def run_episode(self):
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
        # while step != self.max_step:
        while not done:
            # self.env.render()
            # add observation to buffer
            episode_obs.append(step_obs)  # act
            obs_tensor = Variable(torch.from_numpy(step_obs[np.newaxis, :]).float(), volatile=True).cuda()
            step_action, step_value = self.model.act(obs_tensor)
            step_action, step_value = step_action.cpu().numpy()[0][0], step_value.cpu().numpy()[0][0]

            # step
            step_obs, step_rewards, done, _ = self.env.step(step_action)
            # add reward, action, value to buffer
            episode_rewards.append(step_rewards)
            episode_actions.append(step_action)
            episode_values.append(step_value)
            step += 1
            total_reward += step_rewards
        if done:
            episode_rewards.append(0)
        #
        # if done:
        #     # append 0 value to episode_rewards
        #     episode_rewards.append(0)
        # else:
        #     # bootstrap from the last state
        #     obs_tensor = Variable(torch.from_numpy(step_obs[np.newaxis, :]).float(), volatile=True).cuda()
        #     _, value = self.model.act(obs_tensor)
        #     value = value.cpu().numpy()[0][0]
        #     episode_rewards.append(value)

        return np.asarray(episode_obs), np.asarray(episode_rewards), np.asarray(episode_actions), np.asarray(episode_values), total_reward

    def train(self, returns, obs, actions):
        # nstep = returns.shape[0]
        returns = Variable(torch.from_numpy(returns).cuda()).float()
        obs = Variable(torch.from_numpy(obs).cuda()).float()
        actions = Variable(torch.LongTensor(actions)).cuda().view(-1, 1)

        logits, train_values = self.model.forward(obs)
        advantage = returns - train_values.view(-1)
        # print(returns.size(), train_values.size())
        # print(advantage.data)
        # print(returns, train_values)
        log_logits = F.log_softmax(logits)
        selected_logtis = log_logits.gather(1, actions).view(-1)
        # print('Logits', selected_logtis)
        # print('Softmax log', F.softmax(logits))
        # print(torch.LongTensor(actions).cuda())
        # print(logits)
        policy_loss = -Variable(advantage.data) * selected_logtis
        # print('advantage', advantage)
        # print('logits', logits.gather(1, Variable(torch.LongTensor(actions)).cuda().view(-1, 1)))
        # print('softmax', F.log_softmax(logits.gather(1, Variable(torch.LongTensor(actions)).cuda().view(-1, 1))))
        policy_loss = torch.mean(policy_loss)
        entropy = log_logits * F.softmax(logits)
        entropy = - entropy.sum(-1).mean()
        mse = torch.mean(torch.pow(advantage, 2))
        loss = policy_loss + mse - entropy * 0.02
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss, policy_loss, mse, advantage, train_values


if __name__ == '__main__':
    import gym
    pg = PolicyGradient(gym.make('CartPole-v0'), 0.99, 1e-2, max_step=500)
    r = 0
    for i in range(0, 10000):
        obs, rws, acts, values, total_reward = pg.run_episode()
        # print(obs.shape, rws.shape, acts.shape)
        # print(obs.shape)
        # print(rws.shape)
        # print(acts.shape)
        # print(values.shape)
        returns = pg.calculate_returns(rws)
        # print(rws, returns)
        loss, pl, mse, adv, tv = pg.train(returns[:-1], obs=obs, actions=acts)
        # print(pl.data[0], mse.data[0])
        # print(loss, pl, mse)
        # print(adv)
        # print(tv)
        r += total_reward
        if i % 100 == 0:
            print('Update', i, r/100)
            r = 0
    # print(returns)
    # print(rws)
    # print(values)
