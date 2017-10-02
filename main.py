import os

import gym
import numpy as np
import torch
from torch.autograd import Variable

from envs.bench import Monitor

num_updates = 5000
num_steps = 50
render = True

def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = Monitor(env, os.path.join(log_dir, '{}.monitor.json'.format(rank)))
        return env
    return _thunk


def calculate_returns(R, next_return, gamma=0.99):
    """
        R is a N*M matrix, where N is number of process, M is number of steps
    """
    returns = []
    for s in reversed(range(R.shape[1])):
        next_return = R[:, s] + gamma * next_return
        print(next_return)
        returns.append(next_return)
    return returns


def wrap_pytorch(x):
    return Variable(torch.from_numpy(x).float().cuda())

if __name__ == '__main__':
    # # init env
    # envs = SubprocVecEnv([make_env('CartPole-v0', 0, i, 'logs') for i in range(0, 4)])
    # policy = MLP(envs.observation_space.shape[0], envs.action_space.n)
    # policy.cuda()
    # observations = envs.reset()
    # for update in range(num_updates):
    #     observations = wrap_pytorch(observations)
    #     rewards = []
    #     values = []
    #     actions = []
    #     if render:
    #         envs.render()
    #     for i in range(0, num_steps):
    #         if render:
    #             envs.render()
    #         # act
    #         action, value = policy.act(observations)
    #         action = action.data.cpu().numpy().flatten()
    #         value = value.data.cpu().numpy().flatten()
    #         # receive observations and rewards
    #         observations, reward, dones, info = envs.step(action)
    #         print(reward, dones)
    #         observations = wrap_pytorch(observations)
    #         rewards.append(reward)
    #         values.append(value)
    #         actions.append(action)
    #     print(np.stack(rewards, 1).shape)
    #     print(np.stack(actions, 1).shape)
    #     print(np.stack(values, 1).shape)
    #     break
        # calculate returns
        # returns = calculate_returns(rewards, values, actions)

        # update policy
        # update_policy(policy, returns, )
        # for i in range(0, 1000):
            # envs.
    # # start training process
    # for update in range(num_updates):
    #     # run one episode
    #
    #     # evaluate returns
    #
    #     # update
    # print(bench.Monitor)
    R = np.array([
        [1, 1, 0, 1],
        [0, 0, 1, 1]
    ])

    calculate_returns(R, np.array([
        [0], [0]
    ]))
