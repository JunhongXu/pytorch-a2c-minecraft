import os
import gym_minecraft
import gym
import numpy as np

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
    env = gym.make('MinecraftBasic-v0')
    env.init(start_minecraft=True)
    obs = env.reset()

    # this import order is necessary to perform in Minecraft env
    from torch.autograd import Variable
    from agents.pytorch.models import CNNPolicy
    import cv2
    import torch


    policy = CNNPolicy(num_actions=4, obs_space=(84, 84))
    policy.cuda()
    done = False
    for u in range(num_steps):
        while not done:
            obs = cv2.resize(obs, (84, 84))
            obs = np.transpose(obs, (2, 0, 1))
            tensor_obs = wrap_pytorch(obs)
            tensor_obs = tensor_obs.unsqueeze(dim=0)
            results = policy.forward(tensor_obs)
            print(results)
            env.render()
            action = env.action_space.sample()
            obs, r, done, _ = env.step(action)

        if done:
            env.reset()
            done = False
