import gym
from gym import spaces
import cv2
from collections import deque
import numpy as np
import os
from bench import Monitor


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, h, w, c):
        super(WarpFrame, self).__init__(env)
        self.observation_space = spaces.Box(0, 255, shape=(h, w, c))
        self.h, self.w, self.c = h, w, c

    def _observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.h, self.w))
        return obs.reshape(self.h, self.w, self.c)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        assert shp[2] == 1  # can only stack 1-channel frames
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], k))

    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k): self.frames.append(ob)
        return self._observation()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)


class ClipRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


def wrap(env):
    env = ClipRewardEnv(env)
    env = WarpFrame(env, 84, 84, 1)
    env = FrameStack(env, 4)
    return env


def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = Monitor(env, os.path.join(log_dir, '{}.monitor.json'.format(rank)))
        env = wrap(env)
        return env
    return _thunk
