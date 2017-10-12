import gym
from gym import spaces
import cv2
from collections import deque
import numpy as np
from gym.wrappers import Monitor
import gym_minecraft


class MinecraftWrapper(gym.Wrapper):
    def __init__(self, env, k=4):
        gym.Wrapper.__init__(self, env)
        h, w, c = self.observation_space.shape
        self.k = k
        self.frames = deque(maxlen=k)
        self.observation_space = spaces.Box(0, 255, shape=(h, w, 1))

    def _to_gray(self, obs):
        return np.expand_dims(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), -1)

    def _reset(self):
        obs = self.env.reset()
        obs = self._to_gray(obs)
        for _ in range(self.k): self.frames.append(obs)
        return self._observation()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = self._to_gray(ob)
        self.frames.append(ob)
        reward = np.sign(reward)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)/255.


def make_minecraft(task_id, seed, rank, resolution=(84, 84), log_dir=None, record_fn=None):
    def _thunk():
        env = gym.make(task_id)
        env.seed(seed+rank)
        env.init(start_minecraft=True, videoResolution=resolution, allowDiscreteMovement=['turn', 'move'])
        if log_dir is not None:
            env = Monitor(env, log_dir, video_callable=record_fn)
        env = MinecraftWrapper(env)
        return env
    return _thunk
