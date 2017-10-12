import gym

import numpy as np
from gym.wrappers import Monitor


class LunarLanderWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def _reset(self):
        obs = self.env.reset()
        return obs

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        reward = np.sign(reward)
        return ob, reward, done, info


def make_lunarlander(task_id, seed, rank, log_dir=None, record_fn=None):
    def _thunk():
        env = gym.make(task_id)
        env.seed(seed+rank)
        if log_dir is not None:
            env = Monitor(env, log_dir, video_callable=record_fn)
        env = LunarLanderWrapper(env)
        return env
    return _thunk
