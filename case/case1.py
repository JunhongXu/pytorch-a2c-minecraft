from envs.subproc_vec_env import SubprocVecEnv
from envs.env_wrappers import wrapper, make_env
import gym_minecraft


# def is_not_nan(data):
#     return data is not None
#
#
# class Test(unittest.TestCase):
#     def test_action_space(self):
#         envs = SubprocVecEnv([make_env('MinecraftBasic-v0', 0, i, '../logs', wrap=wrapper) for i in range(0, 1)])
#         envs.init()
#         self.failIf(is_not_nan(envs.action_space))
#
#     def test_action_num(self):
#         envs = SubprocVecEnv([make_env('MinecraftBasic-v0', 0, i, '../logs', wrap=wrapper) for i in range(0, 1)])
#         envs.init()
#         self.failIf(envs.action_space.n != 14)


if __name__ == '__main__':
     envs = SubprocVecEnv([make_env('MinecraftBasic-v0', 0, i, '../logs', wrap=wrapper) for i in range(0, 5)], minecraft=True)
     import pdb
     pdb.set_trace()
