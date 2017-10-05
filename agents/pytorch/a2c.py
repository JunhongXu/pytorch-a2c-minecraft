from models import CNNPolicy, MLP
from rollouts import Rollouts
import torch
from torch.autograd import Variable


class A2C(object):
    def __init__(self, envs, model, gamma, v_coeff, e_coeff, lr, nstack, nprocess):
        h, w, c = envs.observation_space.shape
        c = nstack * c
        # self.rollout = Rollouts()
