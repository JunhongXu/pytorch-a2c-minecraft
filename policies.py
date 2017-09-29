import torch
import torch.nn as nn
from torch.nn import functional as F


class CNNPolicy(nn.Module):
    def __init__(self, obs_space, num_actions):
        """64*3*3 -> 128*3*3 -> 256*3*3 -> 512"""
        super(CNNPolicy, self).__init__()
        h, w = obs_space
        self.obs_space = obs_space
        self.num_actions = num_actions

        self.conv1 = self.__make_conv_elu(3, 32, 8, 4)
        self.conv2 = self.__make_conv_elu(32, 64, 4, 2)
        self.conv3 = self.__make_conv_elu(64, 64, 3, 1)

        self.fc1 = nn.Sequential(
            nn.Linear(256*(h//8), 512),
            nn.ELU()
        )

        self.value = nn.Linear(512, 1)
        self.action = nn.Linear(512, num_actions)

        self.init_weight()

    def init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module.bias.data.zero_()
                nn.init.orthogonal(module.weight.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.fc1(x)
        action = F.softmax(self.action(x))
        value = self.value(x)

        return action, value

    def act(self, x):
        action, _ = self.forward(x)
        # return action.

    @staticmethod
    def __make_conv_elu(input_feats, output_feats, size, stride, padding=0):
        return nn.Sequential(nn.Conv2d(input_feats, output_feats, kernel_size=size, stride=stride, padding=padding), nn.ELU())

if __name__ == '__main__':
    cnn = CNNPolicy((84, 84), 4)

