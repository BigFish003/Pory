import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.physics.units import action
from tensorflow.python.ops.gen_math_ops import xlogy

from game import *

def encode_state(state):

                    #  0         1         2           3      4        5           6           7          8           9           10          11
                    #terrain  resource improvement climate  border  imp owner  unit owner   unit type   health   imp progress  has atckd    has mvd
    feature_lens = [    6,     8,       6,         17,      3,       3,        3,           5,          1,        1,            2,           2]

    state_tensor = torch.zeros((sum(feature_lens), 11, 11))

    for i in range(len(state)):
        x =  i % 11
        y =  i // 11
        state_tensor[state[i][0],x,y] = 1
        state_tensor[feature_lens[0:1] + state[i][1], x, y] = 1
        state_tensor[sum(feature_lens[0:2]) + state[i][2], x, y] = 1
        state_tensor[sum(feature_lens[0:3]) + state[i][3], x, y] = 1
        state_tensor[sum(feature_lens[0:4]) + state[i][4], x, y] = 1
        state_tensor[sum(feature_lens[0:5]) + state[i][5], x, y] = 1
        state_tensor[sum(feature_lens[0:6]) + state[i][6], x, y] = 1
        state_tensor[sum(feature_lens[0:7]) + state[i][7], x, y] = 1
        state_tensor[sum(feature_lens[0:8]), x, y] = state[i][8]
        state_tensor[sum(feature_lens[0:9]), x, y] = state[i][9]
        state_tensor[sum(feature_lens[0:10]) + state[i][10], x, y] = 1
        state_tensor[sum(feature_lens[0:11]) + state[i][11], x, y] = 1

    return state_tensor


class ResNet(nn.Module):
    def __init__(self,game,num_resBlocks, num_hidden):
        super().__init__()
        action_size = 121
        height = 11
        width = 11
        num_features = 58
        self.startBlock = nn.Sequential(
            nn.Conv2d(num_features, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.selectTile = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * height * width  , action_size)
        )

        self.actionType = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * height * width  , action_size)
        )

        self.targetTile = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * height * width  , action_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * height * width, 1)
        )

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = ResBlock(x)

        policy = self.policyHead(x)
        value = self.valueHead(x)

        return policy, value
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self,x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

env = PolytopiaEnv()
env.reset()
state = env.get_obs(1)
state = state [0:120]
print(encode_state(state)[57])