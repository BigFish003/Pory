import time
import warnings
from torch import multiprocessing
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from game_glory import PolytopiaEnv
from render import Render
from Agent import *
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2)

class PPOConvAgent(nn.Module):
    def __init__(self, input_channels=57,
                 num_actions1=122,
                 num_actions2=6,
                 num_actions3=121,
                 embedding_dim=32):
        super(PPOConvAgent, self).__init__()

        # Convolutional backbone for policy
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(128 * 121, 256)

        self.action1_embedding = nn.Embedding(num_actions1, embedding_dim)
        self.action2_embedding = nn.Embedding(num_actions2, embedding_dim)

        self.fc_policy1 = nn.Linear(256, num_actions1)
        self.fc_policy2 = nn.Linear(256 + embedding_dim, num_actions2)
        self.fc_policy3 = nn.Linear(256 + 2 * embedding_dim, num_actions3)

        self.num_actions1 = num_actions1
        self.num_actions2 = num_actions2
        self.num_actions3 = num_actions3
        self.embedding_dim = embedding_dim

    def compute_base(self, x):
        """Compute the base state representation for policy."""
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = torch.flatten(h, start_dim=1)
        h = F.relu(self.fc1(h))
        return h

    def get_action1_logits(self, h):
        return self.fc_policy1(h)

    def get_action2_logits(self, h, a1):
        # Ensure a1 is [B]
        if a1.dim() == 2 and a1.size(1) == 1:
            a1 = a1.squeeze(1)
        a1_emb = self.action1_embedding(a1)
        h2 = torch.cat([h, a1_emb], dim=-1)
        return self.fc_policy2(h2)

    def get_action3_logits(self, h, a1, a2):
        # Ensure a1, a2 are [B]
        if a1.dim() == 2 and a1.size(1) == 1:
            a1 = a1.squeeze(1)
        if a2.dim() == 2 and a2.size(1) == 1:
            a2 = a2.squeeze(1)
        a1_emb = self.action1_embedding(a1)
        a2_emb = self.action2_embedding(a2)
        h3 = torch.cat([h, a1_emb, a2_emb], dim=-1)
        return self.fc_policy3(h3)


class PPOValueNet(nn.Module):
    def __init__(self, input_channels=57):
        super(PPOValueNet, self).__init__()
        # A separate network for value estimation
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(128 * 121, 256)
        self.fc_value = nn.Linear(256, 1)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = torch.flatten(h, start_dim=1)
        h = F.relu(self.fc1(h))
        value = self.fc_value(h)
        return value


def collect_trajectories(policy1, policy2, value_net, num_steps=10, gamma=0.99, lam=0.95):
    env = PolytopiaEnv()
    env.reset()
    render = Render()

    states1 = []
    tiles1 = []
    actions1 = []
    target_tiles1 = []
    dones1 = []
    rewards1 = []

    states2 = []
    tiles2 = []
    actions2 = []
    target_tiles2 = []
    dones2 = []
    rewards2 = []

    rest = 0.2

    previous_points1 = env.get_turn_and_points()[1]
    previous_points2 = env.get_turn_and_points()[2]
    for _ in range(num_steps):
        if env.turn == 1:
            policy = policy1
            state_p1 = env.encode_state(env.get_obs(1)[:120]).unsqueeze(0)  # Player 1 State
            states1.append(state_p1)
            # Compute base features once using the policy
            h_p1 = policy.compute_base(state_p1)

            # Compute value separately
            value_p1 = value_net(state_p1)

            # First action
            action1_logits = policy.get_action1_logits(h_p1)
            mask = env.get_mask()
            for i in range(len(mask)):
                if mask[i] == 0:
                    options = True
                    action1_logits[0, i] = -1e10
            action_probs = F.softmax(action1_logits, dim=-1)
            action1 = torch.multinomial(action_probs, 1)  # still a tensor
            action1_idx = action1.item()  # convert to int
            env.step(action1_idx)  # now stepping with an int

            # Second action
            action2 = None
            if env.step_phase == 1:
                print(env.selected_tile)
                action2_logits = policy.get_action2_logits(h_p1, action1)  # use action1 as tensor for embeddings
                mask = env.get_mask()
                for i in range(len(mask)):
                    if mask[i] == 0:
                        action2_logits[0, i] = -1e10
                action_probs = F.softmax(action2_logits, dim=-1)
                action2 = torch.multinomial(action_probs, 1)
                action2_idx = action2.item()
                env.step(action2_idx)

            # Third action (if phase == 2)
            action3 = None
            if env.step_phase == 2:
                print(env.selected_tile, env.selected_action)
                action3_logits = policy.get_action3_logits(h_p1, action1, action2)
                mask = env.get_mask()
                for i in range(len(mask)):
                    if mask[i] == 0:
                        action3_logits[0, i] = -1e10
                action_probs = F.softmax(action3_logits, dim=-1)
                action3 = torch.multinomial(action_probs, 1)
                action3_idx = action3.item()
                env.step(action3_idx)

            print("Value:", value_p1)
            print("Action1 logits (masked):", action1)
            print("Action2 logits (masked):", action2)
            print("Action3 logits (masked):", action3)

            tiles1.append(action1)
            actions1.append(action2)
            target_tiles1.append(action3)

            done ,points, x = env.get_turn_and_points()
            rewards1.append(points - previous_points1)
            dones1.append(0 if done != True else 1)
            previous_points1 = points
            obs = env.get_obs(1)

            render.render(obs)
            time.sleep(rest)
        if env.turn == 2:
            policy = policy2
            state_p1 = env.encode_state(env.get_obs(2)[:120]).unsqueeze(0)  # Player 1 State
            states2.append(state_p1)
            # Compute base features once using the policy
            h_p1 = policy.compute_base(state_p1)

            # Compute value separately
            value_p1 = value_net(state_p1)

            # First action
            action1_logits = policy.get_action1_logits(h_p1)
            mask = env.get_mask()
            for i in range(len(mask)):
                if mask[i] == 0:
                    options = True
                    action1_logits[0, i] = -1e10
            action_probs = F.softmax(action1_logits, dim=-1)
            action1 = torch.multinomial(action_probs, 1)  # still a tensor
            action1_idx = action1.item()  # convert to int
            env.step(action1_idx)  # now stepping with an int

            # Second action
            action2 = None
            if env.step_phase == 1:
                action2_logits = policy.get_action2_logits(h_p1, action1)  # use action1 as tensor for embeddings
                mask = env.get_mask()
                for i in range(len(mask)):
                    if mask[i] == 0:
                        action2_logits[0, i] = -1e10
                action_probs = F.softmax(action2_logits, dim=-1)
                action2 = torch.multinomial(action_probs, 1)
                action2_idx = action2.item()
                env.step(action2_idx)

            # Third action (if phase == 2)
            action3 = None
            if env.step_phase == 2:
                print(env.selected_tile, env.selected_action)
                action3_logits = policy.get_action3_logits(h_p1, action1, action2)
                mask = env.get_mask()
                for i in range(len(mask)):
                    if mask[i] == 0:
                        action3_logits[0, i] = -1e10
                action_probs = F.softmax(action3_logits, dim=-1)
                action3 = torch.multinomial(action_probs, 1)
                action3_idx = action3.item()
                env.step(action3_idx)

            print("Value:", value_p1)
            print("Action1 logits (masked):", action1)
            print("Action2 logits (masked):", action2)
            print("Action3 logits (masked):", action3)

            tiles2.append(action1)
            actions2.append(action2)
            target_tiles2.append(action3)

            done, x, points = env.get_turn_and_points()
            rewards2.append(points - previous_points1)
            dones2.append(0 if done != True else 1)
            previous_points2 = points
            obs = env.get_obs(2)

            render.render(obs)

            time.sleep(rest)

    return states1, tiles1, actions1, target_tiles1, rewards1, dones1

policy1 = PPOConvAgent()
policy2 = PPOConvAgent()
value_net = PPOValueNet()

policy1old = PPOConvAgent()
policy2old = PPOConvAgent()

s, t, a, ta, r, d = (collect_trajectories(policy1,policy2, value_net,num_steps=1000))
print(len(s),len(t),len(a),len(ta),len(r),len(d))