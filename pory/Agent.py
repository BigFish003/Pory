import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
