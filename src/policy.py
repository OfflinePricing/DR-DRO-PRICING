# policy.py
import torch
import torch.nn as nn
from . import config

class BasePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.min_p = config.PRICE_MIN
        self.max_p = config.PRICE_MAX

    def forward_raw(self, x):
        raise NotImplementedError

    def forward(self, x):
        """Output scaled to [PRICE_MIN, PRICE_MAX]"""
        raw_out = self.forward_raw(x)
        # Sigmoid compresses output to (0, 1), then scale
        return torch.sigmoid(raw_out) * (self.max_p - self.min_p) + self.min_p

class LinearPolicy(BasePolicy):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward_raw(self, x):
        return self.linear(x)

class MLPPolicy(BasePolicy):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward_raw(self, x):
        return self.net(x)