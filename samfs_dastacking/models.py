import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, max_size: int, state_dim: int, action_dim: int):
        self.max_size = int(max_size)
        self.s = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.a = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.r = np.zeros((self.max_size, 1), dtype=np.float32)
        self.s2 = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.d = np.zeros((self.max_size, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s2, done):
        i = self.ptr
        self.s[i] = np.asarray(s, dtype=np.float32)
        self.a[i] = np.asarray(a, dtype=np.float32)
        self.r[i, 0] = float(r)
        self.s2[i] = np.asarray(s2, dtype=np.float32)
        self.d[i, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class OUNoise:
    def __init__(self, size: int, theta: float = 0.15, sigma: float = 0.2):
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(size, dtype=np.float32)

    def reset(self):
        self.state[:] = 0.0

    def sample(self):
        dx = self.theta * (-self.state) + self.sigma * np.random.randn(*self.state.shape)
        self.state = self.state + dx
        return self.state


class FeatureAttention(nn.Module):
    """Eq. (11)-(12): alpha=softmax(W_att x + b), X=alpha*x."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        alpha = F.softmax(self.linear(x), dim=1)
        return alpha * x


class ResidualBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.use_residual = dim_in == dim_out

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_residual else y


class AttentionStateMixin:
    def encode_state(self, state):
        x = state[:, : self.feature_dim]
        expert_preds = state[:, self.feature_dim :]
        x_att = self.attention(x)
        return torch.cat([x_att, expert_preds], dim=1)


class Actor(nn.Module, AttentionStateMixin):
    def __init__(
        self,
        feature_dim: int,
        expert_dim: int,
        hidden=(256, 160, 96),
        dropout: float = 0.28,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention = FeatureAttention(feature_dim)
        state_dim = feature_dim + expert_dim
        layers = []
        prev = state_dim
        for width in hidden:
            layers.append(ResidualBlock(prev, width, dropout))
            prev = width
        self.shared = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(prev, max(prev // 2, expert_dim)),
            nn.ReLU(),
            nn.LayerNorm(max(prev // 2, expert_dim)),
            nn.Linear(max(prev // 2, expert_dim), expert_dim),
        )

    def forward(self, state):
        encoded = self.encode_state(state)
        logits = self.head(self.shared(encoded))
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        return F.softmax(logits, dim=1)


class Critic(nn.Module, AttentionStateMixin):
    def __init__(
        self,
        feature_dim: int,
        expert_dim: int,
        hidden=(512, 320, 192),
        dropout: float = 0.28,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention = FeatureAttention(feature_dim)
        state_dim = feature_dim + expert_dim
        s_h = hidden[0]
        a_h = max(expert_dim, 32)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, s_h),
            nn.ReLU(),
            nn.LayerNorm(s_h),
            nn.Dropout(dropout),
            nn.Linear(s_h, s_h),
            nn.ReLU(),
            nn.LayerNorm(s_h),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(expert_dim, a_h),
            nn.ReLU(),
            nn.LayerNorm(a_h),
        )
        merged_dim = s_h + a_h
        layers = []
        prev = merged_dim
        for width in hidden[1:]:
            layers.append(ResidualBlock(prev, width, dropout))
            prev = width
        self.merged = nn.Sequential(*layers)
        self.q_head = nn.Sequential(
            nn.Linear(prev, max(prev // 2, 32)),
            nn.ReLU(),
            nn.LayerNorm(max(prev // 2, 32)),
            nn.Linear(max(prev // 2, 32), 1),
        )

    def forward(self, state, action):
        state = self.encode_state(state)
        z = torch.cat([self.state_encoder(state), self.action_encoder(action)], dim=1)
        return self.q_head(self.merged(z))


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

