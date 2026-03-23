import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = int(max_size)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.s = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.a = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
        self.r = np.zeros((self.max_size, 1), dtype=np.float32)
        self.s2 = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.d = np.zeros((self.max_size, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.absorbing_state = np.zeros((self.state_dim,), dtype=np.float32)

    def add(self, s, a, r, s2=None, d=1.0):
        i = self.ptr
        self.s[i] = np.asarray(s, dtype=np.float32)
        self.a[i] = np.asarray(a, dtype=np.float32)
        self.r[i,0] = float(r)
        if s2 is None:
            self.s2[i] = self.absorbing_state
        else:
            self.s2[i] = np.asarray(s2, dtype=np.float32)
        self.d[i,0] = float(d)
        self.ptr = (self.ptr + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("Sampling from empty buffer")
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.s[idx].copy(), self.a[idx].copy(), self.r[idx].copy(), self.s2[idx].copy(), self.d[idx].copy()

    def __len__(self):
        return self.size

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out)
        self.ln = nn.LayerNorm(dim_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.use_res = (dim_in == dim_out)

    def forward(self, x):
        y = self.fc(x)
        y = self.ln(y)
        y = self.relu(y)
        y = self.dropout(y)
        if self.use_res:
            return x + y
        return y

# ==========================================
# 新增：特征注意力模块 (Feature Attention Module)
# ==========================================
class FeatureAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 对应论文中的 W_att，不加 bias 或者使用默认 bias 均可
        self.attention_weights = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # 1. 计算注意力权重 alpha_i，使用 softmax 保证权重之和为 1
        weights = F.softmax(self.attention_weights(x), dim=-1)
        # 2. 逐元素相乘，返回加权后的特征向量
        weighted_x = weights * x
        return weighted_x, weights

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(128,64), dropout=0.15):
        super().__init__()
        # 初始化注意力模块
        self.attention = FeatureAttention(state_dim)
        
        hs = list(hidden) if isinstance(hidden, (tuple, list)) else [hidden]
        layers = []
        prev = state_dim
        for h in hs:
            layers.append(ResidualBlock(prev, h, dropout=dropout))
            prev = h
        self.shared = nn.Sequential(*layers)
        self.post = nn.Sequential(
            nn.Linear(prev, max(prev // 2, action_dim)),
            nn.ReLU(),
            nn.LayerNorm(max(prev // 2, action_dim))
        )
        self.out = nn.Linear(max(prev // 2, action_dim), action_dim)

    def forward(self, x):
        # 首先让输入状态经过注意力模块进行特征加权
        x, att_weights = self.attention(x)
        
        h = self.shared(x)
        h = self.post(h)
        logits = self.out(h)
        # numerical stability: subtract max before softmax
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        weights = F.softmax(logits, dim=1)
        return weights

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(256,128), dropout=0.15):
        super().__init__()
        # 初始化注意力模块
        self.attention = FeatureAttention(state_dim)
        
        s_h = hidden[0] if isinstance(hidden, (tuple, list)) and len(hidden) > 0 else hidden
        a_h = max(action_dim, 32)
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, s_h),
            nn.ReLU(),
            nn.LayerNorm(s_h),
            nn.Dropout(dropout),
            nn.Linear(s_h, s_h),
            nn.ReLU(),
            nn.LayerNorm(s_h)
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, a_h),
            nn.ReLU(),
            nn.LayerNorm(a_h)
        )
        merged_dim = s_h + a_h
        rb_layers = []
        prev = merged_dim
        for h in (hidden[0], hidden[1]) if isinstance(hidden, (tuple, list)) and len(hidden) > 1 else [merged_dim]:
            if h is None:
                continue
            rb_layers.append(ResidualBlock(prev, h, dropout=dropout))
            prev = h
        self.merged = nn.Sequential(*rb_layers)
        self.q_head = nn.Sequential(
            nn.Linear(prev, max(prev//2, 32)),
            nn.ReLU(),
            nn.LayerNorm(max(prev//2, 32)),
            nn.Linear(max(prev//2, 32), 1)
        )

    def forward(self, s, a):
        # 同样，Critic 的状态输入也经过注意力模块加权
        s, att_weights = self.attention(s)
        
        s_enc = self.state_encoder(s)
        a_enc = self.action_encoder(a)
        x = torch.cat([s_enc, a_enc], dim=1)
        x = self.merged(x) if len(self.merged) > 0 else x
        return self.q_head(x)

class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(size, dtype=np.float32)
        
    def reset(self):
        self.state[:] = 0.0
        
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(*self.state.shape)
        self.state = self.state + dx
        return self.state

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(tau * s.data + (1.0 - tau) * t.data)