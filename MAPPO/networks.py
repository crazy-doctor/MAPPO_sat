import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta, Categorical


class ContinuousActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims,args,
                 hidden_dims=256):
        super(ContinuousActorNetwork, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, hidden_dims)
        self.alpha = nn.Linear(hidden_dims, n_actions)
        self.beta = nn.Linear(hidden_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=args.actor_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        x = T.tanh(self.fc3(x))
        alpha = F.softplus(self.alpha(x)) + 1.0
        beta = F.softplus(self.beta(x)) + 1.0

        return alpha, beta

    def get_dist(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist

    # 提供直接获取alpha beta的外部接口
    def get_alpha_beta(self,state):
        return self.forward(state)

class ContinuousCriticNetwork(nn.Module):
    def __init__(self, input_dims, args,
                 hidden_dims=256):
        super(ContinuousCriticNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, hidden_dims)
        self.v = nn.Linear(hidden_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=args.critic_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        x = T.tanh(self.fc3(x))
        v = self.v(x)
        return v