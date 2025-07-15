import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta, Categorical

class ContinuousActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 args,hidden_dims=128):
        super(ContinuousActorNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, hidden_dims)

        self.alpha = nn.Linear(hidden_dims, n_actions)
        self.beta = nn.Linear(hidden_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = args.device
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


class ContinuousCriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, hidden_dims=128, args=None):
        super(ContinuousCriticNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, hidden_dims)
        self.out = nn.Linear(hidden_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = args.device
        self.to(self.device)


    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        x = T.tanh(self.fc3(x))
        v = self.out(x)
        return v

