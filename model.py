import torch
import torch.nn.functional as F
from torch import nn



def body_layer(dim1, dim2):
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ReLU(),
    )


def network_body(obs_dim, hidden_dim=[64], n_layers=2):
    if len(hidden_dim) == 1:
        layer_dims = [(obs_dim, hidden_dim[0])] + [(hidden_dim[0], hidden_dim[0])]*(n_layers - 1)
    else:
        dims = [obs_dim] + hidden_dim
        layer_dims = [(dims[i], dims[i+1]) for i in range(len(dims)-1)]
    return [body_layer(dim1, dim2) for (dim1, dim2) in layer_dims]


class Network(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=[64], n_layers=2):
        super(Network, self).__init__()
        self.body = nn.Sequential(*network_body(obs_dim, hidden_dim, n_layers))
    
        self.actor = nn.Linear(hidden_dim[-1], n_actions)
        self.critic = nn.Linear(hidden_dim[-1], 1)

    def forward(self, x):
        x = self.body(x)
        logits = self.actor(x)
        state_value = self.critic(x)
        return F.softmax(logits, dim=-1), state_value