import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, state_dim,  name='', out_state_dim=None):
        super(GraphConvolution, self).__init__()
        self.state_dim = state_dim

        if out_state_dim == None:
            self.out_state_dim = state_dim
        else:
            self.out_state_dim = out_state_dim
        self.fc1 = nn.Linear(
            in_features=self.state_dim,
            out_features=self.out_state_dim,
        )

        self.fc2 = nn.Linear(
            in_features=self.state_dim,
            out_features=self.out_state_dim,
        )

        self.name = name
        #self.fc2.weight = nn.Parameter(torch.zeros_like(self.fc2.weight))
        #self.fc2.bias = nn.Parameter(torch.zeros_like(self.fc2.bias))
    def forward(self, input, adj):

        state_in = self.fc1(input)

        forward_input = self.fc2(torch.bmm(adj, input))

        return state_in + forward_input



    def __repr__(self):
        return self.__class__.__name__ + ' (' +  self.name + ')'