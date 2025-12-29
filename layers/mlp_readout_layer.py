import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation.
    This maps the final graph embedding to the regression value.
"""

class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, L=2):
        # L = number of hidden layers
        super().__init__()
        list_FC_layers = []

        # We create a series of linear layers that gradually reduce the dimension
        for l in range(L):
            list_FC_layers.append(nn.Linear(input_dim // 2**l, input_dim // 2**(l+1), bias=True))

        # Final layer to the output dimension (for ZINC: 1)
        list_FC_layers.append(nn.Linear(input_dim // 2**L, output_dim, bias=True))

        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)

        # Final linear output without activation for regression
        y = self.FC_layers[self.L](y)
        return y