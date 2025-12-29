import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, ICLR 2017

    Optimized for modern DGL and GPU execution.
"""

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual

        # Residual connections only work if dimensions match
        if in_dim != out_dim:
            self.residual = False

        # We use the built-in GraphConv which is highly optimized for CUDA.
        # allow_zero_in_degree=True is crucial for stability with some molecule graphs.
        self.conv = GraphConv(in_dim, out_dim, activation=None, allow_zero_in_degree=True)

        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim)

        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, feature):
        h_in = feature   # backup for residual connection

        # 1. Graph Convolution (highly optimized CUDA kernel)
        h = self.conv(g, feature)

        # 2. Batch Normalization
        if self.batch_norm:
            h = self.batchnorm_h(h)

        # 3. Activation
        if self.activation:
            h = self.activation(h)

        # 4. Residual connection
        if self.residual:
            h = h_in + h

        # 5. Dropout
        h = self.dropout(h)

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.residual
        )