import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, ICLR 2017
    Regression version for ZINC molecules.
"""

class GCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        # Parameters from the config file
        num_atom_type = net_params['num_atom_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']

        # Initial atom embedding and dropout
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        # GCN Layers
        self.layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                     self.batch_norm, self.residual)
            for _ in range(n_layers-1)
        ])

        # Final layer to output dimension
        self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu,
                                   dropout, self.batch_norm, self.residual))

        # Readout to a single scalar (for molecule regression)
        self.MLP_layer = MLPReadout(out_dim, 1)

    def forward(self, g, h, e):
        # 1. Atom embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        # 2. Graph Convolutions
        for conv in self.layers:
            h = conv(g, h)

        # 3. Readout (Graph Pooling)
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        else:
            # Default for ZINC is usually 'mean'
            hg = dgl.mean_nodes(g, 'h')

        # 4. Final MLP for regression value
        return self.MLP_layer(hg)

    def loss(self, scores, targets):
        # MAE (L1Loss) is the standard metric for ZINC benchmarking
        loss = nn.L1Loss()(scores, targets)
        return loss