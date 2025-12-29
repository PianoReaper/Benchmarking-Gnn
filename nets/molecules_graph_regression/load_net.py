"""
    Utility file to select GraphNN model.
    Optimized for GCN and ZINC regression.
"""

from nets.molecules_graph_regression.gcn_net import GCNNet

def GCN(net_params):
    return GCNNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    """
    Selects the GNN model.
    Only GCN is kept to avoid import errors with deleted files.
    """
    models = {
        'GCN': GCN
    }

    if MODEL_NAME not in models:
        raise ValueError(f"Model '{MODEL_NAME}' is not supported in this minimal version. Please use 'GCN'.")

    return models[MODEL_NAME](net_params)