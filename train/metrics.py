import torch
import torch.nn.functional as F

"""
    Specialized Metrics for ZINC Graph Regression
    Optimized for GPU usage.
"""

def MAE(scores, targets):
    """
    Mean Absolute Error (MAE) für ZINC.
    Berechnet den L1-Verlust zwischen Vorhersage und Zielwert.
    """
    # l1_loss ist der Standard für MAE in PyTorch
    # Funktioniert direkt auf der GPU ohne .cpu() Aufrufe
    loss = F.l1_loss(scores, targets)

    return loss.detach().item()

# Wir behalten nur MAE, da GCN bei ZINC
# ausschließlich über diesen Wert evaluiert wird.