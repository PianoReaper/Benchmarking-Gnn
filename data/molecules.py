import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np
import dgl

class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, name):
        """
            Loading ZINC dataset - Minimal & Modern Version
        """
        start = time.time()
        print(f"[I] Loading dataset {name}...")
        self.name = name
        data_dir = 'data/molecules/'

        # Path to the .pkl file (ZINC or ZINC-full)
        file_path = os.path.join(data_dir, f"{name}.pkl")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[idx] File {file_path} not found. Please run the download script first.")

        with open(file_path, "rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
            self.num_atom_type = f[3]
            self.num_bond_type = f[4]

        print(f'[I] Train, test, val sizes: {len(self.train)}, {len(self.test)}, {len(self.val)}')
        print("[I] Finished loading.")
        print(f"[I] Data load time: {time.time() - start:.4f}s")

    def collate(self, samples):
        """
        Clusters individual graphs into a batch for GPU processing.
        """
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)

        # Efficient batching for modern DGL versions
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels

def self_loop(g):
    """
    Modernized Self-Loop function:
    Removes existing loops and adds new ones.
    Maintains node features automatically, crucial for GPU stability.
    """
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g