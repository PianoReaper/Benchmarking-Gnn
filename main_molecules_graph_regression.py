"""
    IMPORTING LIBS
"""
import dgl
import numpy as np
import os
import time
import random
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

# Erzwinge PyTorch Backend für DGL
os.environ['DGLBACKEND'] = 'pytorch'

"""
    IMPORTING CUSTOM MODULES
"""
from nets.molecules_graph_regression.load_net import gnn_model
from data.data import LoadData
from train.train_molecules_graph_regression import train_epoch_sparse, evaluate_network_sparse

"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    if torch.cuda.is_available() and use_gpu:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        print(f'Using GPU: {torch.cuda.get_device_name(device)}')
    else:
        print('Using CPU')
        device = torch.device("cpu")
    return device

def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'MODEL/Total parameters: {MODEL_NAME} {total_param}')
    return total_param

"""
    TRAINING PIPELINE
"""
def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    t0 = time.time()
    per_epoch_time = []
    DATASET_NAME = dataset.name
    root_log_dir, root_ckpt_dir, root_result_dir, root_config_dir = dirs
    device = net_params['device']

    # GCN benötigt Self-Loops für stabile Performance auf ZINC
    if MODEL_NAME == 'GCN':
        net_params['self_loop'] = True
        dataset._add_self_loops()
        print("[!] Added self-loops for GCN.")

    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    # Seeds setzen für Reproduzierbarkeit
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    model = gnn_model(MODEL_NAME, net_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    # DataLoaders
    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

    writer = SummaryWriter(log_dir=os.path.join(root_log_dir, f"RUN_SEED_{params['seed']}"))

    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:
                start = time.time()

                # Nutze die bereinigten Funktionen aus train_molecules_graph_regression.py
                epoch_train_loss, epoch_train_mae, optimizer = train_epoch_sparse(model, optimizer, device, train_loader, epoch)
                epoch_val_loss, epoch_val_mae = evaluate_network_sparse(model, device, val_loader, epoch)
                _, epoch_test_mae = evaluate_network_sparse(model, device, test_loader, epoch)

                scheduler.step(epoch_val_loss)

                # Fortschritt anzeigen
                t.set_postfix(val_MAE=f"{epoch_val_mae:.4f}", test_MAE=f"{epoch_test_mae:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

                # Loggen
                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_mae', epoch_val_mae, epoch)
                writer.add_scalar('test/_mae', epoch_test_mae, epoch)

                # Checkpointing (alle 10 Epochen oder letzte)
                if epoch % 10 == 0 or epoch == params['epochs'] - 1:
                    if not os.path.exists(root_ckpt_dir): os.makedirs(root_ckpt_dir)
                    torch.save(model.state_dict(), f"{root_ckpt_dir}/epoch_{epoch}.pkl")

                if optimizer.param_groups[0]['lr'] < float(params['min_lr']):
                    print("\n!! LR reached MIN LR. Stopping.")
                    break

    except KeyboardInterrupt:
        print('Exiting training early...')

    print(f"Total Time: {time.time()-t0:.2f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="JSON config file")
    parser.add_argument('--model', default='GCN', help="GNN Model name")
    parser.add_argument('--dataset', default='ZINC', help="Dataset name")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID")
    parser.add_argument('--seed', type=int, help="Override seed")
    parser.add_argument('--out_dir', help="Output directory override")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # CLI Overrides (wichtig für die .sh Skripte)
    if args.seed: config['params']['seed'] = args.seed
    if args.dataset: config['dataset'] = args.dataset
    if args.out_dir: config['out_dir'] = args.out_dir

    device = gpu_setup(config['gpu']['use'], args.gpu_id)
    dataset = LoadData(config['dataset'])

    params = config['params']
    net_params = config['net_params']
    net_params['device'] = device
    net_params['num_atom_type'] = dataset.num_atom_type
    net_params['num_bond_type'] = dataset.num_bond_type

    # Verzeichnisse basierend auf Modell und Dataset erstellen
    out_dir = config['out_dir'] + f"{args.model}_{args.dataset}_GPU{args.gpu_id}/"
    dirs = (out_dir+'logs/', out_dir+'checkpoints/', out_dir+'results/', out_dir+'configs/')
    for d in dirs:
        if not os.path.exists(d): os.makedirs(d)

    net_params['total_param'] = view_model_param(args.model, net_params)
    train_val_pipeline(args.model, dataset, params, net_params, dirs)

if __name__ == "__main__":
    main()