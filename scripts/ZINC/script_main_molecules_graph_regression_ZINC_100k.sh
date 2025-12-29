#!/bin/bash

############
# ZINC - 4 RUNS SEQUENTIAL
############

seed0=41
seed1=95
seed2=12
seed3=35
code=main_molecules_graph_regression.py
dataset=ZINC

tmux new -s benchmark -d
tmux send-keys "source activate benchmark_gnn" C-m

# Entferne das '&' am Ende und das 'wait'.
# So startet Seed 95 erst, wenn Seed 41 fertig ist.
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_GCN_ZINC_100k.json'
python $code --dataset $dataset --gpu_id 0 --seed $seed1 --config 'configs/molecules_graph_regression_GCN_ZINC_100k.json'
python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/molecules_graph_regression_GCN_ZINC_100k.json'
python $code --dataset $dataset --gpu_id 0 --seed $seed3 --config 'configs/molecules_graph_regression_GCN_ZINC_100k.json'

" C-m
tmux send-keys "tmux kill-session -t benchmark" C-m