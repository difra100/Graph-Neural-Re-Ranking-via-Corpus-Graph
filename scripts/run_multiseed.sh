#!/usr/bin/env bash
# Multi-seed variance (reviewer request): train GCN (and the no-GNN control) across
# several seeds so mean +/- std can be reported. main.py already loops n_seeds over
# its seed_list (src/config.py); set N_SEEDS to control how many.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
CONDA_BASE="$(conda info --base)"
PY="${PYTHON:-${CONDA_BASE}/envs/GNRR/bin/python}"
EMB="${EMB:-tctcolbert}"; N_SEEDS="${N_SEEDS:-3}"
BASE="--dataset_name msmarco_data --mode hp --fast_train True --save_best_model True \
  --length_train 1000 --length_val 200 --device cuda --patience 20 --loss_type lambdarank \
  --lr 0.01 --wd 0 --embedding_name ${EMB} --aggr hadamard --modality multistage \
  --K_multistage 200 --hidden_dim 128 --n_layers 2 --n_layers_mlp 1 --dropout_prob 0.6 \
  --score True --n_seeds ${N_SEEDS}"

echo "##### GCN x ${N_SEEDS} seeds #####"
${PY} main.py ${BASE} --conv_type gcn

echo "##### no-GNN control x ${N_SEEDS} seeds #####"
${PY} main.py ${BASE/--modality multistage/--modality local} --conv_type gcn --disable_gnn True
