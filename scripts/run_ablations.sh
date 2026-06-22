#!/usr/bin/env bash
# Component ablations realigned to GNRR's actual contribution (reviewer request).
# Trains GCN variants on MS MARCO sweeping ONE component at a time:
#   - f1 interaction      : hadamard | sum | concat
#   - GNN depth L         : 1 | 2 | 3
#   - corpus cardinality c: K_cg in 4 | 8 | 16
#   - BM25/score feature  : score True | False
#   - no-GNN control      : disable_gnn True (same capacity, no propagation)
# Evaluate the resulting checkpoints with scripts/evaluate_testset.py (see notes).
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
CONDA_BASE="$(conda info --base)"
PY="${PYTHON:-${CONDA_BASE}/envs/GNRR/bin/python}"
EMB="${EMB:-tctcolbert}"
BASE="--dataset_name msmarco_data --mode hp --n_seeds 1 --fast_train True \
  --save_best_model True --length_train 1000 --length_val 200 --device cuda --patience 20 \
  --loss_type lambdarank --lr 0.01 --wd 0 --embedding_name ${EMB} \
  --conv_type gcn --modality multistage --K_multistage 200 --hidden_dim 128 --n_layers 2 \
  --n_layers_mlp 1 --dropout_prob 0.6"

echo "##### f1 interaction #####"
for f1 in hadamard sum concat; do ${PY} main.py ${BASE} --aggr "${f1}" --score True; done

echo "##### GNN depth #####"
for L in 1 2 3; do ${PY} main.py ${BASE} --aggr hadamard --n_layers "${L}" --score True; done

echo "##### corpus cardinality c #####"
for c in 4 8 16; do ${PY} main.py ${BASE} --aggr hadamard --K_cg "${c}" --score True; done

echo "##### BM25/score feature on/off #####"
for s in True False; do ${PY} main.py ${BASE} --aggr hadamard --score "${s}"; done

echo "##### no-GNN control (local modality) #####"
${PY} main.py ${BASE/--modality multistage/--modality local} --aggr hadamard --score True --disable_gnn True
