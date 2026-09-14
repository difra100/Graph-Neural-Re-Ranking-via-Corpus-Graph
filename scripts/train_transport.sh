#!/usr/bin/env bash
# Train PhysicsTransportGNN on the full MS MARCO training set.
#
# z_i* is the direct ranking score — no downstream MLP.
# Physics parameters (λ, α, γ) are learned jointly with potential_net / mass_init
# via LambdaRank on the full TCT-ColBERT semantic-graph corpus.
#
# Usage:
#   bash scripts/train_transport.sh
#   N_STEPS=8 LR=1e-3 bash scripts/train_transport.sh
#
# To sweep over n_steps and lr use:
#   bash scripts/sweep_transport.sh

set -euo pipefail

N_STEPS="${N_STEPS:-6}"
DROPOUT="${DROPOUT:-0.1}"
GRAPH_TYPE="${GRAPH_TYPE:-semantic}"
LR="${LR:-3e-4}"
WD="${WD:-1e-5}"
PATIENCE="${PATIENCE:-20}"
EPOCHS="${EPOCHS:-200}"
DEVICE="${DEVICE:-cuda}"

echo "=== PhysicsTransportGNN training (full dataset) ==="
echo "  n_steps=${N_STEPS}  lr=${LR}  dropout=${DROPOUT}"
echo "  graph_type=${GRAPH_TYPE}  device=${DEVICE}"

python main.py \
  --dataset_name msmarco_data \
  --conv_type transport \
  --embedding_name tctcolbert \
  --n_steps "${N_STEPS}" \
  --dropout_prob "${DROPOUT}" \
  --aggr hadamard \
  --modality local \
  --graph_type "${GRAPH_TYPE}" \
  --loss_type lambdarank \
  --lr "${LR}" \
  --wd "${WD}" \
  --fast_train True \
  --patience "${PATIENCE}" \
  --epochs "${EPOCHS}" \
  --device "${DEVICE}" \
  --save_best_model True \
  --wb False
