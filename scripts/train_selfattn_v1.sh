#!/usr/bin/env bash
# Train and evaluate Self-Attention re-ranker with TCT-ColBERT-v1 features
# (to add to Table 1, which uses v1 features for all GNN baselines)
#
# Training matches the original paper setup:
#   - 1000 training queries (length_train=1000)
#   - fast_train=True using data/msmarco_data/train_data_fast/ (TCT-v1 tensors)
#   - single-stage only (modality=single)
#
# Usage:
#   bash scripts/train_selfattn_v1.sh
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
PY="${PYTHON:-$(conda info --base)/envs/GNRR/bin/python}"

M="models/msmarco_data"
unset EMB
EMB="tctcolbert"   # TCT-ColBERT-v1 — force, do not inherit from environment
SEEDS=3; PATIENCE=7; LENGTH_TRAIN=1000
HIDDEN=128; NLAYERS=2; DROPOUT=0.1; LR=0.01; WD=0.0; HEADS=1

# --- Single-stage ---
CKPT_SINGLE="${M}/hadamard_${NLAYERS}_789_${LR}_${WD}_${HIDDEN}_${DROPOUT}_transformer_single_1_${EMB}.pt"
if [ ! -f "${CKPT_SINGLE}" ]; then
    echo "=== Training Self-Attention single-stage (TCT-v1) ==="
    "${PY}" main.py \
        --dataset_name msmarco_data \
        --conv_type transformer \
        --modality single \
        --hidden_dim "${HIDDEN}" \
        --n_layers "${NLAYERS}" \
        --heads "${HEADS}" \
        --dropout_prob "${DROPOUT}" \
        --lr "${LR}" \
        --wd "${WD}" \
        --embedding_name "${EMB}" \
        --fast_train True \
        --length_train "${LENGTH_TRAIN}" \
        --save_best_model True \
        --n_seeds "${SEEDS}" \
        --patience "${PATIENCE}" \
        --loss_type lambdarank \
        --device cuda
else
    echo "skip (exists): ${CKPT_SINGLE}"
fi

# --- Evaluate on DL19, DL20, DLHard ---
echo "=== Evaluating on test sets ==="
for DS in dl19 dl20 dlhard; do
    "${PY}" scripts/evaluate_testset.py \
        --dataset "${DS}" \
        --pipeline gnrr \
        --conv_type transformer \
        --modality single \
        --embedding_name "${EMB}" \
        --hidden_dim "${HIDDEN}" \
        --n_layers "${NLAYERS}" \
        --heads "${HEADS}" \
        --model_path "${CKPT_SINGLE}"
done

echo "Done. Add the printed numbers to the Self-Attention row of Table 1."
