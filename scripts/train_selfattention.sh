#!/usr/bin/env bash
# Train the self-attention re-ranker baselines (TransformerEncoder over TCT features).
# Same training method as the GNNs: LambdaRank + Adam + early stopping on val nDCG@10.
# Self-attention ignores graph edges, so it trains on the fast tensors (no graph needed).
#
#   (1) 1-stage     : attention over BM25 top-1000        (modality=single)
#   (2) multi-stage : BM25 top-1000 -> TCT top-200 -> attn (modality=multistage)
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
CONDA_BASE="$(conda info --base)"
PY="${PYTHON:-${CONDA_BASE}/envs/GNRR/bin/python}"
EMB="${EMB:-tctcolbert}"; N_SEEDS="${N_SEEDS:-3}"; PATIENCE="${PATIENCE:-15}"; EPOCHS="${EPOCHS:-100}"

COMMON="--dataset_name msmarco_data --mode hp --device cuda --conv_type transformer \
  --fast_train True --save_best_model True --n_seeds ${N_SEEDS} \
  --length_train 1000 --length_val 200 --epochs ${EPOCHS} --patience ${PATIENCE} \
  --loss_type lambdarank --lr 0.01 --wd 0 --aggr hadamard --K_cg 8 --score True \
  --embedding_name ${EMB} --hidden_dim 128 --n_layers 2 --dropout_prob 0.1 --heads 4 --n_layers_mlp 1"

echo "############ (1) 1-stage self-attention (top-1000) ############"
${PY} main.py ${COMMON} --modality single

echo "############ (2) multi-stage self-attention (TCT top-200) ############"
${PY} main.py ${COMMON} --modality multistage --K_multistage 200

echo ""
echo "[done] Evaluate the saved transformer checkpoints with scripts/evaluate_testset.py"
echo "  (--pipeline gnrr --conv_type transformer --modality {single,multistage} --heads 4 ...)."
