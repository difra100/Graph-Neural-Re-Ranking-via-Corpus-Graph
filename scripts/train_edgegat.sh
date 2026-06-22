#!/usr/bin/env bash
# Train the Edge-GAT query-aware re-ranker (src/edge_gat_reranker.py):
#   encoder -> 5 x EdgeGATConv (query injected as edge features) + norm + residual -> decoder.
# No pooling (node-level re-ranking).
#
# Trains the full grid:  {semantic, lexical} corpus graph  x  N seeds.
# (main.py loops the seeds internally via --n_seeds, so each graph_type call does N seeds.)
#
# Training method: LambdaRank + Adam, up to 100 epochs, EARLY STOPPING on val nDCG@10.
# Node features = frozen TCT-ColBERT doc embeddings; only the graph STRUCTURE differs
# between semantic (TCT kNN) and lexical (BM25 kNN, corpusgraph_bm25_k16) runs.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
CONDA_BASE="$(conda info --base)"
PY="${PYTHON:-${CONDA_BASE}/envs/GNRR/bin/python}"

EMB="${EMB:-tctcolbert}"
N_SEEDS="${N_SEEDS:-3}"
PATIENCE="${PATIENCE:-15}"
EPOCHS="${EPOCHS:-100}"
HIDDEN="${HIDDEN:-128}"
HEADS="${HEADS:-1}"
N_LAYERS="${N_LAYERS:-5}"
DROPOUT="${DROPOUT:-0.1}"
NORM="${NORM:-False}"            # False = LayerNorm (robust to small subgraphs), True = BatchNorm

COMMON="--dataset_name msmarco_data --mode hp --device cuda --conv_type edgegat \
  --modality local --save_best_model True --n_seeds ${N_SEEDS} \
  --length_train 1000 --length_val 200 --epochs ${EPOCHS} --patience ${PATIENCE} \
  --loss_type lambdarank --lr 0.01 --wd 0 --aggr hadamard --K_cg 8 --score False \
  --embedding_name ${EMB} --hidden_dim ${HIDDEN} --n_layers ${N_LAYERS} \
  --heads ${HEADS} --dropout_prob ${DROPOUT} --norm ${NORM} --corpusgraph_name corpusgraph_bm25_k16"

for GT in semantic lexical; do
    # EdgeGAT uses the corpus-graph EDGE WEIGHTS (cosine for semantic, BM25 for lexical).
    # The cached fast tensors only store a *binary* adjacency, so we train in slow mode
    # for BOTH graph types to get real weights (semantic=cosine, lexical=BM25).
    echo "############ Edge-GAT | graph=${GT} | ${N_SEEDS} seeds | L=${N_LAYERS} ############"
    ${PY} main.py ${COMMON} --graph_type "${GT}" --fast_train False
done

echo ""
echo "[done] checkpoints under models/msmarco_data/ (lexical ones suffixed '_lexical')."
echo "Evaluate, e.g.:"
echo "  ${PY} scripts/evaluate_testset.py --dataset dl19 --pipeline gnrr --conv_type edgegat \\"
echo "      --modality local --graph_type lexical --heads ${HEADS} --hidden_dim ${HIDDEN} \\"
echo "      --n_layers ${N_LAYERS} --embedding_name ${EMB} --model_path models/msmarco_data/<exp>.pt"
