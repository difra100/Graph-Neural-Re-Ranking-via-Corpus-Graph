#!/usr/bin/env bash
# GAR (recall) + GNRR (re-rank) analysis on DL19/DL20/DLHard.
# Rows: GAR-alone (recall baseline) and GAR + {gat-lexical, signed-lexical, edgegat-semantic}.
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
CONDA_BASE="$(conda info --base)"
PY="${PYTHON:-${CONDA_BASE}/envs/GNRR/bin/python}"
M="models/msmarco_data"; EMB=tctcolbert   # TCT-ColBERT-v1 — use v1 for inference consistency
DATASETS=(dl19 dl20 dlhard)

g() { "${PY}" -u scripts/run_gar_plus.py --embedding_name "${EMB}" "$@" || echo "[FAIL] $*"; }

for ds in "${DATASETS[@]}"; do
    echo "################# ${ds} #################"
    # recall baseline (GAR alone, semantic graph = best recall)
    g --dataset "${ds}" --gar_only --graph_type semantic --tag GAR-semantic
    # GAR + my method (graph_type matches the trained model's graph)
    g --dataset "${ds}" --graph_type lexical  --conv_type gat    --modality local \
      --hidden_dim 128 --n_layers 1 --heads 1 \
      --model_path "${M}/hadamard_1_789_0.01_0.0_128_0.3_gat_local_1_tctcolbert2_1_lexical.pt" \
      --tag "GAR+gat-lexical"
    g --dataset "${ds}" --graph_type lexical  --conv_type signed --modality local \
      --hidden_dim 64 --n_layers 2 \
      --model_path "${M}/hadamard_2_789_0.01_0.0_64_0.1_signed_local_1_tctcolbert2_1_lexical.pt" \
      --tag "GAR+signed-lexical"
    g --dataset "${ds}" --graph_type semantic --conv_type edgegat --modality local \
      --hidden_dim 128 --n_layers 5 --heads 1 \
      --model_path "${M}/hadamard_5_789_0.01_0.0_128_0.1_edgegat_local_1_tctcolbert2.pt" \
      --tag "GAR+edgegat-semantic"
done
echo "[done] GAR analysis"
