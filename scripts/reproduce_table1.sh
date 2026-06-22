#!/usr/bin/env bash
# Reproduce the paper's Table 1: BM25, TCT-ColBERT, and the 5 GNRR variants
# (GCN/GraphSAGE/GAT/GIN/SignedConv) on DL19, DL20, DLHard.
#
# Usage:  bash scripts/reproduce_table1.sh [MODALITY]
#   MODALITY defaults to "local" (the GNRR variant checkpoints in config_models.json).
#   Pass "multistage" to evaluate the multistage checkpoints instead.
#
# Results land in results/benchmarks/<pipeline>/<dataset>/{aggregate,perquery}.csv
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
CONDA_BASE="$(conda info --base)"
PY="${PYTHON:-${CONDA_BASE}/envs/GNRR/bin/python}"
EMB="${EMB:-tctcolbert}"
MODALITY="${1:-local}"
DATASETS=(dl19 dl20 dlhard)
CONVS=(gcn sage gat gin signed)

run () { echo "+ $*"; "$@"; }

for ds in "${DATASETS[@]}"; do
    # --- baselines ---
    run "${PY}" scripts/evaluate_testset.py --dataset "${ds}" --pipeline bm25
    run "${PY}" scripts/evaluate_testset.py --dataset "${ds}" --pipeline tct \
        --embedding_name "${EMB}"

    # --- GNRR variants (checkpoints + hyperparams come from config_models.json) ---
    for cv in "${CONVS[@]}"; do
        run "${PY}" scripts/evaluate_testset.py --dataset "${ds}" --pipeline gnrr \
            --conv_type "${cv}" --modality "${MODALITY}" --embedding_name "${EMB}"
    done
done

echo ""
echo "[done] aggregating into one table ..."
run "${PY}" scripts/collect_results.py --out_dir results/benchmarks \
    --table_out results/benchmarks/table1.csv
echo "[done] see results/benchmarks/table1.csv"
