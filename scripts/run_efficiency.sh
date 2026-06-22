#!/usr/bin/env bash
# Time + memory complexity for all models (reviewer request).
# Reports, per model: #params, peak GPU memory, per-query subgraph-extraction time,
# GNN/attention forward time, and the offline corpus-graph storage size.
#
# Out of the box it profiles the 5 GNN variants from config_models.json (semantic,
# local). To profile YOUR trained lexical / transformer checkpoints, pass them via
# CONV/MODALITY/GRAPH_TYPE/MODEL_PATH (see the loop / run_efficiency.py --help).
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
CONDA_BASE="$(conda info --base)"
PY="${PYTHON:-${CONDA_BASE}/envs/GNRR/bin/python}"
DS="${DS:-dl19}"; EMB="${EMB:-tctcolbert}"
GRAPH_TYPE="${GRAPH_TYPE:-semantic}"; MODALITY="${MODALITY:-local}"
CONVS=("${CONVS[@]:-gcn sage gat gin signed}")

for cv in ${CONVS[@]}; do
    echo "############ efficiency: ${cv} (${MODALITY}, graph=${GRAPH_TYPE}) ############"
    ${PY} scripts/run_efficiency.py --dataset "${DS}" --conv_type "${cv}" \
        --modality "${MODALITY}" --graph_type "${GRAPH_TYPE}" --embedding_name "${EMB}" \
        || echo "[skip] ${cv} (need a matching trained checkpoint / config)"
done

echo ""
echo "[done] JSON reports in results/efficiency/. For the self-attention model add:"
echo "  ${PY} scripts/run_efficiency.py --dataset ${DS} --conv_type transformer \\"
echo "      --modality single --hidden_dim 128 --n_layers 2 --heads 4 --model_path models/msmarco_data/<exp>.pt"
