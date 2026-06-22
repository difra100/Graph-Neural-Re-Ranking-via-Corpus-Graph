#!/usr/bin/env bash
# Non-neural graph baselines (score smoothing / kNN interp / PageRank / label prop)
# on DL19/DL20/DLHard. Inference-only.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
CONDA_BASE="$(conda info --base)"
PY="${PYTHON:-${CONDA_BASE}/envs/GNRR/bin/python}"
EMB="${EMB:-tctcolbert}"; ALPHA="${ALPHA:-0.5}"

for ds in dl19 dl20 dlhard; do
    echo "############ graph baselines on ${ds} (alpha=${ALPHA}) ############"
    ${PY} scripts/run_graph_baselines.py --dataset "${ds}" --embedding_name "${EMB}" \
        --alpha "${ALPHA}"
done
