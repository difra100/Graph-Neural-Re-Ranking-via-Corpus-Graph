#!/usr/bin/env bash
# Graph controls (original / empty / random edges) + subgraph density, for a trained
# GNRR model on all three test sets. Set MODEL_PATH to the checkpoint to probe.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
CONDA_BASE="$(conda info --base)"
PY="${PYTHON:-${CONDA_BASE}/envs/GNRR/bin/python}"
CONV="${CONV:-gcn}"; MODALITY="${MODALITY:-multistage}"; EMB="${EMB:-tctcolbert}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH=models/msmarco_data/...pt}"

for ds in dl19 dl20 dlhard; do
    echo "############ controls on ${ds} ############"
    ${PY} scripts/run_controls.py --dataset "${ds}" --conv_type "${CONV}" \
        --modality "${MODALITY}" --embedding_name "${EMB}" --model_path "${MODEL_PATH}"
done
