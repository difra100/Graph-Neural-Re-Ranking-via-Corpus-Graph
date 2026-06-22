#!/usr/bin/env bash
# GAR baseline (MacAvaney et al., ref [18]) on DL19/DL20/DLHard. Inference-only.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
CONDA_BASE="$(conda info --base)"
PY="${PYTHON:-${CONDA_BASE}/envs/GNRR/bin/python}"
EMB="${EMB:-tctcolbert}"; BUDGET="${BUDGET:-1000}"

for ds in dl19 dl20 dlhard; do
    echo "############ GAR on ${ds} ############"
    ${PY} scripts/run_gar.py --dataset "${ds}" --embedding_name "${EMB}" --budget "${BUDGET}"
done
