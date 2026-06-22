#!/usr/bin/env bash
# Significance testing: GNRR-GCN vs the key baselines on each test set.
# Assumes evaluate_testset.py / run_gar.py / run_graph_baselines.py have been run
# so per-query CSVs exist under results/benchmarks/.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
CONDA_BASE="$(conda info --base)"
PY="${PYTHON:-${CONDA_BASE}/envs/GNRR/bin/python}"

SYSTEM="${SYSTEM:-GNRR-gcn-multistage}"
BASELINES="${BASELINES:-TCT-ColBERT GAR graph-smoothing-a0.5 graph-ppr-a0.5}"

for ds in dl19 dl20 dlhard; do
    for metric in "AP(rel=2)" "nDCG@10"; do
        echo "############ ${ds}  ${metric} ############"
        ${PY} scripts/significance.py --dataset "${ds}" --metric "${metric}" \
            --system "${SYSTEM}" --baselines ${BASELINES} \
            --save "results/significance/${ds}_${metric//\//_}.csv" || true
    done
done
