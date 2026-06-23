#!/usr/bin/env bash
# Significance testing: all Table 1 and Table 2 systems vs TCT-ColBERT.
# Uses pt.Experiment() with Bonferroni-corrected paired t-tests on saved TREC runs.
#
# Prerequisites:
#   bash scripts/reproduce_table1.sh   (produces TCT-ColBERT and GNRR-*-local runs)
#   bash scripts/reproduce_table2.sh   (produces GAR-semantic and GAR+*-semantic runs)
#
# Results are saved to results/significance/table{1,2}_{dl19,dl20,dlhard}.csv
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
CONDA_BASE="$(conda info --base)"
PY="${PYTHON:-${CONDA_BASE}/envs/GNRR/bin/python}"

TABLE="${TABLE:-both}"   # 1 | 2 | both

"${PY}" scripts/significance.py \
    --table  "${TABLE}" \
    --dataset all \
    --out_dir results/benchmarks \
    --save    results/significance
