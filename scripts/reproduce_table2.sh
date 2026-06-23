#!/usr/bin/env bash
# Reproduce Table 2 — Sensitivity to Candidate Distribution (GAR + GNRR).
#
# Runs GAR (retrieval-stage augmentation) alone and then each of the five
# GNRR re-rankers applied to the GAR-enlarged pool without retraining.
# Both GAR and GNRR re-ranking use the same semantic corpus graph, matching
# the graph type used during GNRR training.
#
# Prerequisites:
#   - conda env GNRR activated
#   - GNRR checkpoints present (produced by bash scripts/train_gnn.sh)
#   - Test graphs present in data/msmarco_data/test_graphs/
#
# Output: results/benchmarks/GAR-semantic/ and results/benchmarks/GAR+{gcn,...}-semantic/
#
# Usage:
#   bash scripts/reproduce_table2.sh
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"

bash "${REPO_DIR}/scripts/run_gar_all.sh"
