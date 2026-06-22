#!/usr/bin/env bash
# GAR (recall) + GNRR re-ranking for all five GNN architectures.
# Uses config_models.json model files; all inference with TCT-ColBERT-v1.
# Results land in results/benchmarks/GAR+{gcn,gat,sage,gin,signed}-semantic/
# and results/benchmarks/GAR-semantic/ (recall-only baseline).
#
# NOTE: GNRR models use --graph_type semantic to match the graph used during
# training and evaluation (reproduce_table1.sh default is semantic).
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
CONDA_BASE="$(conda info --base)"
PY="${PYTHON:-${CONDA_BASE}/envs/GNRR/bin/python}"
EMB="tctcolbert"   # TCT-ColBERT-v1
DATASETS=(dl19 dl20 dlhard)
M="models/msmarco_data"

g() { "${PY}" -u scripts/run_gar_plus.py --embedding_name "${EMB}" "$@" || echo "[FAIL] $*"; }

for ds in "${DATASETS[@]}"; do
    echo "################# ${ds} #################"

    g --dataset "${ds}" --gar_only --graph_type semantic --tag GAR-semantic

    g --dataset "${ds}" --graph_type semantic --conv_type gcn --modality local \
      --hidden_dim 64 --n_layers 2 \
      --model_path "${M}/gcn/bestTrue_gcn_hadamard_2_1_2_0.01_0_64_0.1_local_8_True_tctcolbert.pt" \
      --tag "GAR+gcn-semantic"

    g --dataset "${ds}" --graph_type semantic --conv_type sage --modality local \
      --hidden_dim 64 --n_layers 1 \
      --model_path "${M}/sage/bestTrue_sage_hadamard_1_1_3_0.01_0_64_0.1_local_8_True_tctcolbert_max.pt" \
      --tag "GAR+sage-semantic"

    g --dataset "${ds}" --graph_type semantic --conv_type gat --modality local \
      --hidden_dim 128 --n_layers 1 --heads 1 \
      --model_path "${M}/gat/bestTrue_gat_patched.pt" \
      --tag "GAR+gat-semantic"

    g --dataset "${ds}" --graph_type semantic --conv_type gin --modality local \
      --hidden_dim 64 --n_layers 2 \
      --model_path "${M}/gin/bestTrue_gin_hadamard_2_1_3_0.01_0_64_0.1_local_8_True_tctcolbert.pt" \
      --tag "GAR+gin-semantic"

    g --dataset "${ds}" --graph_type semantic --conv_type signed --modality local \
      --hidden_dim 64 --n_layers 2 \
      --model_path "${M}/signed/bestTrue_signed_hadamard_2_1_2_0.01_0_64_0.1_local_8_True_tctcolbert_1.pt" \
      --tag "GAR+signed-semantic"
done

echo "[done] GAR+GNRR analysis — all five architectures"
