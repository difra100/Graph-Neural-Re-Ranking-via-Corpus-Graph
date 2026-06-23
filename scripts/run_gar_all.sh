#!/usr/bin/env bash
# GAR (recall) + GNRR re-ranking for all five GNN architectures.
# Uses the verified paper checkpoints from models/paper_checkpoints/.
# Results land in results/benchmarks/GAR+{gcn,gat,sage,gin,signed}-semantic/
# and results/benchmarks/GAR-semantic/ (recall-only baseline).
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
CONDA_BASE="$(conda info --base)"
PY="${PYTHON:-${CONDA_BASE}/envs/GNRR/bin/python}"
EMB="tctcolbert"
P="models/paper_checkpoints"
DATASETS=(dl19 dl20 dlhard)

g() { "${PY}" -u scripts/run_gar_plus.py --embedding_name "${EMB}" "$@" || echo "[FAIL] $*"; }

for ds in "${DATASETS[@]}"; do
    echo "################# ${ds} #################"

    g --dataset "${ds}" --gar_only --graph_type semantic --tag GAR-semantic

    g --dataset "${ds}" --graph_type semantic --conv_type gcn --modality local \
      --hidden_dim 128 --n_layers 2 --n_layers_mlp 2 \
      --model_path "${P}/gcn.pt" --tag "GAR+gcn-semantic"

    g --dataset "${ds}" --graph_type semantic --conv_type sage --modality local \
      --hidden_dim 64 --n_layers 1 --n_layers_mlp 1 \
      --model_path "${P}/sage.pt" --tag "GAR+sage-semantic"

    g --dataset "${ds}" --graph_type semantic --conv_type gat --modality local \
      --hidden_dim 128 --n_layers 1 --n_layers_mlp 1 --heads 1 \
      --model_path "${P}/gat.pt" --tag "GAR+gat-semantic"

    g --dataset "${ds}" --graph_type semantic --conv_type gin --modality local \
      --hidden_dim 64 --n_layers 2 --n_layers_mlp 1 \
      --model_path "${P}/gin.pt" --tag "GAR+gin-semantic"

    g --dataset "${ds}" --graph_type semantic --conv_type signed --modality local \
      --hidden_dim 64 --n_layers 2 --n_layers_mlp 1 \
      --model_path "${P}/signed.pt" --tag "GAR+signed-semantic"
done

echo "[done] GAR+GNRR — all five architectures with paper checkpoints"
