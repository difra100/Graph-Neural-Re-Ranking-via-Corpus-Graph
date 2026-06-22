#!/usr/bin/env bash
# Train the GNRR GNN variants (GCN / GraphSAGE / GAT / GIN / SignedConv).
#
# Graph choice (the new bit):
#   GRAPH_TYPE=lexical  -> BM25 kNN corpus graph (NpTopKCorpusGraph.from_dataset(
#                          'msmarco_passage','corpusgraph_bm25_k16').to_limit_k(K_cg))
#   GRAPH_TYPE=semantic -> original TCT-ColBERT kNN graph (data/msmarco-index_*).
#
# Training method: LambdaRank loss, Adam, up to 100 epochs with EARLY STOPPING on
# validation nDCG@10 (patience), best-checkpoint selection, multi-seed. Node features
# are frozen TCT-ColBERT embeddings either way -- only the graph STRUCTURE changes.
#
# NOTE: lexical training runs in slow mode (--fast_train False) because the cached
# fast tensors bake in the *semantic* adjacency. Semantic training can use the fast
# tensors. Set GRAPH_TYPE before running:  GRAPH_TYPE=lexical bash scripts/train_gnn.sh
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
CONDA_BASE="$(conda info --base)"
PY="${PYTHON:-${CONDA_BASE}/envs/GNRR/bin/python}"

GRAPH_TYPE="${GRAPH_TYPE:-lexical}"
MODALITY="${MODALITY:-local}"      # GNRR-local = full GNN propagation over the subgraph
EMB="${EMB:-tctcolbert}"
N_SEEDS="${N_SEEDS:-3}"
PATIENCE="${PATIENCE:-15}"
EPOCHS="${EPOCHS:-100}"
CONVS=("${CONVS[@]:-gcn sage gat gin signed}")

# fast tensors only exist for the semantic graph; lexical must build subgraphs live.
if [ "${GRAPH_TYPE}" = "semantic" ]; then FAST=True; else FAST=False; fi

COMMON="--dataset_name msmarco_data --mode hp --device cuda \
  --graph_type ${GRAPH_TYPE} --corpusgraph_name corpusgraph_bm25_k16 \
  --fast_train ${FAST} --save_best_model True --n_seeds ${N_SEEDS} \
  --length_train 1000 --length_val 200 --epochs ${EPOCHS} --patience ${PATIENCE} \
  --loss_type lambdarank --lr 0.01 --wd 0 --aggr hadamard --K_cg 8 --score True \
  --embedding_name ${EMB} --modality ${MODALITY}"

hp () {  # per-variant known-good hyperparameters (from config_models.json)
  case "$1" in
    gcn)    echo "--hidden_dim 128 --n_layers 2 --dropout_prob 0.6 --n_layers_mlp 1" ;;
    gat)    echo "--hidden_dim 128 --n_layers 1 --dropout_prob 0.3 --n_layers_mlp 1 --heads 1" ;;
    sage)   echo "--hidden_dim 64  --n_layers 1 --dropout_prob 0.1 --n_layers_mlp 1 --aggr_sage max" ;;
    gin)    echo "--hidden_dim 64  --n_layers 2 --dropout_prob 0.1 --n_layers_mlp 1" ;;
    signed) echo "--hidden_dim 64  --n_layers 2 --dropout_prob 0.1 --n_layers_mlp 1 --negatives 1" ;;
  esac
}

# Resume support: skip a variant if its checkpoint already exists for this graph type.
# Lexical checkpoints end in '_lexical.pt'; semantic ones do not. RESUME=0 to force all.
model_exists () {
  local cv="$1"
  for f in models/msmarco_data/*_${cv}_${MODALITY}_*${EMB}*.pt; do
    [ -e "$f" ] || continue
    case "$f" in
      *_lexical.pt) [ "${GRAPH_TYPE}" != "semantic" ] && { echo "$f"; return; } ;;
      *)            [ "${GRAPH_TYPE}" = "semantic" ] && { echo "$f"; return; } ;;
    esac
  done
}

trained=0
for cv in ${CONVS[@]}; do
    if [ "${RESUME:-1}" = "1" ]; then
        found="$(model_exists "${cv}")"
        if [ -n "${found}" ]; then
            echo "[skip] ${cv} (${GRAPH_TYPE}) already trained -> ${found}"
            continue
        fi
    fi
    echo "############ training ${cv} | graph=${GRAPH_TYPE} | modality=${MODALITY} ############"
    ${PY} main.py ${COMMON} --conv_type "${cv}" $(hp "${cv}")
    trained=$((trained + 1))
done

if [ "${trained}" -eq 0 ]; then
    echo ""
    echo "[nothing to train] all requested variants already have a ${GRAPH_TYPE} checkpoint."
    echo "  - to force a full retrain:        RESUME=0 GRAPH_TYPE=${GRAPH_TYPE} bash scripts/train_gnn.sh"
    echo "  - to retrain specific variants:   RESUME=0 CONVS='gcn gin' GRAPH_TYPE=${GRAPH_TYPE} bash scripts/train_gnn.sh"
    echo "  - or delete the stale checkpoint(s) and rerun."
fi

echo ""
echo "[done] checkpoints saved under models/msmarco_data/. Evaluate with, e.g.:"
echo "  ${PY} scripts/evaluate_testset.py --dataset dl19 --pipeline gnrr --conv_type gcn \\"
echo "      --modality ${MODALITY} --graph_type ${GRAPH_TYPE} --model_path models/msmarco_data/<exp>.pt"
