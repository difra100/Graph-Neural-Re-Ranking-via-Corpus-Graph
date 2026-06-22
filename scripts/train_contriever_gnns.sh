#!/usr/bin/env bash
# Train standard GNN re-rankers on Contriever features + TCT k16 graph.
#
# Graph stays identical to the TCT experiments (semantic k16 kNN from tctcolbert2).
# Only the node features change: z_q ⊙ z_d from facebook/contriever instead of TCT.
#
# Research question: does GNN improve more over Contriever than over TCT?
# Hypothesis: TCT-ColBERT-v2 is already fine-tuned on MS MARCO → small room for GNN.
#             Contriever is not MS MARCO fine-tuned (~0.60-0.65 DL19) → more room for GNN.
#
# Prerequisite:
#   python scripts/build_contriever_index.py       # ~2-3 hrs
#   python scripts/precompute_contriever_tensors.py # ~2 hrs
#
# Usage:
#   bash scripts/train_contriever_gnns.sh           # train GCN + GAT on contriever features
#   EMB=contriever-msmarco bash scripts/train_contriever_gnns.sh
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
PY="${PYTHON:-$(conda info --base)/envs/GNRR/bin/python}"

EMB="${EMB:-contriever}"   # or contriever-msmarco
M="models/msmarco_data"
SEEDS=3; PATIENCE=7; LENGTH_TRAIN=14994

for CONV in gcn gat edgegat; do
    for NLAYERS in 2 3; do
        case "${CONV}" in
            gcn)    HIDDEN=128; DROPOUT=0.6 ;;
            gat)    HIDDEN=128; DROPOUT=0.1 ;;
            edgegat) HIDDEN=128; DROPOUT=0.1 ;;
        esac
        LR=0.01; WD=0.0; HEADS=1

        CKPT="${M}/hadamard_${NLAYERS}_789_${LR}_${WD}_${HIDDEN}_${DROPOUT}_${CONV}_local_1_${EMB}.pt"
        if [ -f "${CKPT}" ]; then
            echo "skip (exists): ${CKPT}"
            continue
        fi
        echo "=== ${CONV} L=${NLAYERS} on ${EMB} ==="

        "${PY}" main.py \
            --dataset_name msmarco_data \
            --conv_type "${CONV}" \
            --modality local \
            --hidden_dim "${HIDDEN}" \
            --n_layers "${NLAYERS}" \
            --heads "${HEADS}" \
            --dropout_prob "${DROPOUT}" \
            --lr "${LR}" \
            --wd "${WD}" \
            --graph_type semantic \
            --embedding_name "${EMB}" \
            --fast_train True \
            --length_train "${LENGTH_TRAIN}" \
            --save_best_model True \
            --n_seeds "${SEEDS}" \
            --patience "${PATIENCE}" \
            --loss_type lambdarank \
            --device cuda
        echo ""
    done
done

echo "=== Evaluate on DL19 ==="
for CONV in gcn gat edgegat; do
    for NLAYERS in 2 3; do
        case "${CONV}" in
            gcn)    HIDDEN=128; DROPOUT=0.6 ;;
            gat)    HIDDEN=128; DROPOUT=0.1 ;;
            edgegat) HIDDEN=128; DROPOUT=0.1 ;;
        esac
        LR=0.01; WD=0.0
        CKPT="${M}/hadamard_${NLAYERS}_789_${LR}_${WD}_${HIDDEN}_${DROPOUT}_${CONV}_local_1_${EMB}.pt"
        [ -f "${CKPT}" ] || continue
        "${PY}" scripts/evaluate_testset.py \
            --dataset dl19 --pipeline gnrr \
            --conv_type "${CONV}" --modality local \
            --embedding_name "${EMB}" \
            --hidden_dim "${HIDDEN}" --n_layers "${NLAYERS}" --heads 1 \
            --model_path "${CKPT}"
    done
done
