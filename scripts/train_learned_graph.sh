#!/usr/bin/env bash
# Train LearnedEdgeGATReranker with SLAPS-style auxiliary reconstruction loss.
#
# Why this works (and why the previous run was stuck):
#   MS MARCO has ~1-3 relevant docs per 1000 BM25 candidates.
#   With only ranking loss, edges between irrelevant-irrelevant document pairs
#   receive zero gradient (supervision starvation from SLAPS NeurIPS 2021).
#   The Gumbel selector for those edges trains on noise, not signal.
#
#   Fix: mask mask_ratio fraction of input feature dims (z_q ⊙ z_d), run the GNN
#   with the learned graph, reconstruct the masked dims from the GNN output.
#   This MSE loss propagates gradient to ALL edges regardless of relevance label.
#
# Loss balance:
#   ranking loss     ~ 1e-4   (lambdarank, sparse MS MARCO labels)
#   recon MSE        ~ 0.05   (masked dims of TCT-ColBERT hadamard features)
#   feat_recon_reg   = 0.001  → auxiliary ~ 5e-5  ≈ 50% of ranking (reasonable)
#   sparsity_reg     = 0.0    (no explicit pruning pressure; selector learns from rank+recon)
#
# Tuning:
#   FEAT_RECON_REG=0.01  bash scripts/train_learned_graph.sh  # more auxiliary signal
#   FEAT_RECON_REG=0     bash scripts/train_learned_graph.sh  # ranking only (ablation)
#   SPARSITY_REG=1e-5    bash scripts/train_learned_graph.sh  # add sparsity pressure
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
PY="${PYTHON:-$(conda info --base)/envs/GNRR/bin/python}"

M="models/msmarco_data"; EMB=tctcolbert2
HIDDEN=128; NLAYERS=3; HEADS=1; DROPOUT=0.1; LR=0.01; WD=0.0
SPARSITY_REG="${SPARSITY_REG:-0.0}"
FEAT_RECON_REG="${FEAT_RECON_REG:-0.001}"
MASK_RATIO="${MASK_RATIO:-0.15}"
GUMBEL_TEMP="${GUMBEL_TEMP:-1.0}"
LENGTH_TRAIN="${LENGTH_TRAIN:-1000}"  # all precomputed fast tensors → 117 steps/epoch
SEEDS=3; PATIENCE=7

# Warm-start backbone from matching EdgeGAT-3L checkpoint.
# strict=False: backbone keys load, gumbel_selector stays at fresh init (bias=+1 → keep all).
EDGEGAT_CKPT="${M}/hadamard_${NLAYERS}_789_${LR}_${WD}_${HIDDEN}_${DROPOUT}_edgegat_local_1_${EMB}.pt"
if [ ! -f "${EDGEGAT_CKPT}" ]; then
    echo "WARNING: EdgeGAT warm-start checkpoint not found: ${EDGEGAT_CKPT}"
    echo "         Training from scratch (no --model_path).  Results may be weaker."
    EDGEGAT_CKPT=""
fi

for GRAPH in semantic lexical; do
    echo "=== Training learned_edgegat on ${GRAPH} (λ_dae=${FEAT_RECON_REG}, mask=${MASK_RATIO}, λ_s=${SPARSITY_REG}) ==="
    CKPT="${M}/hadamard_${NLAYERS}_789_${LR}_${WD}_${HIDDEN}_${DROPOUT}_learned_edgegat_local_1_${EMB}"
    [ "${GRAPH}" != "semantic" ] && CKPT="${CKPT}_${GRAPH}"
    CKPT="${CKPT}.pt"

    if [ -f "${CKPT}" ]; then
        echo "  checkpoint exists: ${CKPT}"
        echo "  delete to retrain:  rm ${CKPT}"
        continue
    fi

    MODEL_PATH_ARG=""
    [ -n "${EDGEGAT_CKPT}" ] && MODEL_PATH_ARG="--model_path ${EDGEGAT_CKPT}"

    "${PY}" main.py \
        --dataset_name msmarco_data \
        --conv_type learned_edgegat \
        --modality local \
        --hidden_dim "${HIDDEN}" \
        --n_layers "${NLAYERS}" \
        --heads "${HEADS}" \
        --dropout_prob "${DROPOUT}" \
        --lr "${LR}" \
        --wd "${WD}" \
        --graph_type "${GRAPH}" \
        --gumbel_temp "${GUMBEL_TEMP}" \
        --sparsity_reg "${SPARSITY_REG}" \
        --feat_recon_reg "${FEAT_RECON_REG}" \
        --mask_ratio "${MASK_RATIO}" \
        --embedding_name "${EMB}" \
        --fast_train False \
        --length_train "${LENGTH_TRAIN}" \
        --save_best_model True \
        --n_seeds "${SEEDS}" \
        --patience "${PATIENCE}" \
        --loss_type lambdarank \
        --device cuda \
        ${MODEL_PATH_ARG}

    echo ""
done

echo "=== done — evaluate with: ==="
echo "python scripts/evaluate_testset.py --pipeline gnrr --conv_type learned_edgegat --graph_type semantic ..."
echo ""
echo "Ablation sweeps:"
echo "  FEAT_RECON_REG=0    bash scripts/train_learned_graph.sh  # ranking-only (no DAE)"
echo "  FEAT_RECON_REG=0.01 bash scripts/train_learned_graph.sh  # stronger DAE signal"
echo "  SPARSITY_REG=1e-5   bash scripts/train_learned_graph.sh  # add sparsity pressure"
