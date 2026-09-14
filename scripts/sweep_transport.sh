#!/usr/bin/env bash
# Hyperparameter sweep for PhysicsTransportGNN.
#
# Grid: n_steps × lr  (most impactful axes for the Euler-transport model)
# Completed runs are skipped on re-run (checkpoint file already present).
# Results are written to results/transport_sweep/summary.tsv.
#
# Usage:
#   bash scripts/sweep_transport.sh
#   DEVICE=cpu bash scripts/sweep_transport.sh
#   N_STEPS_LIST="4 8" bash scripts/sweep_transport.sh   # partial grid

set -uo pipefail

DEVICE="${DEVICE:-cuda}"
DROPOUT=0.1
WD=1e-5
PATIENCE=20
EPOCHS=200
GRAPH_TYPE=semantic
N_STEPS_LIST="${N_STEPS_LIST:-4 6 8}"
LR_LIST="${LR_LIST:-1e-3 3e-4 1e-4}"

RESULTS_DIR="results/transport_sweep"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/summary.tsv"

# Write header only if file doesn't exist yet (partial re-runs append rows)
[ -f "${SUMMARY}" ] || printf "n_steps\tlr\tbest_ndcg10\tcheckpoint\n" > "${SUMMARY}"

# ---------------------------------------------------------------------------
# Checkpoint path formula — must match main.py exactly.
#
# main.py non-sweep path (modality != 'global'):
#   exp_name = f"{aggr}_{n_layers}_{seed}_{lr}_{wd}_{hidden_dim}_{dropout}
#              _{conv_type}_{modality}_{n_layers_mlp}_{embedding_name}"
#   + f"_{K_multistage}"      (because modality == 'multistage', default K=100)
#   + f"_ns{n_steps}"         (transport-specific tag added in main.py)
#
# Fixed values:
#   aggr=hadamard  n_layers=2  seed=789 (config default)
#   hidden_dim=128  n_layers_mlp=1  modality=local
#   embedding_name=tctcolbert
# ---------------------------------------------------------------------------
py_float() {
    python3 -c "print(float('$1'))"
}

for N_STEPS in ${N_STEPS_LIST}; do
  for LR in ${LR_LIST}; do

    PY_LR=$(py_float "${LR}")   # bash "3e-4" → Python repr "0.0003"
    PY_WD=$(py_float "${WD}")   # "1e-5" → "1e-05"

    EXP_NAME="hadamard_2_789_${PY_LR}_${PY_WD}_128_${DROPOUT}_transport_local_1_tctcolbert_ns${N_STEPS}"
    CKPT="models/msmarco_data/${EXP_NAME}.pt"
    LOG="${RESULTS_DIR}/n${N_STEPS}_lr${LR}.log"

    if [ -f "${CKPT}" ]; then
        echo "[skip] n_steps=${N_STEPS} lr=${LR}  (${CKPT})"
        continue
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " n_steps=${N_STEPS}  lr=${LR}  dropout=${DROPOUT}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python main.py \
      --dataset_name    msmarco_data \
      --conv_type       transport \
      --embedding_name  tctcolbert \
      --n_steps         "${N_STEPS}" \
      --dropout_prob    "${DROPOUT}" \
      --aggr            hadamard \
      --modality        local \
      --graph_type      "${GRAPH_TYPE}" \
      --loss_type       lambdarank \
      --lr              "${LR}" \
      --wd              "${WD}" \
      --fast_train      True \
      --patience        "${PATIENCE}" \
      --epochs          "${EPOCHS}" \
      --device          "${DEVICE}" \
      --save_best_model True \
      --wb              False \
      2>&1 | tee "${LOG}"

    # Extract best nDCG@10 from _LiveSave lines:
    #   "[checkpoint] model saved (nDCG@10=X.XXXX) → path"
    BEST=$(grep -oP 'nDCG@10=\K[0-9.]+' "${LOG}" 2>/dev/null | tail -1)
    [ -z "${BEST}" ] && BEST="0.0000"

    printf "%s\t%s\t%s\t%s\n" "${N_STEPS}" "${LR}" "${BEST}" "${CKPT}" \
        >> "${SUMMARY}"

    echo "[done] n_steps=${N_STEPS} lr=${LR} → best nDCG@10=${BEST}"

  done
done

# ---------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Sweep complete — top-3 by nDCG@10"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -n +2 "${SUMMARY}" | sort -t$'\t' -k3 -rn | head -3 \
    | awk -F'\t' '{printf "  n_steps=%-3s  lr=%-6s  nDCG@10=%s\n", $1, $2, $3}'

echo ""
BEST_ROW=$(tail -n +2 "${SUMMARY}" | sort -t$'\t' -k3 -rn | head -1)
BEST_NS=$(echo   "${BEST_ROW}" | cut -f1)
BEST_LR=$(echo   "${BEST_ROW}" | cut -f2)
BEST_CKPT=$(echo "${BEST_ROW}" | cut -f4)
echo "Best config: --n_steps ${BEST_NS} --lr ${BEST_LR}"
echo "Checkpoint : ${BEST_CKPT}"
echo ""
echo "Evaluate best checkpoint on DL19:"
echo "  python scripts/evaluate_testset.py \\"
echo "    --conv_type transport --n_steps ${BEST_NS} \\"
echo "    --model_path ${BEST_CKPT} --dataset dl19"
