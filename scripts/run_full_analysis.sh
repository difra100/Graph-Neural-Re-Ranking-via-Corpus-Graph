#!/usr/bin/env bash
# Full analysis: every trained model on DL19/DL20/DLHard, in TWO protocols:
#   - single-stage : model re-ranks the full BM25 top-1000        (tag: <name>)
#   - multi-stage  : model re-ranks the TCT top-200, rest backfilled (tag: <name>-ms200)
# Plus BM25 and TCT-ColBERT-v2 reference rows. All models use tctcolbert2 embeddings.
# Results -> results/benchmarks/<tag>/<dataset>/; tables via scripts/collect_analysis.py.
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
CONDA_BASE="$(conda info --base)"
PY="${PYTHON:-${CONDA_BASE}/envs/GNRR/bin/python}"
M="models/msmarco_data"; EMB=tctcolbert2; KMS="${KMS:-200}"
DATASETS=(dl19 dl20 dlhard)

ev() { "${PY}" -u scripts/evaluate_testset.py --embedding_name "${EMB}" "$@" \
       || echo "[FAIL] $*"; }

# name|conv|graph|hidden|nlayers|heads|checkpoint
SPECS=(
  "gcn-semantic|gcn|semantic|128|2|1|${M}/hadamard_2_789_0.01_0.0_128_0.6_gcn_local_1_tctcolbert2.pt"
  "edgegat-semantic|edgegat|semantic|128|5|1|${M}/hadamard_5_789_0.01_0.0_128_0.1_edgegat_local_1_tctcolbert2.pt"
  "gcn-lexical|gcn|lexical|128|2|1|${M}/hadamard_2_789_0.01_0.0_128_0.6_gcn_local_1_tctcolbert2_lexical.pt"
  "sage-lexical|sage|lexical|64|1|1|${M}/hadamard_1_789_0.01_0.0_64_0.1_sage_local_1_tctcolbert2_lexical.pt"
  "gat-lexical|gat|lexical|128|1|1|${M}/hadamard_1_789_0.01_0.0_128_0.3_gat_local_1_tctcolbert2_1_lexical.pt"
  "gin-lexical|gin|lexical|64|2|1|${M}/hadamard_2_789_0.01_0.0_64_0.1_gin_local_1_tctcolbert2_lexical.pt"
  "signed-lexical|signed|lexical|64|2|1|${M}/hadamard_2_789_0.01_0.0_64_0.1_signed_local_1_tctcolbert2_1_lexical.pt"
  "edgegat-lexical|edgegat|lexical|128|5|1|${M}/hadamard_5_789_0.01_0.0_128_0.1_edgegat_local_1_tctcolbert2_lexical.pt"
)
SA="${M}/hadamard_2_789_0.01_0.0_128_0.1_transformer_single_1_tctcolbert2.pt"

for ds in "${DATASETS[@]}"; do
    echo "################# ${ds} #################"
    # --- reference rows (same in both tables) ---
    ev --dataset "${ds}" --pipeline bm25 --tag BM25
    ev --dataset "${ds}" --pipeline tct  --tag TCT-ColBERT-v2

    # --- graph re-rankers: single-stage + multi-stage ---
    for spec in "${SPECS[@]}"; do
        IFS='|' read -r name conv graph hid nl heads ckpt <<< "${spec}"
        [ -e "${ckpt}" ] || { echo "[skip] ${name}: missing ${ckpt}"; continue; }
        common="--dataset ${ds} --pipeline gnrr --conv_type ${conv} --modality local \
                --graph_type ${graph} --hidden_dim ${hid} --n_layers ${nl} --heads ${heads} \
                --model_path ${ckpt}"
        ev ${common} --tag "${name}"
        ev ${common} --restrict_k "${KMS}" --tag "${name}-ms${KMS}"
    done

    # --- self-attention (transformer, full graph) ---
    if [ -e "${SA}" ]; then
        sa="--dataset ${ds} --pipeline gnrr --conv_type transformer --modality single \
            --hidden_dim 128 --n_layers 2 --heads 4 --model_path ${SA}"
        ev ${sa} --tag "selfattn-1stage"
        ev ${sa} --restrict_k "${KMS}" --tag "selfattn-ms${KMS}"
    fi
done

echo ""; echo "[done] building tables ..."
"${PY}" scripts/collect_analysis.py --kms "${KMS}"
