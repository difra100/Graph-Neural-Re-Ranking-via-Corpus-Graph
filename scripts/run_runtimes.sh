#!/usr/bin/env bash
# Per-query inference runtime for ALL models, single-stage vs multi-stage (DL19).
# Writes results/efficiency/runtimes.csv (read by scripts/make_report.py).
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_DIR}"
PY="${PYTHON:-$(conda info --base)/envs/GNRR/bin/python}"
M="models/msmarco_data"; EMB=tctcolbert2
OUT="results/efficiency"; mkdir -p "${OUT}"; CSV="${OUT}/runtimes.csv"
echo "model,params,ms_single,ms_multi,mem_single_MB,mem_multi_MB" > "${CSV}"

# label|conv|modality|graph|hidden|nlayers|heads|checkpoint
SPECS=(
  "GCN (lexical)|gcn|local|lexical|128|2|1|${M}/hadamard_2_789_0.01_0.0_128_0.6_gcn_local_1_tctcolbert2_lexical.pt"
  "GraphSAGE (lexical)|sage|local|lexical|64|1|1|${M}/hadamard_1_789_0.01_0.0_64_0.1_sage_local_1_tctcolbert2_lexical.pt"
  "GAT (lexical)|gat|local|lexical|128|1|1|${M}/hadamard_1_789_0.01_0.0_128_0.3_gat_local_1_tctcolbert2_1_lexical.pt"
  "GIN (lexical)|gin|local|lexical|64|2|1|${M}/hadamard_2_789_0.01_0.0_64_0.1_gin_local_1_tctcolbert2_lexical.pt"
  "SignedConv (lexical)|signed|local|lexical|64|2|1|${M}/hadamard_2_789_0.01_0.0_64_0.1_signed_local_1_tctcolbert2_1_lexical.pt"
  "EdgeGAT (5L)|edgegat|local|lexical|128|5|1|${M}/hadamard_5_789_0.01_0.0_128_0.1_edgegat_local_1_tctcolbert2_lexical.pt"
  "Self-attention|transformer|single|semantic|128|2|4|${M}/hadamard_2_789_0.01_0.0_128_0.1_transformer_single_1_tctcolbert2.pt"
)

run1() {  # args: conv modality graph hidden nl heads ckpt restrict_k -> prints "params total_ms mem_MB"
    "${PY}" -u scripts/run_efficiency.py --dataset dl19 --embedding_name "${EMB}" \
        --conv_type "$1" --modality "$2" --graph_type "$3" --hidden_dim "$4" \
        --n_layers "$5" --heads "$6" --model_path "$7" --restrict_k "$8" 2>/dev/null \
      | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['model_params'],d['online_total_ms']['mean'],d['peak_gpu_mem_MB'])" 2>/dev/null \
      || echo "NA NA NA"
}

for spec in "${SPECS[@]}"; do
    IFS='|' read -r label conv mod graph hid nl heads ckpt <<< "${spec}"
    [ -e "${ckpt}" ] || { echo "${label},NA,NA,NA,NA,NA" >> "${CSV}"; continue; }
    echo ">> timing ${label}"
    read -r p s_ms s_mem <<< "$(run1 "${conv}" "${mod}" "${graph}" "${hid}" "${nl}" "${heads}" "${ckpt}" 0)"
    read -r _ m_ms m_mem <<< "$(run1 "${conv}" "${mod}" "${graph}" "${hid}" "${nl}" "${heads}" "${ckpt}" 200)"
    echo "${label},${p},${s_ms},${m_ms},${s_mem},${m_mem}" >> "${CSV}"
done

echo ""; echo "=== runtimes.csv ==="; column -t -s, "${CSV}"
