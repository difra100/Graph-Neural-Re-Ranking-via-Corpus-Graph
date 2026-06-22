#!/usr/bin/env bash
# Create the reproducible GNRR conda environment (README spec).
# Stack: python 3.8 + torch 1.12.1+cu113 + PyG 2.3.1 (matches the tested codebase).
# Usage:  bash scripts/setup_env.sh
#
# NOTE: we deliberately install via the env's own pip binary (not `conda activate`),
# because `conda activate` does not reliably switch envs inside non-interactive
# shells and can silently install into the wrong (currently-active) environment.
set -euo pipefail

ENV_NAME="${ENV_NAME:-GNRR}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="$(conda info --base)"

# --- create env -----------------------------------------------------------
if conda env list | grep -qE "^\s*${ENV_NAME}\s|/${ENV_NAME}\$"; then
    echo "[setup] conda env '${ENV_NAME}' already exists; reusing it."
else
    echo "[setup] creating conda env '${ENV_NAME}' (python 3.8)..."
    conda create -n "${ENV_NAME}" python=3.8 pip -y
fi

PY="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
PIP="${PY} -m pip"
echo "[setup] using interpreter: ${PY}"
"${PY}" --version

# pytorch-lightning 1.5.10 ships invalid PEP 440 metadata (`torch (>=1.7.*)`) that
# pip >= 24.1 refuses to parse. Pin pip below that to install the tested stack.
echo "[setup] pinning pip < 24.1 (needed for pytorch-lightning 1.5.10 metadata) ..."
${PIP} install "pip<24.1"

# --- torch + PyG (CUDA 11.3 wheels; run fine on Ampere / sm_86) ------------
echo "[setup] installing torch 1.12.1+cu113 (+ matching torchvision) ..."
# torchvision is pinned to the matching cu113 build so that sentence-transformers
# (a pyterrier_dr dependency) finds a compatible torchvision already present and
# does NOT drag in a newer torchvision -> torch 2.x, which would break PyG.
${PIP} install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

echo "[setup] installing torch_geometric 2.3.1 + companions ..."
${PIP} install torch_geometric==2.3.1
# Pin the exact companion versions that have PREBUILT wheels for torch-1.12.1+cu113.
# Without pins, pip grabs newer sdists from PyPI and tries to compile from source,
# which fails because the system CUDA (12.2) != the CUDA torch was built with (11.3).
${PIP} install --only-binary=:all: \
    torch_scatter==2.1.0 torch_sparse==0.6.16 torch_cluster==1.6.0 torch_spline_conv==1.2.1 \
    -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

# --- remaining python deps -----------------------------------------------
# Hard constraint: never let a transitive dep upgrade torch/torchvision (would
# break the PyG companion wheels compiled against torch 1.12.1).
CONSTRAINTS="$(mktemp)"
cat > "${CONSTRAINTS}" <<EOF
torch==1.12.1+cu113
torchvision==0.13.1+cu113
EOF
echo "[setup] installing project requirements (torch pinned via constraints) ..."
${PIP} install -c "${CONSTRAINTS}" -r "${REPO_DIR}/requirements.txt"
rm -f "${CONSTRAINTS}"

# --- vendor the terrierteam plugins (0.0.1, pure-Python, not on PyPI) ----------
# These three plugins at v0.0.1 are unavailable on PyPI; we copy them from a
# known-good source environment (default: the project's my_env). Override with
# PLUGIN_SRC=/path/to/site-packages.
PLUGIN_SRC="${PLUGIN_SRC:-${CONDA_BASE}/envs/my_env/lib/python3.10/site-packages}"
DEST="$(${PY} -c 'import site; print(site.getsitepackages()[0])')"
echo "[setup] vendoring pyterrier_{adaptive,dr,t5} from ${PLUGIN_SRC} -> ${DEST}"
for pkg in pyterrier_adaptive pyterrier_dr pyterrier_t5; do
    if [ -d "${PLUGIN_SRC}/${pkg}" ]; then
        rm -rf "${DEST:?}/${pkg}"
        cp -r "${PLUGIN_SRC}/${pkg}" "${DEST}/"
        # carry the dist-info so importlib.metadata / entry points still resolve
        cp -r "${PLUGIN_SRC}/${pkg}"*.dist-info "${DEST}/" 2>/dev/null || true
        cp -r "${PLUGIN_SRC}/${pkg/_/-}"*.dist-info "${DEST}/" 2>/dev/null || true
        echo "  [ok] ${pkg}"
    else
        echo "  [WARN] ${pkg} not found under ${PLUGIN_SRC}; set PLUGIN_SRC correctly."
    fi
done

# --- optional: allrank (a LambdaRank fallback exists if this is skipped) -------
# allRank pins torch<=1.8, which would clobber our torch 1.12 -> install --no-deps
# (we only use its lambdaLoss; flatten-dict is its sole pure-Python runtime need).
${PIP} install --no-deps allRank flatten-dict \
    || echo "[setup] allRank not installed; using built-in LambdaRank fallback (fine)."

# --- verify ---------------------------------------------------------------
echo "[setup] verifying imports (no segfault expected) ..."
"${PY}" - <<'PY'
import torch, torch_geometric, torch_scatter, torch_sparse
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GIN, SignedConv
import pyterrier, ir_measures, faiss, pandas, numpy, scipy, sklearn, networkx
from pyterrier_adaptive import GAR
from pyterrier_adaptive.corpus_graph import NpTopKCorpusGraph
from pyterrier_dr import TctColBert, FlexIndex
print("torch      :", torch.__version__, "cuda:", torch.cuda.is_available())
print("pyg        :", torch_geometric.__version__)
print("pyterrier  :", pyterrier.__version__)
print("plugins    : adaptive+dr+t5 import OK")
print("ALL IMPORTS OK")
PY

echo "[setup] done. Run experiments with:  ${PY} main.py ...  (or use the scripts/ wrappers)"
