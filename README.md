# Cross-Document Neural Re-Ranking via Query-Induced Subgraphs

Neural re-rankers typically score query-document pairs independently, discarding relationships among candidates in the retrieved set. This repository contains the code for **GNRR** (Graph Neural Re-Ranking), a framework that extracts a sparse, query-induced subgraph from a pre-computed semantic corpus graph and applies Graph Neural Networks to propagate cross-document signals.

Unlike self-attention re-rankers that scale quadratically in the candidate set size ($O(K^2)$), GNRR achieves $O(c \cdot K)$ online complexity, where $c$ is the fixed corpus graph degree and $K$ the candidate set size.

---

## Pipeline overview

![GNRR pipeline](figures/pipeline_graphical_overview)

**Offline:** TCT-ColBERT encodes the entire corpus once and connects each document to its $c$ nearest semantic neighbors, building corpus graph $\mathcal{G}$. **Online:** BM25 retrieves the top-$K$ candidates; the candidate set induces a sparse subgraph of $\mathcal{G}$; a GNN propagates cross-document signals over that subgraph; a learned scorer produces the final re-ranking.

---

## Key results (TREC benchmarks, TCT-ColBERT features, $K=1000$)

**Table 1 — Re-ranking vs. baselines**

| Pipeline | DL19 AP | DL20 AP | DLHard AP |
|---|---|---|---|
| BM25 | 0.286 | 0.293 | 0.147 |
| +TCT-ColBERT | 0.430 | 0.453 | 0.230 |
| +Self-Attention | 0.431 | 0.454 | 0.222 |
| **GNRR+GCN** | **0.455** | **0.470** | **0.242** |
| GNRR+GraphSAGE | 0.434 | 0.454 | 0.223 |
| GNRR+GAT | 0.441 | 0.453 | 0.219 |
| GNRR+GIN | 0.449 | 0.455 | 0.215 |
| GNRR+SignedConv | 0.426 | 0.445 | 0.218 |

GCN is the only architecture that improves AP on all three benchmarks. On DLHard, GCN achieves **+5.2% AP** over TCT-ColBERT and **+9.0% AP** over self-attention, while self-attention *degrades* below TCT-ColBERT on the same benchmark.

**Table 2 — Inference efficiency at $K=1000$**

| Method | Complexity | Params | ms/query |
|---|---|---|---|
| Self-Attention | $O(K^2)$ | 363.9K | 37.3 |
| GNRR+GCN | $O(c \cdot K)$ | 263.0K | 34.2 |
| GNRR+GraphSAGE | $O(c \cdot K)$ | 160.2K | 32.8 |
| GNRR+GAT | $O(c \cdot K)$ | 246.8K | 34.3 |
| GNRR+GIN | $O(c \cdot K)$ | 123.5K | 32.4 |
| GNRR+SignedConv | $O(c \cdot K)$ | 295.7K | 35.2 |

---

## Setup

One command creates the full reproducible environment (Python 3.8, PyTorch 1.12.1+cu113, PyG 2.3.1, pinned IR stack):

```bash
bash scripts/setup_env.sh     # creates conda env "GNRR"; verifies imports
conda activate GNRR
```

`requirements.txt` pins all package versions. `environment.yml` is also provided for `conda env create -f environment.yml`. The CUDA-specific torch/PyG wheels and pyterrier plugins that are no longer on PyPI are handled by `setup_env.sh`.

---

## Data and checkpoints

The corpus used is the MS MARCO passage ranking corpus (8.8M documents). Evaluation is on TREC-DL19, TREC-DL20, and TREC-DLHard (accessed via `ir-datasets`/`pyterrier`; no manual download needed for the test sets).

The BM25 training and validation pools, precomputed fast tensors, and corpus graphs are expected under `data/msmarco_data/`. Trained model checkpoints live under `models/msmarco_data/`.

---

## Reproducing the paper

### Environment check

```bash
conda activate GNRR
python -c "import torch_geometric, pyterrier, ir_measures; print('ok')"
```

### Table 1 — Main re-ranking results

```bash
bash scripts/reproduce_table1.sh
```

Evaluates BM25, TCT-ColBERT, Self-Attention, and all five GNRR variants (GCN, GraphSAGE, GAT, GIN, SignedConv) on DL19, DL20, and DLHard.
Output: `results/benchmarks/<pipeline>/<dataset>/{aggregate,perquery}.csv` and the combined `results/benchmarks/table1.csv`.

### Table 2 — Sensitivity to candidate distribution (GAR + GNRR)

```bash
bash scripts/reproduce_table2.sh
```

Runs GAR (retrieval-stage graph augmentation) alone and then each GNRR re-ranker on the GAR-enlarged pool, without retraining.
Output: `results/benchmarks/GAR-semantic/` and `results/benchmarks/GAR+{gcn,sage,gat,gin,signed}-semantic/`.

### Efficiency table

```bash
bash scripts/run_efficiency.sh
```

Reports parameters, per-query GPU latency (subgraph extraction + GNN/attention forward), and theoretical complexity for all models.
Output: `results/efficiency/`.

---

## Additional experiments

### GAR baseline (inference only)

```bash
bash scripts/run_gar.sh
```

Evaluates GAR (MacAvaney et al., CIKM 2022) — corpus-graph-guided candidate expansion — without any re-ranking.

### Non-neural graph baselines

```bash
bash scripts/run_graph_baselines.sh
```

Inference-only baselines: score smoothing (1-step neighbor mean), kNN score interpolation, PageRank, and label propagation over the precomputed test graphs.

## Training your own models

### GNN variants

```bash
# semantic corpus graph (paper setting)
GRAPH_TYPE=semantic bash scripts/train_gnn.sh

# lexical (BM25 kNN) corpus graph
GRAPH_TYPE=lexical bash scripts/train_gnn.sh
```

Controls: `CONVS`, `N_SEEDS`, `PATIENCE`, `EPOCHS`, `MODALITY`. See the script header for full details. Checkpoints are saved under `models/msmarco_data/`.

### Self-attention baseline

```bash
bash scripts/train_selfattention.sh
```

Trains a multi-layer TransformerEncoder re-ranker on the same TCT-ColBERT features with the same LambdaRank + early stopping protocol as the GNN variants.

---

## Repository structure

```
.
├── main.py                        # Training entry point
├── src/
│   ├── GNN.py                     # GNN re-ranker architectures
│   ├── attention.py               # Self-attention re-ranker
│   ├── datamodule.py              # PyTorch Lightning data module
│   ├── lightningmodule.py         # Training / validation loop
│   ├── utils.py                   # Feature construction, metrics
│   └── config.py                  # Shared defaults
├── scripts/
│   ├── setup_env.sh               # Environment setup
│   ├── reproduce_table1.sh        # Table 1 (main results)
│   ├── reproduce_table2.sh        # Table 2 (GAR + GNRR)
│   ├── train_gnn.sh               # Train GNN variants
│   ├── train_selfattention.sh     # Train self-attention baseline
│   ├── evaluate_testset.py        # Per-model inference + evaluation
│   ├── run_gar_all.sh             # GAR + all GNRR variants (called by reproduce_table2.sh)
│   ├── run_gar.sh / run_gar.py    # GAR-only baseline
│   ├── run_efficiency.sh          # Efficiency measurements
│   ├── run_significance.sh        # Significance tests
│   ├── run_ablations.sh           # Component ablations
│   ├── run_controls.sh            # Graph controls
│   ├── run_graph_baselines.sh     # Non-neural graph baselines
│   ├── run_multiseed.sh           # Multi-seed evaluation
│   └── collect_results.py         # Aggregate CSVs into tables
├── figures/
│   └── pipeline_graphical_overview.pdf
├── data/                          # Corpus, graphs, tensors (not tracked)
├── models/                        # Checkpoints (not tracked)
└── results/                       # Evaluation outputs
```

---

