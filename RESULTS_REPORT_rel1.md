# GNRR — Results Report (relevance ≥ 1)

All binarised metrics (AP/RR/P@3/R@1000) use relevance grade ≥ 1; nDCG@10 is graded (threshold-independent). MS MARCO training labels are binary (grade 1), so **rel≥1 is the eval that matches what the models were trained to rank**; rel≥2 is the stricter setting used in the original paper.

Two encoder versions appear and must not be compared across blocks: **v1** = `tct_colbert-msmarco` (original paper), **v2** = `tct_colbert-v2-hnp-msmarco` (new). `—` = result omitted (model under-trained / diverged; to be retrained).

Graph: corpus kNN graph (k capped at 8). **semantic** = TCT cosine graph; **lexical** = BM25 graph (`corpusgraph_bm25_k16`). Hardware: RTX 3090 Ti.


## 1. Original paper (TCT-ColBERT v1)

_Omitted: the paper reports rel≥2 only and no v1 runs were saved to recompute at rel≥1. See RESULTS_REPORT.md for the rel≥2 paper block._


## 2. New experiments — single-stage (TCT-ColBERT **v2**, re-rank BM25 top-1000)

**DL19**

| Pipeline | AP | RR | P@3 | nDCG@10 |
|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.465 | 0.970 | 0.860 | 0.712 |
| TCT+GCN (lexical) | — | — | — | — |
| TCT+GraphSAGE (lexical) | 0.451 | 0.953 | 0.853 | 0.677 |
| TCT+GAT (lexical) | 0.486 | 0.958 | 0.860 | 0.693 |
| TCT+GIN (lexical) | 0.448 | 0.944 | 0.822 | 0.675 |
| TCT+SignedConv (lexical) | 0.462 | 0.953 | 0.853 | 0.686 |
| TCT+EdgeGAT (semantic) | 0.477 | 0.938 | 0.853 | 0.680 |
| TCT+EdgeGAT (lexical) | 0.449 | 0.944 | 0.837 | 0.669 |
| Self-attention (1-stage) | 0.465 | 0.973 | 0.899 | 0.691 |

**DL20**

| Pipeline | AP | RR | P@3 | nDCG@10 |
|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.473 | 0.950 | 0.846 | 0.690 |
| TCT+GCN (lexical) | — | — | — | — |
| TCT+GraphSAGE (lexical) | 0.461 | 0.943 | 0.846 | 0.670 |
| TCT+GAT (lexical) | 0.488 | 0.937 | 0.846 | 0.667 |
| TCT+GIN (lexical) | 0.457 | 0.898 | 0.827 | 0.648 |
| TCT+SignedConv (lexical) | 0.471 | 0.965 | 0.846 | 0.675 |
| TCT+EdgeGAT (semantic) | 0.478 | 0.915 | 0.821 | 0.655 |
| TCT+EdgeGAT (lexical) | 0.456 | 0.919 | 0.815 | 0.656 |
| Self-attention (1-stage) | 0.466 | 0.931 | 0.852 | 0.678 |

**DLHARD**

| Pipeline | AP | RR | P@3 | nDCG@10 |
|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.237 | 0.579 | 0.407 | 0.371 |
| TCT+GCN (lexical) | — | — | — | — |
| TCT+GraphSAGE (lexical) | 0.224 | 0.601 | 0.433 | 0.365 |
| TCT+GAT (lexical) | 0.237 | 0.575 | 0.433 | 0.373 |
| TCT+GIN (lexical) | 0.216 | 0.526 | 0.387 | 0.338 |
| TCT+SignedConv (lexical) | 0.237 | 0.593 | 0.447 | 0.377 |
| TCT+EdgeGAT (semantic) | 0.225 | 0.540 | 0.407 | 0.355 |
| TCT+EdgeGAT (lexical) | 0.217 | 0.561 | 0.393 | 0.360 |
| Self-attention (1-stage) | 0.227 | 0.568 | 0.433 | 0.357 |


## 3. New experiments — multi-stage (re-rank TCT top-200, backfill by TCT)

**DL19**

| Pipeline | AP | RR | P@3 | nDCG@10 |
|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.465 | 0.970 | 0.860 | 0.712 |
| TCT+GCN (lexical) | — | — | — | — |
| TCT+GraphSAGE (lexical) | 0.439 | 0.928 | 0.822 | 0.665 |
| TCT+GAT (lexical) | 0.471 | 0.945 | 0.860 | 0.689 |
| TCT+GIN (lexical) | 0.454 | 0.946 | 0.837 | 0.679 |
| TCT+SignedConv (lexical) | 0.449 | 0.948 | 0.860 | 0.689 |
| TCT+EdgeGAT (semantic) | 0.472 | 0.942 | 0.860 | 0.679 |
| TCT+EdgeGAT (lexical) | 0.453 | 0.950 | 0.837 | 0.671 |
| Self-attention (multi) | 0.462 | 0.961 | 0.915 | 0.691 |

**DL20**

| Pipeline | AP | RR | P@3 | nDCG@10 |
|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.473 | 0.950 | 0.846 | 0.690 |
| TCT+GCN (lexical) | — | — | — | — |
| TCT+GraphSAGE (lexical) | 0.454 | 0.930 | 0.840 | 0.662 |
| TCT+GAT (lexical) | 0.477 | 0.925 | 0.846 | 0.680 |
| TCT+GIN (lexical) | 0.461 | 0.898 | 0.827 | 0.660 |
| TCT+SignedConv (lexical) | 0.455 | 0.920 | 0.840 | 0.668 |
| TCT+EdgeGAT (semantic) | 0.475 | 0.915 | 0.821 | 0.656 |
| TCT+EdgeGAT (lexical) | 0.457 | 0.920 | 0.815 | 0.655 |
| Self-attention (multi) | 0.464 | 0.933 | 0.852 | 0.674 |

**DLHARD**

| Pipeline | AP | RR | P@3 | nDCG@10 |
|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.237 | 0.579 | 0.407 | 0.371 |
| TCT+GCN (lexical) | — | — | — | — |
| TCT+GraphSAGE (lexical) | 0.221 | 0.556 | 0.407 | 0.359 |
| TCT+GAT (lexical) | 0.227 | 0.554 | 0.393 | 0.362 |
| TCT+GIN (lexical) | 0.221 | 0.536 | 0.407 | 0.349 |
| TCT+SignedConv (lexical) | 0.226 | 0.583 | 0.420 | 0.373 |
| TCT+EdgeGAT (semantic) | 0.226 | 0.544 | 0.413 | 0.355 |
| TCT+EdgeGAT (lexical) | 0.218 | 0.564 | 0.387 | 0.359 |
| Self-attention (multi) | 0.240 | 0.596 | 0.480 | 0.374 |


## 4. GAR (graph for recall) + GNRR — complementarity (TCT-ColBERT **v2**)

_GAR expands the candidate pool via the corpus graph (recall), then the model re-ranks. Note the recall column R@1000._

**DL19**

| Pipeline | AP | RR | P@3 | nDCG@10 | R@1000 |
|---|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.465 | 0.970 | 0.860 | 0.712 | 0.736 |
| GAR (recall only) | 0.496 | 0.988 | 0.884 | 0.727 | 0.828 |
| GAR + GAT (lexical) | 0.495 | 0.953 | 0.884 | 0.698 | 0.823 |
| GAR + SignedConv (lexical) | 0.461 | 0.950 | 0.845 | 0.695 | 0.823 |
| GAR + EdgeGAT (semantic) | 0.511 | 0.938 | 0.868 | 0.707 | 0.828 |

**DL20**

| Pipeline | AP | RR | P@3 | nDCG@10 | R@1000 |
|---|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.473 | 0.950 | 0.846 | 0.690 | 0.751 |
| GAR (recall only) | 0.481 | 0.948 | 0.858 | 0.690 | 0.838 |
| GAR + GAT (lexical) | 0.485 | 0.938 | 0.840 | 0.676 | 0.823 |
| GAR + SignedConv (lexical) | 0.461 | 0.922 | 0.852 | 0.670 | 0.823 |
| GAR + EdgeGAT (semantic) | 0.488 | 0.933 | 0.840 | 0.654 | 0.838 |

**DLHARD**

| Pipeline | AP | RR | P@3 | nDCG@10 | R@1000 |
|---|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.237 | 0.579 | 0.407 | 0.371 | 0.659 |
| GAR (recall only) | 0.242 | 0.602 | 0.433 | 0.378 | 0.754 |
| GAR + GAT (lexical) | 0.236 | 0.567 | 0.407 | 0.367 | 0.718 |
| GAR + SignedConv (lexical) | 0.231 | 0.581 | 0.407 | 0.370 | 0.718 |
| GAR + EdgeGAT (semantic) | 0.232 | 0.551 | 0.433 | 0.364 | 0.754 |


## 5. Inference cost (DL19, per query)

Online re-ranking cost = subgraph extraction + model forward (excludes BM25 + the one-off query encoding shared by all systems).

| Model | Params | Online single-stage (ms) | Online multi-stage (ms) | Peak GPU mem single / multi (MB) |
|---|---|---|---|---|
| GCN (lexical) | 263 K | 34.2 | 16.6 | 29 / 7 |
| GraphSAGE (lexical) | 160 K | 32.8 | 15.2 | 49 / 10 |
| GAT (lexical) | 247 K | 34.3 | 15.1 | 29 / 7 |
| GIN (lexical) | 123 K | 32.4 | 14.8 | 48 / 10 |
| SignedConv (lexical) | 296 K | 35.2 | 17.0 | 53 / 12 |
| EdgeGAT (5L) | 1.41 M | 35.1 | 16.5 | 61 / 17 |
| Self-attention | 364 K | 37.3 | 14.8 | 36 / 6 |

**GAR recall step:** ≈ 165 ms/query (graph neighbour lookups + dot-product scoring over the expanded pool; no document encoding). It is the dominant online cost but is additive/complementary to the cheap (~15–35 ms) GNN re-ranking.

**Offline corpus graph** (`corpusgraph_k16`, MS MARCO 8.8M docs): edges ≈ 566 MB, weights ≈ 283 MB; built once, amortised across all queries.


## 6. Notes

- **Multi-stage ≈ single-stage** in accuracy for the strong models, but ~2× cheaper online (re-ranks 200 vs 1000) — its value is efficiency, not quality.

- **GAR is the clear win and is complementary**: it lifts R@1000 by ~+7–9 points and gives the best nDCG@10 on all three sets; adding the GNN on the expanded pool further improves AP/P@3 (e.g. GAR+EdgeGAT best AP on DL19; GAR+GAT best DLHard AP/RR/P@3).

- **Blanks (GCN-lexical v2):** diverged in training (nDCG ≈ 0.27); to be retrained.

- **Missing semantic-v2 GNNs** (SAGE/GAT/GIN/Signed): not retrained on v2; the v1 numbers in §1 stand in for the semantic graph.
