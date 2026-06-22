# GNRR — Results Report

All metrics use relevance grade ≥ 2. Two encoder versions appear and must not be compared across blocks: **v1** = `tct_colbert-msmarco` (original paper), **v2** = `tct_colbert-v2-hnp-msmarco` (all new experiments). `—` = result intentionally omitted (model under-trained / diverged; to be retrained).

Graph: corpus kNN graph (k capped at 8). **semantic** = TCT cosine graph; **lexical** = BM25 graph (`corpusgraph_bm25_k16`). Hardware: RTX 3090 Ti.


## 1. Original paper (TCT-ColBERT **v1**, semantic graph, single-stage, no GAR)

_Kept verbatim from the original submission._

**DL19**

| Pipeline | AP | RR | P@3 | nDCG@10 |
|---|---|---|---|---|
| BM25 | 0.286 | 0.642 | 0.473 | 0.480 |
| TCT-ColBERT | 0.430 | 0.843 | 0.667 | 0.685 |
| TCT+GCN | 0.455 | 0.858 | 0.713 | 0.702 |
| TCT+GraphSAGE | 0.434 | 0.850 | 0.728 | 0.689 |
| TCT+GAT | 0.442 | 0.852 | 0.698 | 0.690 |
| TCT+GIN | 0.449 | 0.853 | 0.729 | 0.691 |
| TCT+SignedConv | 0.426 | 0.828 | 0.698 | 0.675 |

**DL20**

| Pipeline | AP | RR | P@3 | nDCG@10 |
|---|---|---|---|---|
| BM25 | 0.293 | 0.619 | 0.463 | 0.494 |
| TCT-ColBERT | 0.453 | 0.817 | 0.691 | 0.680 |
| TCT+GCN | 0.470 | 0.840 | 0.673 | 0.695 |
| TCT+GraphSAGE | 0.454 | 0.837 | 0.667 | 0.685 |
| TCT+GAT | 0.453 | 0.826 | 0.685 | 0.682 |
| TCT+GIN | 0.455 | 0.793 | 0.642 | 0.675 |
| TCT+SignedConv | 0.445 | 0.813 | 0.685 | 0.676 |

**DLHARD**

| Pipeline | AP | RR | P@3 | nDCG@10 |
|---|---|---|---|---|
| BM25 | 0.147 | 0.422 | 0.240 | 0.274 |
| TCT-ColBERT | 0.230 | 0.538 | 0.353 | 0.373 |
| TCT+GCN | 0.242 | 0.559 | 0.367 | 0.386 |
| TCT+GraphSAGE | 0.223 | 0.531 | 0.366 | 0.379 |
| TCT+GAT | 0.219 | 0.542 | 0.364 | 0.376 |
| TCT+GIN | 0.215 | 0.490 | 0.333 | 0.357 |
| TCT+SignedConv | 0.218 | 0.509 | 0.353 | 0.366 |


## 2. New experiments — single-stage (TCT-ColBERT **v2**, re-rank BM25 top-1000)

**DL19**

| Pipeline | AP | RR | P@3 | nDCG@10 |
|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.438 | 0.875 | 0.713 | 0.712 |
| TCT+GCN (lexical) | — | — | — | — |
| TCT+GraphSAGE (lexical) | 0.427 | 0.820 | 0.721 | 0.677 |
| TCT+GAT (lexical) | 0.446 | 0.871 | 0.729 | 0.693 |
| TCT+GIN (lexical) | 0.426 | 0.852 | 0.698 | 0.675 |
| TCT+SignedConv (lexical) | 0.435 | 0.839 | 0.705 | 0.686 |
| TCT+EdgeGAT (semantic) | 0.451 | 0.837 | 0.729 | 0.680 |
| TCT+EdgeGAT (lexical) | 0.424 | 0.823 | 0.698 | 0.669 |
| Self-attention (1-stage) | 0.442 | 0.873 | 0.744 | 0.691 |

**DL20**

| Pipeline | AP | RR | P@3 | nDCG@10 |
|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.469 | 0.847 | 0.654 | 0.690 |
| TCT+GCN (lexical) | — | — | — | — |
| TCT+GraphSAGE (lexical) | 0.442 | 0.813 | 0.636 | 0.670 |
| TCT+GAT (lexical) | 0.454 | 0.827 | 0.648 | 0.667 |
| TCT+GIN (lexical) | 0.442 | 0.769 | 0.636 | 0.648 |
| TCT+SignedConv (lexical) | 0.451 | 0.821 | 0.642 | 0.675 |
| TCT+EdgeGAT (semantic) | 0.449 | 0.789 | 0.623 | 0.655 |
| TCT+EdgeGAT (lexical) | 0.442 | 0.809 | 0.630 | 0.656 |
| Self-attention (1-stage) | 0.462 | 0.822 | 0.698 | 0.678 |

**DLHARD**

| Pipeline | AP | RR | P@3 | nDCG@10 |
|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.220 | 0.506 | 0.327 | 0.371 |
| TCT+GCN (lexical) | — | — | — | — |
| TCT+GraphSAGE (lexical) | 0.215 | 0.508 | 0.320 | 0.365 |
| TCT+GAT (lexical) | 0.233 | 0.557 | 0.367 | 0.373 |
| TCT+GIN (lexical) | 0.201 | 0.482 | 0.313 | 0.338 |
| TCT+SignedConv (lexical) | 0.229 | 0.542 | 0.347 | 0.377 |
| TCT+EdgeGAT (semantic) | 0.212 | 0.502 | 0.320 | 0.355 |
| TCT+EdgeGAT (lexical) | 0.214 | 0.508 | 0.300 | 0.360 |
| Self-attention (1-stage) | 0.217 | 0.490 | 0.347 | 0.357 |


## 3. New experiments — multi-stage (re-rank TCT top-200, backfill by TCT)

**DL19**

| Pipeline | AP | RR | P@3 | nDCG@10 |
|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.438 | 0.875 | 0.713 | 0.712 |
| TCT+GCN (lexical) | — | — | — | — |
| TCT+GraphSAGE (lexical) | 0.415 | 0.801 | 0.698 | 0.665 |
| TCT+GAT (lexical) | 0.439 | 0.842 | 0.752 | 0.689 |
| TCT+GIN (lexical) | 0.429 | 0.841 | 0.705 | 0.679 |
| TCT+SignedConv (lexical) | 0.427 | 0.863 | 0.721 | 0.689 |
| TCT+EdgeGAT (semantic) | 0.448 | 0.837 | 0.736 | 0.679 |
| TCT+EdgeGAT (lexical) | 0.426 | 0.827 | 0.698 | 0.671 |
| Self-attention (multi) | 0.440 | 0.842 | 0.752 | 0.691 |

**DL20**

| Pipeline | AP | RR | P@3 | nDCG@10 |
|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.469 | 0.847 | 0.654 | 0.690 |
| TCT+GCN (lexical) | — | — | — | — |
| TCT+GraphSAGE (lexical) | 0.439 | 0.789 | 0.630 | 0.662 |
| TCT+GAT (lexical) | 0.460 | 0.808 | 0.648 | 0.680 |
| TCT+GIN (lexical) | 0.445 | 0.764 | 0.636 | 0.660 |
| TCT+SignedConv (lexical) | 0.443 | 0.797 | 0.648 | 0.668 |
| TCT+EdgeGAT (semantic) | 0.449 | 0.789 | 0.623 | 0.656 |
| TCT+EdgeGAT (lexical) | 0.442 | 0.810 | 0.630 | 0.655 |
| Self-attention (multi) | 0.460 | 0.824 | 0.673 | 0.674 |

**DLHARD**

| Pipeline | AP | RR | P@3 | nDCG@10 |
|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.220 | 0.506 | 0.327 | 0.371 |
| TCT+GCN (lexical) | — | — | — | — |
| TCT+GraphSAGE (lexical) | 0.214 | 0.489 | 0.333 | 0.359 |
| TCT+GAT (lexical) | 0.224 | 0.530 | 0.347 | 0.362 |
| TCT+GIN (lexical) | 0.203 | 0.484 | 0.320 | 0.349 |
| TCT+SignedConv (lexical) | 0.221 | 0.530 | 0.320 | 0.373 |
| TCT+EdgeGAT (semantic) | 0.213 | 0.503 | 0.327 | 0.355 |
| TCT+EdgeGAT (lexical) | 0.213 | 0.509 | 0.287 | 0.359 |
| Self-attention (multi) | 0.230 | 0.523 | 0.367 | 0.374 |


## 4. GAR (graph for recall) + GNRR — complementarity (TCT-ColBERT **v2**)

_GAR expands the candidate pool via the corpus graph (recall), then the model re-ranks. Note the recall column R@1000._

**DL19**

| Pipeline | AP | RR | P@3 | nDCG@10 | R@1000 |
|---|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.438 | 0.875 | 0.713 | 0.712 | 0.755 |
| GAR (recall only) | 0.464 | 0.875 | 0.721 | 0.727 | 0.844 |
| GAR + GAT (lexical) | 0.461 | 0.853 | 0.744 | 0.698 | 0.840 |
| GAR + SignedConv (lexical) | 0.439 | 0.860 | 0.682 | 0.695 | 0.840 |
| GAR + EdgeGAT (semantic) | 0.483 | 0.845 | 0.736 | 0.707 | 0.844 |

**DL20**

| Pipeline | AP | RR | P@3 | nDCG@10 | R@1000 |
|---|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.469 | 0.847 | 0.654 | 0.690 | 0.807 |
| GAR (recall only) | 0.477 | 0.832 | 0.648 | 0.690 | 0.884 |
| GAR + GAT (lexical) | 0.470 | 0.824 | 0.636 | 0.676 | 0.871 |
| GAR + SignedConv (lexical) | 0.450 | 0.777 | 0.660 | 0.670 | 0.871 |
| GAR + EdgeGAT (semantic) | 0.458 | 0.806 | 0.623 | 0.654 | 0.884 |

**DLHARD**

| Pipeline | AP | RR | P@3 | nDCG@10 | R@1000 |
|---|---|---|---|---|---|
| TCT-ColBERT (v2) | 0.220 | 0.506 | 0.327 | 0.371 | 0.687 |
| GAR (recall only) | 0.231 | 0.506 | 0.333 | 0.378 | 0.763 |
| GAR + GAT (lexical) | 0.239 | 0.537 | 0.353 | 0.367 | 0.747 |
| GAR + SignedConv (lexical) | 0.230 | 0.528 | 0.320 | 0.370 | 0.747 |
| GAR + EdgeGAT (semantic) | 0.228 | 0.511 | 0.347 | 0.364 | 0.763 |


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
