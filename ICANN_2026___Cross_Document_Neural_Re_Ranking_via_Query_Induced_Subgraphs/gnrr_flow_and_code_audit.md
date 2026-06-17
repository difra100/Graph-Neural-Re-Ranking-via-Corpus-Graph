# GNRR Flow and Code Audit

Source paper: `main.tex`

## One-Sentence Flow

GNRR retrieves a top-K candidate set for a query, induces a sparse document-document subgraph from an offline corpus graph, computes query-document node features with frozen TCT-ColBERT embeddings, propagates cross-document signal with a GNN, and scores documents for final re-ranking.

## Intended Paper Flow

1. Build the corpus graph offline.
   - Node: one corpus document.
   - Edge: semantic-neighbor relation from TCT-ColBERT document embeddings.
   - Sparsity: each node keeps at most `c` neighbors, with `c = 8` in the experiments.
   - Example: if `d5` is close to `d8` and `d11`, the corpus graph stores edges like `d5 -> d8` and `d5 -> d11`.

2. Retrieve query candidates online.
   - Input: query `q`.
   - First-stage retriever: BM25.
   - Output: candidate set `C_q` with up to `K = 1000` documents.
   - Example: BM25 returns `[d5, d2, d8]` for `q`.

3. Extract the query-induced subgraph.
   - Keep only retrieved documents as nodes.
   - Keep only corpus-graph edges whose endpoints are both in `C_q`.
   - Example: corpus edges are `(d5, d8)`, `(d5, d11)`, `(d2, d19)`. For candidates `[d5, d2, d8]`, the induced subgraph keeps only `(d5, d8)`.

4. Build node features.
   - Query embedding: `z_q`, from frozen TCT-ColBERT.
   - Document embedding: `z_d`, from frozen TCT-ColBERT.
   - Paper feature function: `x_i = f1(z_q, z_d_i)`, with `f1 = hadamard` in the hyperparameter section.
   - Example: if `z_q = [2, 3]` and `z_d5 = [4, 5]`, then `x_5 = [8, 15]`.

5. Run GNN propagation.
   - Input: node feature matrix plus the induced adjacency.
   - Paper says the GNN feature is augmented with a BM25 score.
   - Output: contextualized node representations `H_q,GNN`.
   - Example: `d5` can receive signal from `d8` if they are connected in the induced subgraph.

6. Merge and score.
   - Paper merge function: concatenate GNN output with the original query-document feature.
   - A learned scorer maps the merged vector to one scalar per document.
   - Final ranking: sort candidates by descending scalar score.
   - Example: BM25 ranks `[d5, d2, d8]`, but the GNN scorer can output scores `{d8: 2.1, d5: 1.4, d2: 0.3}`, yielding `[d8, d5, d2]`.

## Current Implementation Map

1. Experiment setup: `main.py`
   - Parses dataset, model, training, and sweep arguments.
   - Builds MS MARCO train/validation query lists.
   - Creates `DataModule_terrier`, a GNN/MLP model, and `TrainingModule`.

2. Data and subgraph construction: `src/datamodule.py` and `src/utils.py`
   - `Dataset_terrierlike.get_sample_slow()` loads precomputed BM25 candidates, encodes/loads document embeddings, extracts the induced corpus subgraph, builds COO edges, and returns padded tensors.
   - `Dataset_terrierlike.gat_sample_fast()` loads already-materialized tensors from `data/msmarco_data/*_data_fast/tensors/`.
   - `generate_corpus_subgraph_induced_by_query()` filters corpus-graph neighbors to the retrieved documents only.
   - `build_adjacency_matrix()` symmetrizes kept edges into an unweighted adjacency matrix.

3. Query-document feature construction: `src/utils.py`
   - `compute_output()` repeats the query embedding and combines it with each document embedding using `concat`, `sum`, or `hadamard`.
   - The default paper-aligned path is `hadamard`.

4. Models: `src/GNN.py`
   - `GNN_NR`: single GNN scorer for `gcn`, `sage`, `gat`, `gatv2`, `gin`, or `signed`.
   - `GNN_LG(local)`: intended local GNN plus final MLP, but the current forward path replaces the GNN output with zeros.
   - `GNN_LG(multistage)`: active script path; computes TCT-style dot scores, runs the GNN only on top `K_multistage` documents, and adds the learned GNN/MLP delta back to those scores.
   - `GNN_LG(global)`: loads a pretrained local model, computes graph-level pooled features, and combines global, local, and individual features.

5. Training and evaluation: `src/lightningmodule.py`
   - `TrainingModule.training_step()` computes one query graph at a time inside each batch and trains with MSE, ListNet, ListMLE, RankNet, or LambdaRank.
   - `validation_step()` and `test_step()` collect per-document scores into `doc_test_df`.
   - `Get_Metrics` computes aggregate IR metrics with `ir_measures`.

## Key Paper/Code Mismatches

1. The paper describes full GNN propagation followed by concatenation with original node features. The active `multistage` code instead adds a learned delta to TCT-style dot scores for only the top `K_multistage` documents.

2. The paper says each feature vector is augmented with the BM25 score. The code appends either a positional index (`torch.arange`) in local/global paths or a TCT-style dot-product score in the multistage path.

3. The paper says all GNN variants are the GNRR model over the induced subgraph. The `local` branch currently does not call its GNN at all because `z_local = self.GNN(...)` is commented out and replaced by zeros.

4. The paper states `f2 = concat`. That is true for `local`, but not for the active `multistage` path used by `run_global_exp.sh`.

## Bug Candidates To Fix First

1. `src/GNN.py`: local mode disables the GNN branch.
   - Location: `GNN_LG.forward()`, local branch.
   - Effect: local experiments do not use graph propagation, so they cannot support claims about local GNN context.

2. `main.py`: CLI booleans use `type=bool`.
   - Example: `--fast_train False` parses as `True`.
   - Effect: command-line runs can silently enable sweep, W&B, fast data, score features, and structure learning.

3. `main.py`: sweep training runs `trainer.fit()` twice.
   - Location: after `TrainingModule.load_from_checkpoint(...)`.
   - Effect: sweep runs waste time and may overwrite state/checkpoints before testing.

4. `src/GNN.py`: the extra score feature is not BM25.
   - Location: local/global use `torch.arange`; multistage uses `torch.sum(query_doc_feature)`.
   - Effect: the implementation does not match the manuscript claim that BM25 scores are appended.

5. `run_global_exp.sh`: GAT is invoked with `--heads 0`, while `main.py` declares heads as `bool`.
   - Effect: due Python truthiness, `"0"` becomes `True`, so the actual value is effectively `1`, not `0`. If heads is later fixed to `int`, `heads=0` would be invalid for GAT.

6. `src/datamodule.py`: fast tensor loading still constructs `FlexIndex` and loads the corpus graph.
   - Effect: `--fast_train True` can still fail when the index or corpus graph is unavailable, even though the fast path only needs precomputed tensors.

7. `requirements.txt`: PyTorch Geometric dependencies are missing.
   - Effect: installing only `requirements.txt` leaves `main.py` unable to import `torch_geometric`; the README has a separate install command, but the requirements file is incomplete as a reproducibility artifact.

## Minimal Correctness Checklist

1. Decide whether the paper should describe `local` GNRR or the active `multistage` delta-scoring model.
2. If claiming BM25 augmentation, pass real BM25 scores through the dataloader and append those values, not rank indices or dot-product scores.
3. Replace boolean CLI arguments with `store_true`/`store_false` flags or a strict string-to-bool parser.
4. Remove the second `trainer.fit()` in the sweep path.
5. Either restore `z_local = self.GNN(x_local, edge_index)` in local mode or label that branch as a no-GNN ablation.
6. Keep dependency installation in one reproducible place.
