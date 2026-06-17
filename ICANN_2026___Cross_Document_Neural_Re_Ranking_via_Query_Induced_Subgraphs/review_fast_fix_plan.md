# Few-Day Revision Plan After Reviews

Goal: strengthen the paper quickly without adding avoidable self-damaging language. The safest path is to fix visible manuscript issues, add 2-4 targeted controls if the pipeline runs, and soften claims that reviewers can disprove from Table 1.

## Step 1: Fix Immediately, No New Experiments

Priority: must do in the first half-day.

1. Remove obvious presentation defects.
   - Delete the duplicated `Contribution of the GNN Branch (RQ2)` subsection.
   - Remove the repeated corpus-graph sparsity sentence.
   - Fix `\\geq 2` to `\geq 2`.
   - Enlarge or simplify Fig. 1 labels if possible.

2. Reword overclaims.
   - Replace "confirms", "validates", and "establishes" with "suggests" or "provides evidence".
   - Replace "most robust across all three benchmarks and all evaluated metrics" with "strongest AP/nDCG@10 profile, while P@3 varies by dataset".
   - Do not claim the ablation proves graph propagation unless a scorer-kept no-GNN control is added.

3. Clarify method details reviewers flagged.
   - State whether edges are symmetrized and whether they are weighted or unweighted.
   - Explain how `c=8` is chosen.
   - Specify the scorer architecture and LambdaRank setup.
   - Reconcile Fig. 1 with the text: document embeddings are precomputed offline; query embeddings and node features are computed online.

4. Remove ambiguity that looks like test-set tuning.
   - Say hyperparameters are selected on MS MARCO validation only and then evaluated unchanged on DL19, DL20, and DLHard.
   - Delete or rephrase "Optimal configurations vary across datasets" unless that variation is from validation-only selection.

5. Adjust the training-scale paragraph.
   - Avoid saying 1,000 training queries are "sufficient".
   - Use: "We use a controlled 1,000-query training subset to study the framework under limited supervision; scaling training is left for future work."

## Step 2: Quick Experiments Worth Doing

Run these only if the data/checkpoints are ready. They directly answer the common reviewer objections.

1. Scorer-kept no-GNN control.
   - Compare full GCN against an MLP/scorer using the same TCT features and lexical/rank feature but no edges.
   - This is the most important missing ablation because reviewers think gains may come from extra model capacity.
   - Include if GCN clearly beats it. If it matches GCN, narrow the claim to "learned calibration plus sparse context" and stop claiming isolated graph benefit.

2. Lexical/rank feature ablation.
   - Run GCN with and without the additional BM25/rank feature.
   - This addresses the "BM25 leakage" criticism.
   - Before reporting, verify the implementation really uses BM25 score. If it uses rank or a proxy, update the manuscript to say so.

3. Connectivity perturbation.
   - Evaluate the trained GCN with original edges, empty edges, and degree-matched random/shuffled edges.
   - This is faster than retraining and tests whether graph connectivity matters at inference.
   - Include if original edges outperform random/empty edges.

4. Significance or confidence intervals.
   - Export per-query metrics and run paired bootstrap or randomization tests for GCN vs TCT-ColBERT.
   - If p-values are not significant, report confidence intervals briefly and avoid strong reliability language.

5. Efficiency measurements.
   - Report mean/median online latency for subgraph extraction plus GCN forward pass at `K=1000`.
   - Report corpus graph size/storage or at least the actual edge count.
   - If exact storage is unavailable, remove "substantially reduced overhead" and keep only the formal `O(c*K*L)` statement.

6. Subgraph density statistics.
   - Add mean edges, mean degree, and isolated-node fraction for DL19/DL20/DLHard.
   - This is cheap and helps explain why GCN is more stable than GAT/GIN/SignedConv.

## Step 3: Experiments To Consider Only If There Is Spare Time

1. `c` sensitivity for GCN: `c in {4, 8, 16}`.
   - Strong if it shows `c=8` is not arbitrary.

2. Three random seeds for GCN and the no-GNN control.
   - Stronger than training many architectures once.

3. Larger training subset for GCN only, e.g. 2k-5k queries.
   - Useful if compute is available, but less targeted than the controls above.

## Step 4: Defer, Do Not Promise In This Revision

These are valid reviewer requests but not realistic in a few days.

1. Full MS MARCO training.
2. HotpotQA / 2WikiMultiHopQA transfer.
3. New modern encoder experiments.
4. Full SetRank/permutation/self-attention baseline implementation.
5. Complete GAR/MacAvaney reproduction if code/data are not already wired.

Mention at most one sentence as future work. Do not expand the limitations section into a list of weaknesses.

## Recommended Paper Positioning

Use this framing:

"GNRR studies whether a sparse corpus-graph prior can improve neural re-ranking with limited online computation. The strongest evidence is the GCN variant, which improves AP and nDCG@10 over TCT-ColBERT on all three TREC benchmarks. Additional controls show whether the gains come from graph connectivity rather than only supervised score calibration."

Avoid this framing:

"The GNN branch is proven to be the source of gains."

"GNRR is more efficient than attention-based re-rankers."

"GCN is best on every metric."

"The small training subset is sufficient."

## Minimal Final Checklist

1. Clean duplicate text and typo-level issues.
2. Add method details for graph edges, scorer, LambdaRank, and offline/online split.
3. Replace overclaims with measured language.
4. Add no-GNN scorer control if only one experiment can be done.
5. Add significance or bootstrap intervals if per-query outputs can be exported.
6. Add latency/storage or remove empirical efficiency wording.
7. Add subgraph density stats if available.

Best few-day outcome: a revised paper that is modest, cleaner, and backed by one strong control table plus one small efficiency/density table.
