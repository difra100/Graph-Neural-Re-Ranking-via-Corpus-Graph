## Graph Neural Re-Ranking via Corpus Graph

### Project Description
This project aims at showing that when we consider cross-document interactions for learning-to-rank, we can improve the ranking performance. Modeling this interaction is done through a Corpus Graph, which is a data structure used to improve the recall of documents. Here we leverage this structure to create connections between different documents that are semantically related.  

### Set-up and installation
One command builds the full reproducible environment (python 3.8 + torch 1.12.1+cu113
+ PyG 2.3.1 + the pinned IR stack, and vendors the terrierteam plugins):
```
bash scripts/setup_env.sh          # creates conda env "GNRR"; verifies imports
conda activate GNRR
```
Notes:
- `requirements.txt` is pinned to the known-good versions (`python-terrier==0.10.0`,
  numpy/scipy/sklearn, pytorch-lightning 1.5.10). The CUDA-specific torch/PyG wheels
  and the `pyterrier_{adaptive,dr,t5}` 0.0.1 plugins (no longer on PyPI) are handled
  by `setup_env.sh`. `environment.yml` is provided for `conda env create` as well.

### Reproduce the benchmarks (Table 1)
```
bash scripts/reproduce_table1.sh            # BM25, TCT-ColBERT, 5 GNRR variants x DL19/20/DLHard
```
Per-pipeline metrics land in `results/benchmarks/<pipeline>/<dataset>/` and the
combined table in `results/benchmarks/table1.csv`.

### Additional experiments / baselines (reviewer responses)
```
bash scripts/run_gar.sh             # GAR / [18] corpus-graph re-ranking baseline (inference)
bash scripts/run_graph_baselines.sh # score smoothing / kNN interp / PageRank / label prop
bash scripts/run_selfattention.sh   # self-attention re-rankers (1-stage, multi-stage) + GNRR multistage
bash scripts/run_significance.sh    # paired bootstrap + randomization tests
# Deferred (run on demand): run_controls.sh, run_efficiency.sh, run_ablations.sh, run_multiseed.sh
```
See `scripts/` for single-experiment entry points (each `.py` has `--help`).

### General Framework
Our framework is presented in figure.
![image](figures/general_scheme_2.png)
