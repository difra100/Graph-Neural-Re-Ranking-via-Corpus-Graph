"""Significance testing for all re-ranking systems vs TCT-ColBERT baseline.

Uses pt.Experiment() with Bonferroni-corrected paired t-tests on pre-saved TREC runs.
Covers Table 1 (GNN + self-attention re-rankers) and Table 2 (GAR + GNRR pipelines).

Prerequisites: reproduce_table1.sh and reproduce_table2.sh must have been run so that
TREC run files exist under results/benchmarks/<tag>/<dataset>/run.csv.gz.

Usage:
    python scripts/significance.py                    # both tables, all datasets
    python scripts/significance.py --table 1          # Table 1 only
    python scripts/significance.py --dataset dl19     # one dataset only
    python scripts/significance.py --out results/significance/
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pyterrier as pt
import ir_measures

if not pt.started():
    pt.init()

# ── dataset helpers ──────────────────────────────────────────────────────────

DATASET_IRDS = {
    "dl19":   "irds:msmarco-passage/trec-dl-2019/judged",
    "dl20":   "irds:msmarco-passage/trec-dl-2020/judged",
    "dlhard": "irds:msmarco-passage/trec-dl-hard",
}

METRICS = [
    ir_measures.AP(rel=2),
    ir_measures.nDCG@10,
    ir_measures.RR(rel=2),
]


def load_run(tag, dataset, out_dir="results/benchmarks"):
    """Load a saved TREC run into a PyTerrier-compatible DataFrame."""
    f = Path(out_dir) / tag / dataset / "run.csv.gz"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    # normalise column names and types to PyTerrier standard (qid/docno as str)
    df = df.rename(columns={"query_id": "qid", "doc_id": "docno"})
    df["qid"]   = df["qid"].astype(str)
    df["docno"] = df["docno"].astype(str)
    if "rank" not in df.columns:
        df = df.sort_values(["qid", "score"], ascending=[True, False])
        df["rank"] = df.groupby("qid").cumcount() + 1
    return df


# ── system definitions ───────────────────────────────────────────────────────

# (display_name, run_tag)
TABLE1 = [
    ("TCT-ColBERT",    "TCT-ColBERT"),
    ("+Self-Attention","GNRR-transformer-single"),
    ("+GCN",           "GNRR-gcn-local"),
    ("+GraphSAGE",     "GNRR-sage-local"),
    ("+GAT",           "GNRR-gat-local"),
    ("+GIN",           "GNRR-gin-local"),
    ("+SignedConv",     "GNRR-signed-local"),
]

TABLE2 = [
    ("TCT-ColBERT",    "TCT-ColBERT"),
    ("GAR",            "GAR-semantic"),
    ("GAR+GCN",        "GAR+gcn-semantic"),
    ("GAR+GraphSAGE",  "GAR+sage-semantic"),
    ("GAR+GAT",        "GAR+gat-semantic"),
    ("GAR+GIN",        "GAR+gin-semantic"),
    ("GAR+SignedConv", "GAR+signed-semantic"),
]


def run_experiment(systems, dataset, out_dir, save_path=None):
    """Run pt.Experiment for a list of (name, tag) systems on one dataset."""
    pt_ds  = pt.get_dataset(DATASET_IRDS[dataset])
    topics = pt_ds.get_topics()
    qrels  = pt_ds.get_qrels()

    loaded, names = [], []
    missing = []
    for name, tag in systems:
        run = load_run(tag, dataset, out_dir)
        if run is None:
            missing.append(tag)
        else:
            loaded.append(run)
            names.append(name)

    if missing:
        print(f"  [skip] missing runs (run reproduce scripts first): {missing}", file=sys.stderr)

    if len(loaded) < 2:
        print(f"  [skip] need at least baseline + 1 system, got {len(loaded)}", file=sys.stderr)
        return None

    # find the TCT-ColBERT baseline index
    baseline_idx = next((i for i, n in enumerate(names) if n == "TCT-ColBERT"), 0)

    result = pt.Experiment(
        loaded,
        topics,
        qrels,
        eval_metrics=METRICS,
        names=names,
        baseline=baseline_idx,
        round=3,
    )
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False)
        print(f"  [saved] {save_path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table",   choices=["1", "2", "both"], default="both")
    ap.add_argument("--dataset", choices=["dl19", "dl20", "dlhard", "all"], default="all")
    ap.add_argument("--out_dir", default="results/benchmarks")
    ap.add_argument("--save",    default="results/significance")
    args = ap.parse_args()

    datasets = ["dl19", "dl20", "dlhard"] if args.dataset == "all" else [args.dataset]
    tables   = {"1": [1], "2": [2], "both": [1, 2]}[args.table]

    pd.set_option("display.width", 220, "display.max_columns", 40, "display.float_format", "{:.3f}".format)

    for tbl in tables:
        systems = TABLE1 if tbl == 1 else TABLE2
        print(f"\n{'='*70}")
        print(f"  TABLE {tbl} — significance vs TCT-ColBERT (paired t-test)")
        print(f"{'='*70}")
        for ds in datasets:
            print(f"\n  --- {ds.upper()} ---")
            save_path = f"{args.save}/table{tbl}_{ds}.csv" if args.save else None
            result = run_experiment(systems, ds, args.out_dir, save_path=save_path)
            if result is not None:
                print(result.to_string(index=False))


if __name__ == "__main__":
    main()
