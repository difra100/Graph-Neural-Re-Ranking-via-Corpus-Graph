"""One-sided significance testing: system > TCT-ColBERT baseline.

Uses a paired one-sided t-test (H1: system metric > baseline metric) computed
directly from per-query scores via scipy.stats.ttest_rel(alternative='greater').

Prerequisites: reproduce_table1.sh and reproduce_table2.sh must have been run.

Usage:
    python scripts/significance_greater.py                    # both tables, all datasets
    python scripts/significance_greater.py --table 1
    python scripts/significance_greater.py --dataset dl19
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

import pyterrier as pt
import ir_measures

if not pt.started():
    pt.init()

DATASET_IRDS = {
    "dl19":   "irds:msmarco-passage/trec-dl-2019/judged",
    "dl20":   "irds:msmarco-passage/trec-dl-2020/judged",
    "dlhard": "irds:msmarco-passage/trec-dl-hard",
}

METRICS = {
    "AP":    ir_measures.AP(rel=2),
    "nDCG":  ir_measures.nDCG @ 10,
    "RR":    ir_measures.RR(rel=2),
}

TABLE1 = [
    ("TCT-ColBERT",     "TCT-ColBERT"),
    ("+Self-Attention", "GNRR-transformer-single"),
    ("+GCN",            "GNRR-gcn-local"),
    ("+GraphSAGE",      "GNRR-sage-local"),
    ("+GAT",            "GNRR-gat-local"),
    ("+GIN",            "GNRR-gin-local"),
    ("+SignedConv",      "GNRR-signed-local"),
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


def load_run(tag, dataset, out_dir="results/benchmarks"):
    f = Path(out_dir) / tag / dataset / "run.csv.gz"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df.rename(columns={"query_id": "qid", "doc_id": "docno"})
    df["qid"]   = df["qid"].astype(str)
    df["docno"] = df["docno"].astype(str)
    return df


def perquery_scores(run_df, qrels_df, metric):
    """Return a Series indexed by qid with per-query metric values."""
    run_ir = (run_df
              .rename(columns={"qid": "query_id", "docno": "doc_id"})
              .assign(query_id=lambda d: d["query_id"].astype(str),
                      doc_id=lambda d: d["doc_id"].astype(str)))
    qrels_ir = qrels_df.rename(columns={"qid": "query_id", "docno": "doc_id", "label": "relevance"})
    rows = [{"qid": r.query_id, "value": r.value}
            for r in ir_measures.iter_calc([metric], qrels_ir, run_ir)]
    return pd.Series({r["qid"]: r["value"] for r in rows})


def run_experiment(systems, dataset, out_dir, save_path=None):
    pt_ds  = pt.get_dataset(DATASET_IRDS[dataset])
    qrels  = pt_ds.get_qrels()

    # load all runs
    runs, names = {}, []
    for name, tag in systems:
        run = load_run(tag, dataset, out_dir)
        if run is None:
            print(f"  [skip] missing run: {tag}", file=sys.stderr)
            continue
        runs[name] = run
        names.append(name)

    baseline_name = next((n for n in names if n == "TCT-ColBERT"), None)
    if baseline_name is None or len(runs) < 2:
        print("  [skip] baseline or systems missing", file=sys.stderr)
        return None

    # compute per-query scores for every system × metric
    scores = {}   # (name, metric_key) -> Series[qid -> float]
    for name in names:
        for mkey, m in METRICS.items():
            scores[(name, mkey)] = perquery_scores(runs[name], qrels, m)

    rows = []
    for name in names:
        row = {"name": name}
        for mkey in METRICS:
            base_s = scores[(baseline_name, mkey)]
            sys_s  = scores[(name, mkey)]
            # align on common qids
            qids = base_s.index.intersection(sys_s.index)
            b = base_s[qids].values
            s = sys_s[qids].values
            row[f"{mkey} (mean)"] = round(float(s.mean()), 3)
            if name == baseline_name:
                row[f"{mkey} Δ"]   = None
                row[f"{mkey} p"]   = None
                row[f"{mkey} sig"] = ""
            else:
                delta = float(s.mean() - b.mean())
                row[f"{mkey} Δ"] = round(delta, 4)
                if delta > 0:
                    _, p = ttest_rel(s, b, alternative="greater")
                else:
                    # system is worse — one-sided p has no interest, report as 1
                    p = 1.0
                row[f"{mkey} p"]   = round(float(p), 4)
                row[f"{mkey} sig"] = "*" if p < 0.05 else ("†" if p < 0.10 else "")
        rows.append(row)

    result = pd.DataFrame(rows)
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
    ap.add_argument("--save",    default="results/significance_greater")
    args = ap.parse_args()

    datasets = ["dl19", "dl20", "dlhard"] if args.dataset == "all" else [args.dataset]
    tables   = {"1": [1], "2": [2], "both": [1, 2]}[args.table]

    pd.set_option("display.width", 240, "display.max_columns", 50)

    for tbl in tables:
        systems = TABLE1 if tbl == 1 else TABLE2
        print(f"\n{'='*70}")
        print(f"  TABLE {tbl} — one-sided t-test: system > TCT-ColBERT (H1: greater)")
        print(f"  * p<0.05   † p<0.10")
        print(f"{'='*70}")
        for ds in datasets:
            print(f"\n  --- {ds.upper()} ---")
            save_path = f"{args.save}/table{tbl}_{ds}.csv" if args.save else None
            result = run_experiment(systems, ds, args.out_dir, save_path=save_path)
            if result is not None:
                cols = ["name"] + [c for c in result.columns if c != "name"]
                print(result[cols].to_string(index=False))


if __name__ == "__main__":
    main()
