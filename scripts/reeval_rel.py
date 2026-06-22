"""Recompute metrics at a given relevance threshold from the SAVED runs (no model re-run).

For every results/benchmarks/<tag>/<ds>/run.csv.gz, recompute AP/RR/P@3/R@1000 at
rel>=<rel> (nDCG@10 is graded and unchanged) and write aggregate_rel<rel>.csv +
perquery_rel<rel>.csv next to it. Rationale: MS MARCO training labels are binary
(grade 1), so rel>=1 matches what the models were trained to rank.
"""
import os
import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyterrier as pt
if not pt.started():
    pt.init()
import ir_measures
from ir_measures import nDCG, AP, RR, P, R

from scripts.evaluate_testset import resolve_dataset, qrels_to_df, DATASETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel", type=int, default=1)
    ap.add_argument("--out_dir", default="results/benchmarks")
    args = ap.parse_args()
    rel = args.rel
    measures = [nDCG @ 10, P(rel=rel) @ 3, AP(rel=rel), RR(rel=rel), R(rel=rel) @ 1000]

    qrels_cache = {ds: qrels_to_df(resolve_dataset(ds)[2]) for ds in DATASETS}

    n = 0
    for run_f in Path(args.out_dir).glob("*/*/run.csv.gz"):
        ds = run_f.parent.name
        tag = run_f.parent.parent.name
        if ds not in qrels_cache:
            continue
        run_df = pd.read_csv(run_f)
        run_df["query_id"] = run_df["query_id"].astype(str)
        run_df["doc_id"] = run_df["doc_id"].astype(str)
        run_df["score"] = run_df["score"].astype(float)

        agg = ir_measures.calc_aggregate(measures, qrels_cache[ds], run_df)
        row = {str(m): float(v) for m, v in agg.items()}
        row.update({"pipeline": tag, "dataset": ds})
        pd.DataFrame([row]).to_csv(run_f.parent / f"aggregate_rel{rel}.csv", index=False)

        perq = ir_measures.iter_calc(measures, qrels_cache[ds], run_df)
        pd.DataFrame([{"query_id": m.query_id, "measure": str(m.measure),
                       "value": float(m.value)} for m in perq]
                     ).to_csv(run_f.parent / f"perquery_rel{rel}.csv", index=False)
        n += 1
    print(f"recomputed rel>={rel} metrics for {n} runs")


if __name__ == "__main__":
    main()
