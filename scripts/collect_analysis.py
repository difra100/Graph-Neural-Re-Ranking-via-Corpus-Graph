"""Build the two analysis tables (single-stage and multi-stage) from results/benchmarks."""
import argparse
from pathlib import Path
import pandas as pd

DATASETS = ["dl19", "dl20", "dlhard"]
SHOW = ["AP(rel=2)", "nDCG@10", "RR(rel=2)", "P(rel=2)@3"]
REFS = ["BM25", "TCT-ColBERT-v2"]


def load(out_dir):
    rows = []
    for agg in Path(out_dir).glob("*/*/aggregate.csv"):
        df = pd.read_csv(agg)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def make_table(df, pipelines):
    df = df[df["pipeline"].isin(pipelines)].copy()
    long = df.melt(id_vars=["pipeline", "dataset"],
                   value_vars=[c for c in SHOW if c in df.columns],
                   var_name="metric", value_name="value")
    wide = long.pivot_table(index="pipeline", columns=["dataset", "metric"], values="value")
    # order datasets/metrics + rows
    cols = [(d, m) for d in DATASETS for m in SHOW if (d, m) in wide.columns]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(cols))
    order = [p for p in pipelines if p in wide.index]
    return wide.reindex(order).round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/benchmarks")
    ap.add_argument("--kms", type=int, default=200)
    ap.add_argument("--save_dir", default="results/analysis")
    args = ap.parse_args()

    df = load(args.out_dir)
    if df.empty:
        print("no results found"); return
    ms = f"-ms{args.kms}"
    all_tags = sorted(df["pipeline"].unique())

    single = REFS + [t for t in all_tags if t not in REFS and not t.endswith(ms)]
    multi = REFS + [t for t in all_tags if t.endswith(ms)]

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240, "display.max_columns", 60)

    t1 = make_table(df, single)
    t2 = make_table(df, multi)
    t1.to_csv(f"{args.save_dir}/table_single_stage.csv")
    t2.to_csv(f"{args.save_dir}/table_multi_stage.csv")

    print("\n========== TABLE 1: SINGLE-STAGE (re-rank BM25 top-1000) ==========")
    print(t1.to_string())
    print(f"\n========== TABLE 2: MULTI-STAGE (re-rank TCT top-{args.kms}) ==========")
    print(t2.to_string())
    print(f"\n[saved] {args.save_dir}/table_single_stage.csv , table_multi_stage.csv")


if __name__ == "__main__":
    main()
