"""Collect all per-(pipeline, dataset) aggregate.csv files into one wide table."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/benchmarks")
    ap.add_argument("--table_out", default="results/benchmarks/table1.csv")
    args = ap.parse_args()

    rows = []
    for agg in Path(args.out_dir).glob("*/*/aggregate.csv"):
        rows.append(pd.read_csv(agg))
    if not rows:
        print(f"[collect] no aggregate.csv found under {args.out_dir}")
        return

    df = pd.concat(rows, ignore_index=True)
    metric_cols = [c for c in df.columns if c not in ("pipeline", "dataset")]
    long = df.melt(id_vars=["pipeline", "dataset"], value_vars=metric_cols,
                   var_name="metric", value_name="value")
    wide = long.pivot_table(index="pipeline", columns=["dataset", "metric"],
                            values="value")
    wide = wide.round(3)
    Path(args.table_out).parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.table_out)
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(wide)
    print(f"\n[collect] wrote {args.table_out}")


if __name__ == "__main__":
    main()
