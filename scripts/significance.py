"""
Significance testing for re-ranking comparisons (reviewer request).

Consumes the per-query CSVs written by scripts/evaluate_testset.py
(results/<tag>/<dataset>/perquery.csv) and, for a chosen metric, compares a system
against one or more baselines with:
  - paired two-sided t-test
  - paired bootstrap 95% CI on the mean per-query difference
  - Fisher randomisation (permutation) test

Example:
  python scripts/significance.py --dataset dl19 --metric nDCG@10 \
      --system GNRR-gcn-multistage --baselines TCT-ColBERT GAR graph-smoothing-a0.5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_perquery(out_dir, tag, dataset, metric):
    f = Path(out_dir) / tag / dataset / "perquery.csv"
    if not f.exists():
        raise FileNotFoundError(f"missing per-query results: {f}")
    df = pd.read_csv(f)
    df = df[df["measure"] == metric]
    return df.set_index("query_id")["value"]


def paired_bootstrap_ci(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def randomization_test(a, b, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    diff = a - b
    obs = diff.mean()
    signs = rng.integers(0, 2, size=(n_perm, len(diff))) * 2 - 1
    perm = (signs * np.abs(diff)).mean(axis=1)
    return float((np.abs(perm) >= abs(obs)).mean())


def compare(system, baseline, out_dir, dataset, metric):
    s = load_perquery(out_dir, system, dataset, metric)
    b = load_perquery(out_dir, baseline, dataset, metric)
    common = s.index.intersection(b.index)
    s, b = s.loc[common].values, b.loc[common].values
    diff = s - b
    t, p_t = stats.ttest_rel(s, b)
    lo, hi = paired_bootstrap_ci(diff)
    p_rand = randomization_test(s, b)
    return {
        "system": system, "baseline": baseline, "metric": metric, "dataset": dataset,
        "n_q": len(common),
        "mean_system": round(float(s.mean()), 4),
        "mean_baseline": round(float(b.mean()), 4),
        "delta": round(float(diff.mean()), 4),
        "rel_%": round(100 * float(diff.mean()) / float(b.mean()), 2) if b.mean() else float("nan"),
        "boot_ci_95": f"[{lo:+.4f}, {hi:+.4f}]",
        "p_ttest": round(float(p_t), 4),
        "p_randomization": round(float(p_rand), 4),
        "sig@0.05": "yes" if p_rand < 0.05 else "no",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--metric", default="nDCG@10")
    ap.add_argument("--system", required=True)
    ap.add_argument("--baselines", nargs="+", required=True)
    ap.add_argument("--out_dir", default="results/benchmarks")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    rows = [compare(args.system, b, args.out_dir, args.dataset, args.metric)
            for b in args.baselines]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(df.to_string(index=False))
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save, index=False)
        print(f"\n[saved] {args.save}")


if __name__ == "__main__":
    main()
