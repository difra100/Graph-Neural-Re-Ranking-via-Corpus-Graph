"""Generate RESULTS_REPORT.md: original-paper (v1) + new (v2) single/multi/GAR + runtimes.

Reads the new v2 numbers from results/benchmarks/<tag>/<ds>/aggregate.csv (no manual
transcription); the original-paper TCT-v1 semantic-GNN numbers and the measured runtimes
are hard-coded constants. Bad/uncertain models are emitted blank (—) for retraining.
"""
import os
import argparse
from pathlib import Path
import pandas as pd

DS = ["dl19", "dl20", "dlhard"]
OUT_DIR = "results/benchmarks"
DASH = "—"
# set in main() once --rel is known. NOTE: ir_measures omits the rel suffix when it
# equals the default (rel=1), so columns are "AP" at rel=1 but "AP(rel=2)" at rel=2.
REL = 2
SUFFIX = "(rel=2)"
AGG_NAME = "aggregate.csv"
METRICS = ["AP(rel=2)", "RR(rel=2)", "P(rel=2)@3", "nDCG@10"]

# ---- original paper, TCT-ColBERT v1, semantic kNN graph, single-stage, no GAR ----
PAPER_V1 = {  # name -> {ds: [AP, RR, P@3, nDCG@10]}
    "BM25":            {"dl19": [.286, .642, .473, .480], "dl20": [.293, .619, .463, .494], "dlhard": [.147, .422, .240, .274]},
    "TCT-ColBERT":     {"dl19": [.430, .843, .667, .685], "dl20": [.453, .817, .691, .680], "dlhard": [.230, .538, .353, .373]},
    "TCT+GCN":         {"dl19": [.455, .858, .713, .702], "dl20": [.470, .840, .673, .695], "dlhard": [.242, .559, .367, .386]},
    "TCT+GraphSAGE":   {"dl19": [.434, .850, .728, .689], "dl20": [.454, .837, .667, .685], "dlhard": [.223, .531, .366, .379]},
    "TCT+GAT":         {"dl19": [.442, .852, .698, .690], "dl20": [.453, .826, .685, .682], "dlhard": [.219, .542, .364, .376]},
    "TCT+GIN":         {"dl19": [.449, .853, .729, .691], "dl20": [.455, .793, .642, .675], "dlhard": [.215, .490, .333, .357]},
    "TCT+SignedConv":  {"dl19": [.426, .828, .698, .675], "dl20": [.445, .813, .685, .676], "dlhard": [.218, .509, .353, .366]},
}

# display name -> result tag (None = leave blank: bad model, needs retrain)
SINGLE = [
    ("TCT-ColBERT (v2)",         "TCT-ColBERT-v2"),
    ("TCT+GCN (lexical)",        None),
    ("TCT+GraphSAGE (lexical)",  "sage-lexical"),
    ("TCT+GAT (lexical)",        "gat-lexical"),
    ("TCT+GIN (lexical)",        "gin-lexical"),
    ("TCT+SignedConv (lexical)", "signed-lexical"),
    ("TCT+EdgeGAT (semantic)",   "edgegat-semantic"),
    ("TCT+EdgeGAT (lexical)",    "edgegat-lexical"),
    ("Self-attention (1-stage)", "selfattn-1stage"),
]
MULTI = [
    ("TCT-ColBERT (v2)",         "TCT-ColBERT-v2"),
    ("TCT+GCN (lexical)",        None),
    ("TCT+GraphSAGE (lexical)",  "sage-lexical-ms200"),
    ("TCT+GAT (lexical)",        "gat-lexical-ms200"),
    ("TCT+GIN (lexical)",        "gin-lexical-ms200"),
    ("TCT+SignedConv (lexical)", "signed-lexical-ms200"),
    ("TCT+EdgeGAT (semantic)",   "edgegat-semantic-ms200"),
    ("TCT+EdgeGAT (lexical)",    "edgegat-lexical-ms200"),
    ("Self-attention (multi)",   "selfattn-ms200"),
]
GAR_ROWS = [
    ("TCT-ColBERT (v2)",          "TCT-ColBERT-v2"),
    ("GAR (recall only)",         "GAR-semantic"),
    ("GAR + GAT (lexical)",       "GAR+gat-lexical"),
    ("GAR + SignedConv (lexical)","GAR+signed-lexical"),
    ("GAR + EdgeGAT (semantic)",  "GAR+edgegat-semantic"),
]

RUNTIME_CSV = "results/efficiency/runtimes.csv"


def fmt_params(p):
    try:
        p = int(float(p))
    except (ValueError, TypeError):
        return DASH
    return f"{p/1e6:.2f} M" if p >= 1e6 else f"{p/1e3:.0f} K"


def load_runtimes():
    if not os.path.exists(RUNTIME_CSV):
        return []
    df = pd.read_csv(RUNTIME_CSV)
    rows = []
    for _, r in df.iterrows():
        def num(x, d=1):
            try:
                return f"{float(x):.{d}f}"
            except (ValueError, TypeError):
                return DASH
        rows.append((r["model"], fmt_params(r["params"]), num(r["ms_single"]),
                     num(r["ms_multi"]), num(r["mem_single_MB"], 0), num(r["mem_multi_MB"], 0)))
    return rows


def load(tag):
    if tag is None:
        return None
    row = {}
    for ds in DS:
        f = Path(OUT_DIR) / tag / ds / AGG_NAME
        if f.exists():
            r = pd.read_csv(f).iloc[0]
            row[ds] = r
    return row or None


def cell(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else DASH


def sub_table(ds, rows, extra_recall=False):
    cols = METRICS + ([f"R{SUFFIX}@1000"] if extra_recall else [])
    head = "| Pipeline | " + " | ".join(["AP", "RR", "P@3", "nDCG@10"] + (["R@1000"] if extra_recall else [])) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [f"**{ds.upper()}**", "", head, sep]
    for name, data in rows:
        if data is None:
            vals = [DASH] * len(cols)
        else:
            r = data.get(ds)
            vals = [cell(r[m]) if (r is not None and m in r) else DASH for m in cols]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def paper_sub_table(ds):
    head = "| Pipeline | AP | RR | P@3 | nDCG@10 |"
    sep = "|---|---|---|---|---|"
    lines = [f"**{ds.upper()}**", "", head, sep]
    for name, d in PAPER_V1.items():
        a, rr, p, n = d[ds]
        lines.append(f"| {name} | {a:.3f} | {rr:.3f} | {p:.3f} | {n:.3f} |")
    lines.append("")
    return "\n".join(lines)


def main():
    global REL, AGG_NAME, METRICS, SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    REL = args.rel
    AGG_NAME = "aggregate.csv" if REL == 2 else f"aggregate_rel{REL}.csv"
    SUFFIX = "" if REL == 1 else f"(rel={REL})"   # ir_measures drops the default rel=1
    METRICS = [f"AP{SUFFIX}", f"RR{SUFFIX}", f"P{SUFFIX}@3", "nDCG@10"]
    out_file = args.out or ("RESULTS_REPORT.md" if REL == 2 else f"RESULTS_REPORT_rel{REL}.md")

    single = [(n, load(t)) for n, t in SINGLE]
    multi = [(n, load(t)) for n, t in MULTI]
    gar = [(n, load(t)) for n, t in GAR_ROWS]

    out = []
    A = out.append
    A(f"# GNRR — Results Report (relevance ≥ {REL})\n")
    A(f"All binarised metrics (AP/RR/P@3/R@1000) use relevance grade ≥ {REL}; nDCG@10 is "
      "graded (threshold-independent). MS MARCO training labels are binary (grade 1), so "
      "**rel≥1 is the eval that matches what the models were trained to rank**; rel≥2 is the "
      "stricter setting used in the original paper.\n")
    A("Two encoder versions appear and must not be compared across blocks: **v1** = "
      "`tct_colbert-msmarco` (original paper), **v2** = `tct_colbert-v2-hnp-msmarco` (new). "
      "`—` = result omitted (model under-trained / diverged; to be retrained).\n")
    A("Graph: corpus kNN graph (k capped at 8). **semantic** = TCT cosine graph; "
      "**lexical** = BM25 graph (`corpusgraph_bm25_k16`). Hardware: RTX 3090 Ti.\n")

    if REL == 2:
        A("\n## 1. Original paper (TCT-ColBERT **v1**, semantic graph, single-stage, no GAR)\n")
        A("_Kept verbatim from the original submission._\n")
        for ds in DS:
            A(paper_sub_table(ds))
    else:
        A("\n## 1. Original paper (TCT-ColBERT v1)\n")
        A(f"_Omitted: the paper reports rel≥2 only and no v1 runs were saved to recompute at "
          f"rel≥{REL}. See RESULTS_REPORT.md for the rel≥2 paper block._\n")

    A("\n## 2. New experiments — single-stage (TCT-ColBERT **v2**, re-rank BM25 top-1000)\n")
    for ds in DS:
        A(sub_table(ds, single))

    A("\n## 3. New experiments — multi-stage (re-rank TCT top-200, backfill by TCT)\n")
    for ds in DS:
        A(sub_table(ds, multi))

    A("\n## 4. GAR (graph for recall) + GNRR — complementarity (TCT-ColBERT **v2**)\n")
    A("_GAR expands the candidate pool via the corpus graph (recall), then the model "
      "re-ranks. Note the recall column R@1000._\n")
    for ds in DS:
        A(sub_table(ds, gar, extra_recall=True))

    A("\n## 5. Inference cost (DL19, per query)\n")
    A("Online re-ranking cost = subgraph extraction + model forward (excludes BM25 + the "
      "one-off query encoding shared by all systems).\n")
    A("| Model | Params | Online single-stage (ms) | Online multi-stage (ms) | Peak GPU mem single / multi (MB) |")
    A("|---|---|---|---|---|")
    for name, p, ms_s, ms_m, mem_s, mem_m in load_runtimes():
        A(f"| {name} | {p} | {ms_s} | {ms_m} | {mem_s} / {mem_m} |")
    A("\n**GAR recall step:** ≈ 165 ms/query (graph neighbour lookups + dot-product scoring "
      "over the expanded pool; no document encoding). It is the dominant online cost but is "
      "additive/complementary to the cheap (~15–35 ms) GNN re-ranking.\n")
    A("**Offline corpus graph** (`corpusgraph_k16`, MS MARCO 8.8M docs): edges ≈ 566 MB, "
      "weights ≈ 283 MB; built once, amortised across all queries.\n")

    A("\n## 6. Notes\n")
    A("- **Multi-stage ≈ single-stage** in accuracy for the strong models, but ~2× cheaper "
      "online (re-ranks 200 vs 1000) — its value is efficiency, not quality.\n")
    A("- **GAR is the clear win and is complementary**: it lifts R@1000 by ~+7–9 points and "
      "gives the best nDCG@10 on all three sets; adding the GNN on the expanded pool further "
      "improves AP/P@3 (e.g. GAR+EdgeGAT best AP on DL19; GAR+GAT best DLHard AP/RR/P@3).\n")
    A("- **Blanks (GCN-lexical v2):** diverged in training (nDCG ≈ 0.27); to be retrained.\n")
    A("- **Missing semantic-v2 GNNs** (SAGE/GAT/GIN/Signed): not retrained on v2; the v1 "
      "numbers in §1 stand in for the semantic graph.\n")

    Path(out_file).write_text("\n".join(out))
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
