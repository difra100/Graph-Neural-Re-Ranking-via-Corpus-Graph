"""
Graph controls + subgraph density (reviewer requests).

Takes a TRAINED GNRR model and re-evaluates it at inference with different edge sets:
  - original : the real query-induced subgraph
  - empty    : no edges (isolates the scorer/capacity contribution)
  - random   : degree-matched random edges (isolates *which* edges matter)
This isolates whether cross-document propagation over the real corpus-graph structure
specifically helps, vs. mere extra capacity / arbitrary connectivity.

Also reports subgraph DENSITY per test set (mean nodes, mean edges, mean degree,
isolated-node fraction).

Example:
  python scripts/run_controls.py --dataset dl19 --conv_type gcn --modality multistage \
      --model_path models/msmarco_data/old_models/hadamard_2_789_0.01_0_128_0.6_gcn_multistage_1_tctcolbert.pt
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyterrier as pt
if not pt.started():
    pt.init()

from pyterrier_dr import FlexIndex
from src.datamodule import load_corpus_graph
from src.pipeline import EvalConfig, GNRR_Scorer, GNRR
from src.utils import set_determinism_the_old_way
from scripts.evaluate_testset import resolve_dataset, evaluate, get_bm25


def density_report(stats):
    nodes = np.array([s[0] for s in stats], dtype=float)
    edges = np.array([s[1] for s in stats], dtype=float)  # directed (COO) count
    undirected = edges / 2.0
    mean_deg = np.divide(edges, nodes, out=np.zeros_like(edges), where=nodes > 0)
    return {
        "queries": len(stats),
        "mean_nodes": round(float(nodes.mean()), 1),
        "mean_undirected_edges": round(float(undirected.mean()), 1),
        "mean_degree": round(float(mean_deg.mean()), 3),
        "mean_density": round(float((undirected / (nodes * (nodes - 1) / 2)
                              ).clip(0, 1).mean()), 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dl19", "dl20", "dlhard"])
    ap.add_argument("--conv_type", default="gcn")
    ap.add_argument("--modality", default="multistage")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--embedding_name", default="tctcolbert")
    ap.add_argument("--modes", nargs="+", default=["original", "empty", "random"])
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--K_cg", type=int, default=8)
    ap.add_argument("--out_dir", default="results/controls")
    args = ap.parse_args()

    set_determinism_the_old_way(deterministic=True)
    _, topics, qrels = resolve_dataset(args.dataset)

    corpus_name = Path("data") / f"msmarco-index_{args.embedding_name}"
    flex_index = FlexIndex(str(corpus_name))
    graph = load_corpus_graph(flex_index, corpus_name, args.K_cg)
    bm25 = get_bm25() % args.k

    cfg_path = Path("models/msmarco_data/config_models.json")
    cfg_dict = json.loads(cfg_path.read_text()).get(args.conv_type, {}) if cfg_path.exists() else {}
    cfg_dict.update({"conv_type": args.conv_type, "modality": args.modality,
                     "embedding_name": args.embedding_name, "model_path": args.model_path})
    config = EvalConfig.from_dict(cfg_dict)

    density = None
    for mode in args.modes:
        scorer = GNRR_Scorer(config, fast=True, edge_mode=mode)
        run = (bm25 >> GNRR(scorer, graph, flex_index)).transform(topics)
        evaluate(run, qrels, f"control-{args.conv_type}-{mode}", args.dataset, args.out_dir)
        if mode == "original":
            density = density_report(scorer.density_stats)

    if density:
        out = Path(args.out_dir) / "density"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{args.dataset}.json").write_text(json.dumps(density, indent=2))
        print(f"\n[density] {args.dataset}: {density}")


if __name__ == "__main__":
    main()
