"""
Efficiency measurement (reviewer request): back the O(c*K*L) claim with numbers.

ONLINE: per-query wall-clock for subgraph extraction + GNN forward (excludes the
one-off BM25 retrieval and TCT encoding, which every baseline shares), at K=1000.
OFFLINE: on-disk size of the precomputed corpus graph (edges + weights) and node count.

Example:
  python scripts/run_efficiency.py --dataset dl19 --conv_type gcn --modality multistage \
      --model_path models/msmarco_data/old_models/hadamard_2_789_0.01_0_128_0.6_gcn_multistage_1_tctcolbert.pt
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyterrier as pt
if not pt.started():
    pt.init()

from pyterrier_dr import FlexIndex
from src.datamodule import load_corpus_graph
from src.datamodule import get_corpus_graph
from src.pipeline import EvalConfig, build_model
from src.utils import (generate_corpus_subgraph_induced_by_query,
                       build_adjacency_matrix, adjacency_matrix_to_coo,
                       compute_output, get_n_params, set_determinism_the_old_way)
from scripts.evaluate_testset import resolve_dataset, get_bm25


def offline_graph_size(corpus_name, K_cg):
    info = {}
    for sub in Path(corpus_name).glob("corpusgraph_k*"):
        edges = sub / "edges.u32.np"
        weights = sub / "weights.f16.np"
        if edges.exists():
            info[sub.name] = {
                "edges_MB": round(edges.stat().st_size / 1e6, 1),
                "weights_MB": round(weights.stat().st_size / 1e6, 1) if weights.exists() else 0,
            }
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dl19", choices=["dl19", "dl20", "dlhard"])
    ap.add_argument("--conv_type", default="gcn")
    ap.add_argument("--modality", default="multistage")
    ap.add_argument("--model_path", default="", help="omit to use config_models.json")
    ap.add_argument("--embedding_name", default="tctcolbert")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--K_cg", type=int, default=8)
    ap.add_argument("--graph_type", default="semantic", choices=["semantic", "lexical"])
    ap.add_argument("--corpusgraph_name", default="corpusgraph_bm25_k16")
    ap.add_argument("--hidden_dim", type=int, default=0)
    ap.add_argument("--n_layers", type=int, default=0)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--restrict_k", type=int, default=0, help="time multi-stage re-rank of top-k (0=full)")
    ap.add_argument("--out_dir", default="results/efficiency")
    args = ap.parse_args()

    set_determinism_the_old_way(deterministic=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, topics, _ = resolve_dataset(args.dataset)

    corpus_name = Path("data") / f"msmarco-index_{args.embedding_name}"
    flex_index = FlexIndex(str(corpus_name))
    graph = get_corpus_graph(args.graph_type, corpus_name, args.K_cg,
                             corpusgraph_name=args.corpusgraph_name, flex_index=flex_index)
    payload = flex_index.payload()

    cfg_path = Path("models/msmarco_data/config_models.json")
    cfg_dict = json.loads(cfg_path.read_text()).get(args.conv_type, {}) if cfg_path.exists() else {}
    cfg_dict.update({"conv_type": args.conv_type, "modality": args.modality,
                     "embedding_name": args.embedding_name, "heads": args.heads})
    if args.model_path:
        cfg_dict["model_path"] = args.model_path
    if args.hidden_dim:
        cfg_dict["hidden_dim"] = args.hidden_dim
    if args.n_layers:
        cfg_dict["n_layers"] = args.n_layers
    config = EvalConfig.from_dict(cfg_dict)
    model = build_model(config, 768, device)
    n_params = get_n_params(model)

    dataset_retr = pt.get_dataset("irds:msmarco-passage")
    add_text = pt.text.get_text(dataset_retr, "text")
    bm25 = get_bm25() % args.k
    run = bm25.transform(topics)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    peak_mem_mb = 0.0

    extract_ms, forward_ms = [], []
    for qid, group in run.groupby("qid"):
        topk = add_text(group.loc[:, ["qid", "query", "docno", "score"]])
        topk = topk.drop_duplicates(subset="docno").reset_index(drop=True)
        if args.restrict_k:                       # multi-stage: time re-ranking of top-k only
            topk = topk.head(args.restrict_k).reset_index(drop=True)
        if len(topk) <= 1:
            continue
        idx = {d: i for i, d in enumerate(topk["docno"].unique())}
        i2d = {i: d for d, i in idx.items()}
        d_enc = np.stack([payload[1][payload[0][i2d[i]]] for i in range(len(idx))])
        q_enc = d_enc[0:1]  # placeholder; query enc excluded from timing

        t0 = time.perf_counter()
        sg = generate_corpus_subgraph_induced_by_query(topk_documents_df=topk,
                                                       complete_corpus_graph=graph)
        A = adjacency_matrix_to_coo(build_adjacency_matrix(sg, idx)).to(device)
        torch.cuda.synchronize() if device == "cuda" else None
        t1 = time.perf_counter()

        x = torch.from_numpy(d_enc).float().unsqueeze(0).to(device)
        qf = torch.from_numpy(q_enc).float().unsqueeze(0).to(device)
        with torch.no_grad():
            _ = compute_output(x, A.unsqueeze(0), qf, model, config.aggr, config.conv_type)
        torch.cuda.synchronize() if device == "cuda" else None
        t2 = time.perf_counter()

        extract_ms.append((t1 - t0) * 1e3)
        forward_ms.append((t2 - t1) * 1e3)
        if device == "cuda":
            peak_mem_mb = max(peak_mem_mb, torch.cuda.max_memory_allocated() / 1e6)

    report = {
        "dataset": args.dataset, "conv_type": args.conv_type, "modality": args.modality,
        "graph_type": args.graph_type, "K": args.k, "n_queries": len(forward_ms),
        "model_params": n_params,
        "peak_gpu_mem_MB": round(peak_mem_mb, 1),
        "subgraph_extract_ms": {"mean": round(np.mean(extract_ms), 2),
                                 "median": round(np.median(extract_ms), 2)},
        "gnn_forward_ms": {"mean": round(np.mean(forward_ms), 2),
                            "median": round(np.median(forward_ms), 2)},
        "online_total_ms": {"mean": round(np.mean(extract_ms) + np.mean(forward_ms), 2)},
        "offline_corpus_graph": offline_graph_size(corpus_name, args.K_cg),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.dataset}_{args.conv_type}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
