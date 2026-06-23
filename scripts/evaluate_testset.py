"""
Evaluate a re-ranking pipeline on a TREC test set (DL19 / DL20 / DLHard).

Replaces the notebook-based evaluation (src/test.ipynb, build_datasets.ipynb) with a
clean, scriptable harness. Supports:
  - BM25                 (--pipeline bm25)
  - TCT-ColBERT re-rank  (--pipeline tct)
  - GNRR variants        (--pipeline gnrr  --conv_type {gcn,gat,sage,gin,signed,transformer}
                                            --modality {local,multistage,global,single})

Writes per-query and aggregate metric CSVs under results/<tag>/<dataset>/ and a TREC run.

Example:
  python scripts/evaluate_testset.py --dataset dl19 --pipeline gnrr --conv_type gcn \
      --modality multistage \
      --model_path models/msmarco_data/old_models/hadamard_2_789_0.01_0_128_0.6_gcn_multistage_1_tctcolbert.pt
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyterrier as pt
if not pt.started():
    pt.init()

import ir_measures
from ir_measures import nDCG, AP, RR, P, R

from src.utils import set_determinism_the_old_way

# ---------------------------------------------------------------------------
# Test set registry.
# ---------------------------------------------------------------------------
DATASETS = {
    "dl19":   "irds:msmarco-passage/trec-dl-2019/judged",
    "dl20":   "irds:msmarco-passage/trec-dl-2020/judged",
    "dlhard": "irds:msmarco-passage/trec-dl-hard",
}

EMBED_BY_DEFAULT = "tctcolbert"
MEASURES = [nDCG @ 10, P(rel=2) @ 3, AP(rel=2), RR(rel=2), R(rel=2) @ 1000]


def get_bm25():
    """Prebuilt MS MARCO passage BM25 (downloads terrier_stemmed index on first use)."""
    try:
        Retr = pt.terrier.Retriever
    except AttributeError:
        Retr = pt.BatchRetrieve
    return Retr.from_dataset("msmarco_passage", "terrier_stemmed", wmodel="BM25")


def resolve_dataset(name):
    ds = pt.get_dataset(DATASETS[name])
    topics = ds.get_topics()
    qrels = ds.get_qrels()
    return ds, topics, qrels


def run_to_measure_df(run):
    """Coerce a pyterrier run into the (query_id, doc_id, score) frame ir_measures wants."""
    df = run.rename(columns={"qid": "query_id", "docno": "doc_id"}).copy()
    df["query_id"] = df["query_id"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)
    df["score"] = df["score"].astype(float)
    return df[["query_id", "doc_id", "score"]]


def qrels_to_df(qrels):
    df = qrels.rename(columns={"qid": "query_id", "docno": "doc_id",
                               "label": "relevance"}).copy()
    df["query_id"] = df["query_id"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)
    df["relevance"] = df["relevance"].astype(int)
    return df[["query_id", "doc_id", "relevance"]]


def evaluate(run, qrels, tag, dataset, out_dir):
    qrels_df = qrels_to_df(qrels)
    run_df = run_to_measure_df(run)

    agg = ir_measures.calc_aggregate(MEASURES, qrels_df, run_df)
    perq = list(ir_measures.iter_calc(MEASURES, qrels_df, run_df))

    out = Path(out_dir) / tag / dataset
    out.mkdir(parents=True, exist_ok=True)

    agg_row = {str(m): float(v) for m, v in agg.items()}
    agg_row.update({"pipeline": tag, "dataset": dataset})
    pd.DataFrame([agg_row]).to_csv(out / "aggregate.csv", index=False)

    pd.DataFrame([{"query_id": m.query_id, "measure": str(m.measure),
                   "value": float(m.value)} for m in perq]
                 ).to_csv(out / "perquery.csv", index=False)

    run_df.to_csv(out / "run.csv.gz", index=False, compression="gzip")
    print(f"\n=== {tag} on {dataset} ===", flush=True)
    for k, v in sorted(agg_row.items()):
        if k not in ("pipeline", "dataset"):
            print(f"  {k:14s}: {v:.4f}")
    return agg_row


def build_pipeline(args):
    bm25 = get_bm25() % args.k

    if args.pipeline == "bm25":
        return bm25, "BM25"

    if args.pipeline == "tct":
        from pyterrier_dr import TctColBert
        from src.pipeline import TCT_MODELS
        scorer = TctColBert(TCT_MODELS[args.embedding_name])
        dataset_retr = pt.get_dataset("irds:msmarco-passage")
        get_text = pt.text.get_text(dataset_retr, "text")
        return bm25 >> get_text >> scorer, "TCT-ColBERT"

    if args.pipeline == "gnrr":
        from pyterrier_dr import FlexIndex
        from src.datamodule import get_corpus_graph
        from src.pipeline import EvalConfig, GNRR_Scorer, GNRR

        # config from config_models.json unless overridden on the CLI
        cfg_dict = {}
        cfg_path = Path("models/msmarco_data/config_models.json")
        if cfg_path.exists():
            cfg_dict = json.loads(cfg_path.read_text()).get(args.conv_type, {})
        cfg_dict.update({
            "conv_type": args.conv_type,
            "modality": args.modality,
            "embedding_name": args.embedding_name,
            "disable_gnn": args.disable_gnn,
            "aggr": args.aggr,
            "heads": args.heads,
        })
        if args.model_path:
            cfg_dict["model_path"] = args.model_path
        if args.hidden_dim:
            cfg_dict["hidden_dim"] = args.hidden_dim
        if args.n_layers:
            cfg_dict["n_layers"] = args.n_layers
        if args.n_layers_mlp:
            cfg_dict["n_layers_mlp"] = args.n_layers_mlp
        if args.K_multistage:
            cfg_dict["K_multistage"] = args.K_multistage
        config = EvalConfig.from_dict(cfg_dict)
        print("[config]", config, flush=True)

        corpus_name = Path("data") / f"msmarco-index_{args.embedding_name}"
        flex_index = FlexIndex(str(corpus_name))
        graph = get_corpus_graph(args.graph_type, corpus_name, args.K_cg,
                                 corpusgraph_name=args.corpusgraph_name, flex_index=flex_index)

        scorer = GNRR_Scorer(config, fast=args.fast, restrict_k=args.restrict_k)
        gtag = "" if args.graph_type == "semantic" else f"-{args.graph_type}"
        mtag = f"-ms{args.restrict_k}" if args.restrict_k else ""
        tag = f"GNRR-{args.conv_type}-{args.modality}{gtag}{mtag}" + ("-noGNN" if args.disable_gnn else "")
        return bm25 >> GNRR(scorer, graph, flex_index), tag

    raise ValueError(args.pipeline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--pipeline", required=True, choices=["bm25", "tct", "gnrr"])
    ap.add_argument("--conv_type", default="gcn")
    ap.add_argument("--modality", default="multistage")
    ap.add_argument("--model_path", default="")
    ap.add_argument("--embedding_name", default=EMBED_BY_DEFAULT)
    ap.add_argument("--k", type=int, default=1000, help="BM25 candidate depth")
    ap.add_argument("--K_cg", type=int, default=8, help="corpus-graph neighbours used")
    ap.add_argument("--graph_type", default="semantic", choices=["semantic", "lexical"])
    ap.add_argument("--corpusgraph_name", default="corpusgraph_bm25_k16")
    ap.add_argument("--K_multistage", type=int, default=0, help="0 = use config default")
    ap.add_argument("--restrict_k", type=int, default=0,
                    help="multi-stage cascade: model re-ranks only TCT top-k (0 = full 1-stage)")
    ap.add_argument("--hidden_dim", type=int, default=0)
    ap.add_argument("--n_layers", type=int, default=0)
    ap.add_argument("--n_layers_mlp", type=int, default=0)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--aggr", default="hadamard")
    ap.add_argument("--fast", action="store_true", default=True)
    ap.add_argument("--no-fast", dest="fast", action="store_false")
    ap.add_argument("--disable_gnn", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out_dir", default="results/benchmarks")
    args = ap.parse_args()

    set_determinism_the_old_way(deterministic=True)

    _, topics, qrels = resolve_dataset(args.dataset)
    pipe, default_tag = build_pipeline(args)
    tag = args.tag or default_tag

    print(f"[run] {tag} on {args.dataset}: {len(topics)} topics, {len(qrels)} qrels", flush=True)
    run = pipe.transform(topics)
    evaluate(run, qrels, tag, args.dataset, args.out_dir)


if __name__ == "__main__":
    main()
