"""
GAR + GNRR: use the corpus graph for RECALL (GAR pulls in relevant docs BM25 missed),
then re-rank that expanded pool with the GNRR model.

    BM25 %k  >>  GAR(TCT scorer, corpus_graph, budget)  >>  GNRR(model, corpus_graph)

  --gar_only        evaluate GAR alone (recall baseline, TCT scorer)
  otherwise         GAR followed by the GNRR re-ranker given by --conv_type/--model_path

Evaluated with the same metrics as evaluate_testset, so rows drop into the analysis table.
"""
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier.model import add_ranks

from pyterrier_dr import TctColBert, FlexIndex
from pyterrier_adaptive import GAR


class FastTCTScorer(pt.Transformer):
    """GAR scorer that reuses PRECOMPUTED TCT embeddings (no doc re-encoding, no text
    fetch): score = <query_vec, doc_vec> looked up from the FlexIndex payload. The query
    is encoded once per qid. This makes GAR fast -- it only needs TCT scores to pick which
    graph neighbours to expand, and those vectors already exist in the index."""

    def __init__(self, encoder, payload):
        self.encoder = encoder
        self.payload = payload          # (docno->id lookup, [N, d] vectors)
        self._qcache = {}

    def transform(self, df):
        rows = []
        for qid, g in df.groupby("qid"):
            if qid not in self._qcache:
                self._qcache[qid] = self.encoder.encode_queries(g["query"].iloc[0:1])[0]
            qv = self._qcache[qid]
            scores = []
            for d in g["docno"].tolist():
                try:
                    scores.append(float(self.payload[1][self.payload[0][d]] @ qv))
                except (KeyError, IndexError):
                    scores.append(-1e9)
            sub = g.copy()
            sub["score"] = scores
            rows.append(sub)
        return add_ranks(pd.concat(rows, ignore_index=True))

from src.datamodule import get_corpus_graph
from src.pipeline import EvalConfig, GNRR_Scorer, GNRR, TCT_MODELS
from src.utils import set_determinism_the_old_way
from scripts.evaluate_testset import resolve_dataset, evaluate, get_bm25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dl19", "dl20", "dlhard"])
    ap.add_argument("--conv_type", default="gat")
    ap.add_argument("--modality", default="local")
    ap.add_argument("--graph_type", default="lexical", choices=["semantic", "lexical"])
    ap.add_argument("--corpusgraph_name", default="corpusgraph_bm25_k16")
    ap.add_argument("--model_path", default="")
    ap.add_argument("--hidden_dim", type=int, default=0)
    ap.add_argument("--n_layers", type=int, default=0)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--embedding_name", default="tctcolbert2")
    ap.add_argument("--k", type=int, default=1000, help="BM25 candidate depth")
    ap.add_argument("--budget", type=int, default=1000, help="GAR scoring budget c")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--K_cg", type=int, default=8)
    ap.add_argument("--gar_only", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out_dir", default="results/benchmarks")
    args = ap.parse_args()

    set_determinism_the_old_way(deterministic=True)
    _, topics, qrels = resolve_dataset(args.dataset)

    corpus_name = Path("data") / f"msmarco-index_{args.embedding_name}"
    flex_index = FlexIndex(str(corpus_name))
    graph = get_corpus_graph(args.graph_type, corpus_name, args.K_cg,
                             corpusgraph_name=args.corpusgraph_name, flex_index=flex_index)

    tct = TctColBert(TCT_MODELS[args.embedding_name])
    # fast GAR scorer: reuse precomputed TCT vectors from the index payload
    fast_scorer = FastTCTScorer(tct, flex_index.payload())

    bm25 = get_bm25() % args.k
    gar = GAR(scorer=fast_scorer, corpus_graph=graph,
              num_results=args.budget, batch_size=args.batch, verbose=True)

    if args.gar_only:
        pipe = bm25 >> gar
        tag = args.tag or f"GAR-{args.graph_type}"
    else:
        cfg_path = Path("models/msmarco_data/config_models.json")
        cfg = json.loads(cfg_path.read_text()).get(args.conv_type, {}) if cfg_path.exists() else {}
        cfg.update({"conv_type": args.conv_type, "modality": args.modality,
                    "embedding_name": args.embedding_name, "graph_type": args.graph_type,
                    "heads": args.heads})
        if args.model_path:
            cfg["model_path"] = args.model_path
        if args.hidden_dim:
            cfg["hidden_dim"] = args.hidden_dim
        if args.n_layers:
            cfg["n_layers"] = args.n_layers
        config = EvalConfig.from_dict(cfg)
        scorer = GNRR_Scorer(config, fast=True)
        pipe = bm25 >> gar >> GNRR(scorer, graph, flex_index)
        tag = args.tag or f"GAR+{args.conv_type}-{args.graph_type}"

    print(f"[run] {tag} on {args.dataset}: {len(topics)} topics", flush=True)
    run = pipe.transform(topics)
    evaluate(run, qrels, tag, args.dataset, args.out_dir)


if __name__ == "__main__":
    main()
