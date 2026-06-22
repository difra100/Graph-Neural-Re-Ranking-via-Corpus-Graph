"""
Non-neural graph baselines (reviewer request).

These test whether GNRR's gains come from mere neighbourhood averaging vs. expressive,
learned GNN reasoning. All operate at INFERENCE on the same TCT-ColBERT scores and the
same query-induced subgraph GNRR uses -- no training:

  - tct            : plain TCT-ColBERT scores (reference; equals the TCT baseline).
  - smoothing      : one step of unweighted neighbour mean,   s' = (1-a)s + a*Â s   (Â row-normalised adjacency).
  - knn_interp     : one step of similarity-weighted neighbour mean (corpus-graph cosine weights).
  - ppr            : personalised PageRank propagation, s* = (1-a) (I - a P)^-1 s   (power iteration).
  - label_prop     : iterative score propagation clamped to the TCT seed (a.k.a. label spreading).

Each method re-ranks the BM25 top-k and is evaluated on DL19/DL20/DLHard.
Reuses the subgraph builder + embedding lookup from src/pipeline.py and src/utils.py.
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyterrier as pt
if not pt.started():
    pt.init()

from pyterrier_dr import FlexIndex

from src.datamodule import load_corpus_graph
from src.pipeline import TCT_MODELS
from src.utils import (generate_corpus_subgraph_induced_by_query,
                       build_adjacency_matrix, set_determinism_the_old_way)
from scripts.evaluate_testset import resolve_dataset, evaluate, get_bm25


def row_normalize(A):
    d = A.sum(axis=1, keepdims=True)
    d[d == 0] = 1.0
    return A / d


def propagate(scores, A, method, alpha, iters=20):
    """scores: [n] TCT scores. A: [n,n] (weighted) adjacency. Returns new [n] scores."""
    n = len(scores)
    if n == 0 or A.sum() == 0:
        return scores
    P = row_normalize(A)
    s0 = scores.copy()

    if method in ("smoothing", "knn_interp"):
        return (1 - alpha) * s0 + alpha * (P @ s0)            # one propagation step
    if method == "ppr":
        s = s0.copy()
        for _ in range(iters):
            s = (1 - alpha) * s0 + alpha * (P @ s)            # personalised PageRank
        return s
    if method == "label_prop":
        s = s0.copy()
        for _ in range(iters):
            s = alpha * (P @ s) + (1 - alpha) * s0            # label spreading (clamped seed)
        return s
    raise ValueError(method)


class GraphBaselineScorer(pt.Transformer):
    def __init__(self, flex_index, corpus_graph, method, alpha,
                 embedding_name="tctcolbert", text_field="text"):
        self.flex_index = flex_index
        self.corpus_graph = corpus_graph
        self.method = method
        self.alpha = alpha
        self.text_field = text_field
        self.payload = flex_index.payload()
        self.dataset_retr = pt.get_dataset("irds:msmarco-passage")
        self.add_text = pt.text.get_text(self.dataset_retr, "text")
        from pyterrier_dr import TctColBert
        self.encoder = TctColBert(TCT_MODELS[embedding_name])

    def _scores_and_adj(self, topk):
        docno_to_index = {d: i for i, d in enumerate(topk["docno"].unique())}
        index_to_docno = {i: d for d, i in docno_to_index.items()}
        n = len(docno_to_index)

        q_enc = self.encoder.encode_queries(topk["query"].iloc[0:1])[0]
        d_enc = np.empty((n, q_enc.shape[0]), dtype=q_enc.dtype)
        try:
            for i in range(n):
                d_enc[i] = self.payload[1][self.payload[0][index_to_docno[i]]]
        except (IndexError, KeyError):
            d_enc = self.encoder.encode_docs(topk[self.text_field])
        tct_scores = d_enc @ q_enc                                # dot = TCT score

        subgraph = generate_corpus_subgraph_induced_by_query(
            topk_documents_df=topk, complete_corpus_graph=self.corpus_graph)
        A = build_adjacency_matrix(subgraph, docno_to_index).astype(float)
        return tct_scores, A, index_to_docno

    def transform(self, df):
        out = []
        for qid, group in df.groupby("qid"):
            topk = self.add_text(group.loc[:, ["qid", "query", "docno", "score"]])
            topk = topk.drop_duplicates(subset="docno").reset_index(drop=True)
            if len(topk) <= 1:
                continue
            tct, A, idx2doc = self._scores_and_adj(topk)
            if self.method == "tct":
                new = tct
            else:
                new = propagate(tct, A, self.method, self.alpha)
            sub = pd.DataFrame({"qid": str(qid),
                                "docno": [idx2doc[i] for i in range(len(idx2doc))],
                                "score": new})
            out.append(sub)
        res = pd.concat(out, ignore_index=True)
        res["rank"] = res.groupby("qid")["score"].rank(ascending=False, method="first") - 1
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dl19", "dl20", "dlhard"])
    ap.add_argument("--methods", nargs="+",
                    default=["tct", "smoothing", "knn_interp", "ppr", "label_prop"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--embedding_name", default="tctcolbert")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--K_cg", type=int, default=8)
    ap.add_argument("--out_dir", default="results/benchmarks")
    args = ap.parse_args()

    set_determinism_the_old_way(deterministic=True)
    _, topics, qrels = resolve_dataset(args.dataset)

    corpus_name = Path("data") / f"msmarco-index_{args.embedding_name}"
    flex_index = FlexIndex(str(corpus_name))
    graph = load_corpus_graph(flex_index, corpus_name, args.K_cg)
    bm25_run = (get_bm25() % args.k).transform(topics)

    for method in args.methods:
        scorer = GraphBaselineScorer(flex_index, graph, method, args.alpha,
                                     embedding_name=args.embedding_name)
        run = scorer.transform(bm25_run)
        tag = f"graph-{method}" + ("" if method == "tct" else f"-a{args.alpha}")
        evaluate(run, qrels, tag, args.dataset, args.out_dir)


if __name__ == "__main__":
    main()
