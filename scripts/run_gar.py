"""
GAR baseline: Graph-based Adaptive Re-ranking (MacAvaney et al., CIKM 2022) -- ref [18].

This is the closest prior work GNRR builds on, and the single most-requested missing
baseline. It is INFERENCE-ONLY: GAR uses the pretrained TCT-ColBERT scorer plus the
SAME offline corpus graph GNRR uses to alternate between scoring the initial pool and
scoring graph neighbours, improving recall/precision without any training.

Pipeline:  BM25 % budget  >>  GAR(TCT-ColBERT scorer, corpus_graph)

Evaluated on DL19/DL20/DLHard with the same metrics as evaluate_testset.py, so the
numbers drop straight into the Table-1 comparison.
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyterrier as pt
if not pt.started():
    pt.init()

from pyterrier_dr import TctColBert, FlexIndex
from pyterrier_adaptive import GAR

from src.datamodule import load_corpus_graph
from src.pipeline import TCT_MODELS
from src.utils import set_determinism_the_old_way
from scripts.evaluate_testset import resolve_dataset, evaluate, get_bm25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dl19", "dl20", "dlhard"])
    ap.add_argument("--embedding_name", default="tctcolbert")
    ap.add_argument("--budget", type=int, default=1000, help="GAR scoring budget c")
    ap.add_argument("--batch", type=int, default=16, help="GAR batch size")
    ap.add_argument("--K_cg", type=int, default=8)
    ap.add_argument("--k", type=int, default=1000, help="BM25 candidate depth")
    ap.add_argument("--out_dir", default="results/benchmarks")
    args = ap.parse_args()

    set_determinism_the_old_way(deterministic=True)

    _, topics, qrels = resolve_dataset(args.dataset)

    corpus_name = Path("data") / f"msmarco-index_{args.embedding_name}"
    flex_index = FlexIndex(str(corpus_name))
    graph = load_corpus_graph(flex_index, corpus_name, args.K_cg)

    dataset_retr = pt.get_dataset("irds:msmarco-passage")
    get_text = pt.text.get_text(dataset_retr, "text")
    scorer = TctColBert(TCT_MODELS[args.embedding_name])

    bm25 = get_bm25() % args.k
    # GAR alternates initial-pool and graph-neighbour scoring with the TCT scorer.
    gar = GAR(scorer=get_text >> scorer, corpus_graph=graph,
              num_results=args.budget, batch_size=args.batch, verbose=True)
    pipe = bm25 >> gar

    print(f"[run] GAR on {args.dataset}: {len(topics)} topics")
    run = pipe.transform(topics)
    evaluate(run, qrels, "GAR", args.dataset, args.out_dir)


if __name__ == "__main__":
    main()
