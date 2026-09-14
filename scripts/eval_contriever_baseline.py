"""Quick Contriever-only vs GNN+Contriever evaluation on DL19.

Uses the tctcolbert corpus graph (same as all other GNRR experiments).
Contriever-only: BM25 -> get_text -> Contriever re-scorer (dot product).
GNN+Contriever:  BM25 -> GNRR pipeline with contriever features + tct graph.

Usage:
    python scripts/eval_contriever_baseline.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init(no_download=True)

import ir_measures
from ir_measures import nDCG, AP, RR, R
from pyterrier_dr import FlexIndex

from src.contriever_encoder import ContrieverEncoder


# ── Data ───────────────────────────────────────────────────────────────────
DATASET  = "irds:msmarco-passage/trec-dl-2019/judged"
MODELS   = {
    "GCN-L2":     "models/msmarco_data/hadamard_2_789_0.01_0.0_128_0.6_gcn_local_1_contriever.pt",
    "GCN-L3":     "models/msmarco_data/hadamard_3_789_0.01_0.0_128_0.6_gcn_local_1_contriever.pt",
    "GAT-L2":     "models/msmarco_data/hadamard_2_789_0.01_0.0_128_0.1_gat_local_1_contriever_1.pt",
    "GAT-L3":     "models/msmarco_data/hadamard_3_789_0.01_0.0_128_0.1_gat_local_1_contriever_1.pt",
    "EdgeGAT-L2": "models/msmarco_data/hadamard_2_789_0.01_0.0_128_0.1_edgegat_local_1_contriever.pt",
}


# ── Contriever re-scorer (no GNN, pure dot-product re-ranking) ─────────────
class ContrieverReScorer(pt.Transformer):
    def __init__(self, model_name="facebook/contriever", device="cuda"):
        self.enc = ContrieverEncoder(model_name, device=device)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for qid, grp in df.groupby("qid"):
            q_vec = self.enc.encode_queries([grp["query"].iloc[0]])[0]
            d_vecs = self.enc.encode_docs(grp["text"].tolist())
            scores = d_vecs @ q_vec
            grp = grp.copy()
            grp["score"] = scores.astype(float)
            results.append(grp)
        out = pd.concat(results).sort_values(["qid", "score"], ascending=[True, False])
        out["rank"] = out.groupby("qid").cumcount()
        return out.reset_index(drop=True)


def make_bm25_pipe():
    bm25 = pt.BatchRetrieve.from_dataset("msmarco_passage", "terrier_stemmed", wmodel="BM25", num_results=1000)
    add_text = pt.text.get_text(pt.get_dataset("irds:msmarco-passage"), "text")
    return bm25 >> add_text


def make_gnrr_pipe(conv_type, model_path, n_layers, dropout, heads=1):
    from src.pipeline import EvalConfig, GNRR_Scorer, GNRR
    from src.datamodule import get_corpus_graph

    corpus_name = Path("data/msmarco-index_tctcolbert")  # TCT graph
    flex_index  = FlexIndex(str(Path("data/msmarco-index_contriever")))  # contriever features

    config = EvalConfig(
        conv_type=conv_type,
        modality="local",
        embedding_name="contriever",
        hidden_dim=128,
        n_layers=n_layers,
        n_layers_mlp=1,
        dropout_prob=dropout,
        heads=heads,
        aggr="hadamard",
        model_path=model_path,
    )

    graph  = get_corpus_graph("semantic", corpus_name, K_cg=8, flex_index=FlexIndex(str(corpus_name)))
    # fast=False: encode docs live with ContrieverEncoder.
    # The FlexIndex was built with a different encoder, so live encoding is the
    # only path that matches what the model was trained on.
    # Encoder on CPU to avoid OOM when multiple GNN models share the GPU.
    scorer = GNRR_Scorer(config, fast=False, device="cpu")

    bm25 = pt.BatchRetrieve.from_dataset("msmarco_passage", "terrier_stemmed", wmodel="BM25", num_results=1000)
    return bm25 >> GNRR(scorer, graph, flex_index)


def main():
    ds      = pt.get_dataset(DATASET)
    topics  = ds.get_topics()
    qrels   = ds.get_qrels()
    metrics = [nDCG @ 10, AP(rel=2) @ 1000, RR(rel=2) @ 10]

    bm25_pipe       = make_bm25_pipe()
    contriever_pipe = bm25_pipe >> ContrieverReScorer()

    systems = [
        (bm25_pipe,       "BM25"),
        (contriever_pipe, "Contriever-only"),
    ]

    # GNN variants (skip missing checkpoints)
    gnrr_configs = [
        ("GCN-L2",     "gcn",     2, 0.6, 1),
        ("GCN-L3",     "gcn",     3, 0.6, 1),
        ("GAT-L2",     "gat",     2, 0.1, 1),
        ("GAT-L3",     "gat",     3, 0.1, 1),
        ("EdgeGAT-L2", "edgegat", 2, 0.1, 1),
    ]
    for name, conv, layers, drop, heads in gnrr_configs:
        ckpt = MODELS[name]
        if not Path(ckpt).exists():
            print(f"[skip] {name} — checkpoint not found")
            continue
        try:
            pipe = make_gnrr_pipe(conv, ckpt, layers, drop, heads)
            systems.append((pipe, f"GNRR-{name}+Contriever"))
        except Exception as e:
            print(f"[error] {name}: {e}")

    names = [s[1] for s in systems]
    pipes = [s[0] for s in systems]

    print(f"\n=== pt.Experiment on DL19 — {len(systems)} systems ===\n")
    results = pt.Experiment(
        pipes, topics, qrels,
        eval_metrics=metrics,
        names=names,
        baseline=1,  # compare against Contriever-only
        correction="bonferroni",
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
