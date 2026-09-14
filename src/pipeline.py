"""
Reusable PyTerrier inference pipeline for GNRR and its variants.

This is a cleaned-up extraction of the `GNRR` / `GNRR_Scorer` classes that used to
live inside `build_datasets.ipynb` (the notebook that produced the paper's Table 1).
It is the faithful evaluation path: it retrieves BM25 candidates, induces the
query subgraph from the offline corpus graph, builds query-document features with
frozen TCT-ColBERT embeddings, runs a trained scorer, and returns a re-ranked run.

Used by:
  - scripts/evaluate_testset.py   (reproduce Table 1, evaluate any trained model)
  - scripts/run_graph_baselines.py (non-neural graph baselines reuse the subgraph builder)

Reuses helpers from src/utils.py (no duplication):
  generate_corpus_subgraph_induced_by_query, build_adjacency_matrix,
  adjacency_matrix_to_coo, compute_output.
"""
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
import torch

import pyterrier as pt
from pyterrier_dr import TctColBert

from src.GNN import GNN_NR, GNN_LG, MLP
from src.utils import (
    generate_corpus_subgraph_induced_by_query,
    build_adjacency_matrix,
    adjacency_matrix_to_coo,
    induced_weighted_edges,
    compute_output,
)


# TCT-ColBERT checkpoints by short embedding name.
TCT_MODELS = {
    "tctcolbert": "castorini/tct_colbert-msmarco",
    "tctcolbert2": "castorini/tct_colbert-v2-hnp-msmarco",
}

CONTRIEVER_MODELS = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
}


@dataclass
class EvalConfig:
    """Everything GNN_NR / GNN_LG / MLP read off `config`.

    Defaults are provided for every field so that partial configs (e.g. the entries
    in models/msmarco_data/config_models.json) work without surprises.
    """
    conv_type: str = "gcn"
    modality: str = "multistage"          # 'single' | 'local' | 'multistage' | 'global'
    aggr: str = "hadamard"                # f1: 'hadamard' | 'sum' | 'concat'
    hidden_dim: int = 128
    n_layers: int = 2
    n_layers_mlp: int = 1
    dropout_prob: float = 0.6
    score: bool = True                    # append the extra (BM25/score) feature
    heads: int = 1                        # GAT
    aggr_sage: str = "max"                # GraphSAGE
    negatives: int = 1                    # SignedConv
    K_multistage: int = 200               # multistage subgraph size
    norm: bool = False                    # edgegat: BatchNorm (True) vs LayerNorm (False)
    disable_gnn: bool = False             # no-GNN control (local modality)
    n_steps: int = 6                      # transport: Euler integration steps
    # global-modality extras (kept for completeness)
    pooling: str = "hierarchical"
    pooling_ratio: float = 0.5
    structure_learning: bool = False
    lamb: float = 0.5
    embedding_name: str = "tctcolbert"
    model_path: str = ""

    @staticmethod
    def _as_bool(v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("true", "t", "1", "yes", "y")

    def __post_init__(self):
        # config_models.json stores score as the string "True"
        self.score = self._as_bool(self.score)
        self.disable_gnn = self._as_bool(self.disable_gnn)
        self.structure_learning = self._as_bool(self.structure_learning)

    @classmethod
    def from_dict(cls, d):
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


def build_model(config: EvalConfig, n_feats: int, device: str):
    """Instantiate the scorer described by `config` and load its checkpoint."""
    input_features = n_feats if config.aggr != "concat" else 2 * n_feats

    if config.conv_type == "transformer":
        from src.attention import AttnReranker
        model = AttnReranker(input_features, config, device=device,
                             multistage=(config.modality == "multistage"))
    elif config.conv_type == "edgegat":
        from src.edge_gat_reranker import EdgeGATReranker
        model = EdgeGATReranker(n_feats, config, device=device)
    elif config.conv_type == "learned_edgegat":
        from src.learned_graph import LearnedEdgeGATReranker
        model = LearnedEdgeGATReranker(n_feats, config, device=device)
    elif config.conv_type == "transport":
        from src.physics_gnn import PhysicsTransportGNN
        model = PhysicsTransportGNN(
            input_dim=n_feats,
            n_steps=getattr(config, 'n_steps', 6),
            dropout=config.dropout_prob,
        ).to(device)
    elif config.modality in ("local", "multistage", "global"):
        model = GNN_LG(input_features, config, modality=config.modality,
                       conv_type=config.conv_type, device=device)
    elif config.modality == "single":
        if config.conv_type != "mlp":
            model = GNN_NR(input_features, config, output_dim=1, device=device)
        else:
            model = MLP(input_features, config.hidden_dim, output_dim=1,
                        n_layers=config.n_layers, device=device,
                        dropout_prob=config.dropout_prob)
    else:
        raise ValueError(f"Unknown modality: {config.modality}")

    if config.model_path:
        ckpt = torch.load(config.model_path, map_location=device)
        # checkpoints may be raw state_dicts or lightning-wrapped ("model." prefix)
        if any(k.startswith("model.") for k in ckpt):
            ckpt = {k[len("model."):]: v for k, v in ckpt.items() if k.startswith("model.")}
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"[build_model] missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[build_model] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model.eval()
    return model


class GNRR_Scorer(pt.Transformer):
    """Scores one query's candidate set with a trained GNRR model.

    Required input columns: ['qid', 'query', 'docno', 'text'].
    Returns: ['qid', 'query', 'docno', 'score', 'rank'].
    """

    def __init__(self, config: EvalConfig, text_field="text", fast=True,
                 device=None, n_feats=768, edge_mode="original", restrict_k=0):
        self.config = config
        self.text_field = text_field
        self.fast = fast
        self.n_feats = n_feats
        # restrict_k > 0 => multi-stage cascade: the model re-ranks only the TCT top-k
        # candidates; the remaining candidates are backfilled below them by TCT score.
        self.restrict_k = restrict_k
        # edge_mode controls the graph fed to the model (for ablation controls):
        #   'original' | 'empty' | 'random' (degree-matched random) | 'shuffle'
        self.edge_mode = edge_mode
        self.density_stats = []  # (n_nodes, n_edges) per query, for density reporting
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if config.embedding_name in CONTRIEVER_MODELS:
            from src.contriever_encoder import ContrieverEncoder
            self.encoder = ContrieverEncoder(CONTRIEVER_MODELS[config.embedding_name], device=self.device)
        else:
            self.encoder = TctColBert(TCT_MODELS[config.embedding_name], device=self.device)
        self.model = build_model(config, n_feats, self.device)

    @staticmethod
    def _perturb_edges(edge_index, n, mode, device):
        """Return an edge_index per the requested control mode."""
        if mode == "original":
            return edge_index
        if mode == "empty":
            return torch.empty((2, 0), dtype=torch.long, device=device)
        m = edge_index.shape[1]
        if m == 0 or n <= 1:
            return edge_index
        if mode in ("random", "shuffle"):
            # degree-matched random graph: same edge count, random endpoints
            src = torch.randint(0, n, (m,), device=device)
            dst = torch.randint(0, n, (m,), device=device)
            ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
            return ei
        raise ValueError(mode)

    def __str__(self):
        return f"GNRR_Scorer({self.config.conv_type}/{self.config.modality})"

    def _doc_embeddings(self, cand_df, query_dim, payload):
        """Doc embeddings aligned with cand_df row order (payload when fast)."""
        if self.fast and payload is not None:
            encs = np.zeros((len(cand_df), query_dim), dtype=np.float32)
            for i, docno in enumerate(cand_df["docno"].tolist()):
                try:
                    encs[i] = payload[1][payload[0][docno]]
                except (IndexError, KeyError, LookupError):
                    pass  # zero embedding for docs absent from index
            return encs
        return self.encoder.encode_docs(cand_df[self.text_field])

    def _score_set(self, cand_df, cand_encs, query_enc, corpus_graph):
        """Run the model over one candidate set; returns (index_to_docno, scores[np])."""
        docno_to_index = {d: i for i, d in enumerate(cand_df["docno"].tolist())}
        index_to_docno = {i: d for d, i in docno_to_index.items()}

        edge_weight = None
        if self.config.conv_type in ("edgegat", "learned_edgegat"):
            edge_index, edge_weight = induced_weighted_edges(cand_df, corpus_graph, docno_to_index)
            edge_index = edge_index.to(self.device)
            edge_weight = edge_weight.to(self.device)
        else:
            subgraph = generate_corpus_subgraph_induced_by_query(
                topk_documents_df=cand_df, complete_corpus_graph=corpus_graph)
            adj = build_adjacency_matrix(subgraph, docno_to_index)
            edge_index = adjacency_matrix_to_coo(adj).to(self.device)
        self.density_stats.append((len(docno_to_index), int(edge_index.shape[1])))
        edge_index = self._perturb_edges(edge_index, len(docno_to_index),
                                         self.edge_mode, self.device)

        query_feat = torch.from_numpy(query_enc).clone().unsqueeze(0).unsqueeze(0).to(self.device)
        x = torch.from_numpy(cand_encs).clone().unsqueeze(0).to(self.device)
        ew = edge_weight.unsqueeze(0) if edge_weight is not None else None
        out = compute_output(x, edge_index.unsqueeze(0), query_feat,
                             self.model, self.config.aggr, self.config.conv_type,
                             edge_weight=ew)
        return index_to_docno, np.atleast_1d(out.detach().cpu().squeeze().numpy())

    @torch.no_grad()
    def transform(self, run, corpus_graph, corpus_graph_payload=None):
        topk = run.drop_duplicates(subset="docno").reset_index(drop=True)
        if len(topk) <= 1:
            return run.assign(score=run.get("score", 0.0))

        qid = str(run["qid"].iloc[0])
        query = topk["query"].iloc[0]
        query_enc = self.encoder.encode_queries(topk["query"].iloc[0:1])[0]
        doc_encs = self._doc_embeddings(topk, query_enc.shape[0], corpus_graph_payload)

        if self.restrict_k and len(topk) > self.restrict_k:
            # multi-stage cascade: model re-ranks the TCT top-k, rest backfilled by TCT
            tct = doc_encs @ query_enc
            order = np.argsort(-tct)
            keep, rest = order[:self.restrict_k], order[self.restrict_k:]
            cand_df = topk.iloc[keep].reset_index(drop=True)
            i2d, model_scores = self._score_set(cand_df, doc_encs[keep], query_enc, corpus_graph)
            rest_max = float(tct[rest].max()) if len(rest) else 0.0
            kept = (model_scores - model_scores.min()) + rest_max + 1.0   # above the rest
            docnos = [i2d[i] for i in range(len(i2d))] + topk["docno"].iloc[rest].tolist()
            scores = list(kept) + [float(s) for s in tct[rest]]
        else:
            i2d, out = self._score_set(topk, doc_encs, query_enc, corpus_graph)
            docnos = [i2d[i] for i in range(len(i2d))]
            scores = [float(s) for s in out]

        df = pd.DataFrame({"qid": [qid] * len(docnos), "query": [query] * len(docnos),
                           "docno": docnos, "score": scores})
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(len(df))
        return df


class GNRR(pt.Transformer):
    """Applies a GNRR_Scorer query-by-query over a BM25 run.

    Input columns: ['qid', 'query', 'docno', 'score', 'rank'].
    """

    def __init__(self, scorer: pt.Transformer, corpus_graph, flex_index, text_field="text"):
        self.scorer = scorer
        self.corpus_graph = corpus_graph
        self.flex_index = flex_index
        self.text_field = text_field
        self.dataset_retr = pt.get_dataset("irds:msmarco-passage")
        self.add_text = pt.text.get_text(self.dataset_retr, "text")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        payload = self.flex_index.payload() if self.scorer.fast else None
        ordered_qids = list(df.qid.unique())
        grouped = dict(iter(df.groupby(by=["qid"])))

        results = []
        for i, qid in enumerate(ordered_qids):
            print(f"[GNRR] query {i + 1}/{len(ordered_qids)} (qid={qid})", flush=True)
            batch = grouped[qid].loc[:, ["qid", "query", "docno", "score"]]
            batch = self.add_text(batch)
            results.append(self.scorer.transform(batch, self.corpus_graph,
                                                 corpus_graph_payload=payload))
        out = pd.concat(results, axis=0, ignore_index=True)
        out["rank"] = out["rank"].astype(int)
        return out
