"""
Self-attention re-ranker baselines (apples-to-apples vs GNRR).

`AttnReranker` is a TransformerEncoder over the candidate set's query-document
feature matrix X'_q -- the *same* features GNRR feeds its GNN (f1(z_q, z_d) plus the
optional score column). It ignores the graph edges entirely, so comparing it to GNRR
isolates "self-attention over the candidate set" vs "GNN over the corpus-graph
subgraph". Self-attention is O(K^2) in the number of candidates, matching the
complexity claim in the paper.

Two regimes (selected via `multistage`):
  - 1-stage   : attention over ALL BM25 top-K candidates; output is the score.
  - multistage: BM25 top-K -> TCT top-`K_multistage` -> attention over that subset;
                a learned delta is added to the TCT score (mirrors GNN_LG multistage
                at src/GNN.py:295 so the comparison is controlled).

Drop-in: forward(x, edge_index) matches GNN_NR/GNN_LG, so it works unchanged with
src/utils.py:compute_output and src/pipeline.py:GNRR_Scorer (conv_type='transformer').
"""
import torch
import torch.nn as nn


def cascade_place(base_scores, top_indices, head_scores):
    """Multi-stage cascade scoring WITHOUT a TCT residual.

    The top-K candidates are ranked purely by the stage-2 `head_scores` and placed
    strictly above the remaining candidates (which keep their stage-1 / TCT order).
    This avoids the failure mode where a large-magnitude TCT score swamps a small
    learned delta, leaving the ranking unchanged. Gradients flow through head_scores.
    """
    n = base_scores.shape[0]
    mask = torch.ones(n, dtype=torch.bool, device=base_scores.device)
    mask[top_indices] = False
    rest_max = base_scores[mask].max() if mask.any() else base_scores.min()
    out = base_scores.clone()
    # shift head scores so the lowest re-ranked doc still sits above all the rest
    out[top_indices] = (head_scores - head_scores.min()) + rest_max + 1.0
    return out


class AttnReranker(nn.Module):
    def __init__(self, input_features, config, device="cpu", multistage=False):
        super().__init__()
        self.config = config
        self.dev = device
        self.multistage = multistage
        self.K_multistage = getattr(config, "K_multistage", 200)

        if getattr(config, "score", False):
            input_features += 1
        self.input_features = input_features

        hidden = config.hidden_dim
        heads = max(1, getattr(config, "heads", 4))
        # nhead must divide d_model; fall back to 1 head if it doesn't.
        if hidden % heads != 0:
            heads = 1
        dropout = config.dropout_prob

        self.input_proj = nn.Linear(input_features, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 2,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.dec = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

        self.to(device)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def _encode(self, feats):
        """feats: [n, F] -> per-node scalar [n] via self-attention over the set."""
        h = self.input_proj(feats).unsqueeze(0)          # [1, n, hidden]
        h = self.encoder(h).squeeze(0)                    # [n, hidden]
        return self.dec(h).squeeze(-1)                    # [n]

    def forward(self, x, edge_index=None):
        # x: [N, F]. edge_index is accepted for interface compatibility and ignored.
        x_individual = x.clone()
        scores = torch.sum(x_individual, dim=-1)          # TCT-style dot score proxy

        if getattr(self.config, "score", False):
            feats = torch.cat((x_individual, scores.unsqueeze(-1)), dim=-1)
        else:
            feats = x_individual

        if not self.multistage:
            # 1-stage: attend over all candidates, output the score directly.
            return self._encode(feats).unsqueeze(-1)

        # multistage: re-rank the TCT top-K purely by the attention head (no TCT
        # residual), placing them above the remaining candidates (cascade).
        k = min(self.K_multistage, x_individual.shape[0])
        top_indices = torch.topk(scores, k=k).indices
        head = self._encode(feats[top_indices])           # [k]
        return cascade_place(scores, top_indices, head).unsqueeze(-1)
