"""Learned sparse graph structure via Binary Concrete (Bernoulli Gumbel-Softmax) gates.

The core idea: instead of using the fixed kNN corpus graph, learn *per-query* which
edges to keep. A small MLP scores each candidate edge from its two endpoints' features
(query-conditioned) plus the prior similarity weight. A Binary Concrete relaxation
makes the discrete keep/drop decision differentiable at training time.

At eval time the gate is hard (threshold at 0): truly sparse — dead edges are removed
from edge_index entirely, so the GNN only touches meaningful connections.

Architecture:
  LearnedEdgeGATReranker = GumbelEdgeSelector + EdgeGAT backbone + DAE head
  - node features: z_q ⊙ z_d (query-document hadamard, same as EdgeGATReranker)
  - selector runs on encoded [hidden]-dim features before message passing
  - self-loops added *after* selection (always kept)
  - FiLM per-layer query conditioning (same as EdgeGATReranker)
  - DAE head: mask mask_ratio dims of z_q⊙z_d, reconstruct from GNN output
    → provides dense gradient to ALL edges (SLAPS supervision starvation fix)

Training losses:
  loss = lambdarank + feat_recon_reg * model._recon_loss
                    + sparsity_reg   * model.gumbel_selector.expected_edges()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops

from src.edge_gat import EdgeGATConv


class GumbelEdgeSelector(nn.Module):
    """Binary Concrete per-edge gate.

    Logit = MLP([z_i, z_j, w_ij]) where z_i, z_j are hidden-dim node features
    and w_ij is the prior corpus-graph similarity weight.

    Train: gate = σ((logit + logistic_noise) / T) ∈ (0,1) — differentiable.
    Eval:  gate = 1 iff logit > 0 — hard, truly sparse.
    """

    def __init__(self, node_dim: int, hidden_dim: int = 64, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.scorer = nn.Sequential(
            nn.Linear(node_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # warm start: prefer keeping edges (positive bias)
        nn.init.constant_(self.scorer[-1].bias, 1.0)
        self._last_expected = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor):
        """
        x:           [N, D]  encoded query-conditioned features
        edge_index:  [2, E]  candidate edges (no self-loops)
        edge_weight: [E]     prior similarity weights
        Returns:
            edge_index_out, weight_out, gates [E] (soft at train, hard at eval)
        """
        src, dst = edge_index
        pair = torch.cat([x[src], x[dst], edge_weight.unsqueeze(-1)], dim=-1)
        logit = self.scorer(pair).squeeze(-1)                       # [E]

        if self.training:
            u = torch.empty_like(logit).uniform_().clamp(1e-6, 1 - 1e-6)
            noise = torch.log(u) - torch.log1p(-u)                  # Logistic(0,1)
            gate = torch.sigmoid((logit + noise) / self.temperature)
            self._last_expected = torch.sigmoid(logit)              # E[gate] — gradient kept
            return edge_index, edge_weight * gate, gate
        else:
            gate_hard = (logit > 0).float()
            keep = gate_hard.bool()
            self._last_expected = gate_hard
            return edge_index[:, keep], edge_weight[keep], gate_hard

    def expected_edges(self) -> torch.Tensor:
        """Mean gate probability — use as sparsity regulariser (minimise λ * this)."""
        if self._last_expected is None:
            return torch.tensor(0.0)
        return self._last_expected.mean()


class LearnedEdgeGATReranker(nn.Module):
    """EdgeGAT re-ranker with a query-conditioned learned sparse graph + DAE auxiliary loss.

    Extends EdgeGATReranker with:
      1. GumbelEdgeSelector — per-edge Binary Concrete gate before message passing.
      2. Denoising autoencoder head (SLAPS) — masks mask_ratio fraction of input
         feature dims, reconstructs them from the GNN output using the learned graph.
         This gives dense gradient to ALL edges, fixing supervision starvation:
         even edges between unlabeled (irrelevant) documents get training signal.

    Training signals:
      - Ranking loss (lambdarank): sparse — only from edges near relevant docs.
      - Reconstruction loss (MSE): dense — from all edges (SLAPS-style).
      - Sparsity regulariser: optional λ_s * E[gate] to encourage pruning.

    At inference: feat_decoder is unused; gate is hard (logit > 0) → truly sparse.
    """

    def __init__(self, input_features: int, config, device: str = "cpu"):
        super().__init__()
        hidden = config.hidden_dim
        n_layers = config.n_layers
        heads = max(1, getattr(config, "heads", 1))
        dropout = getattr(config, "dropout_prob", 0.0)
        use_bn = bool(getattr(config, "norm", False))
        gumbel_temp = float(getattr(config, "gumbel_temp", 1.0))
        mask_ratio = float(getattr(config, "mask_ratio", 0.15))

        self.mask_ratio = mask_ratio
        self._recon_loss = None          # set by forward(), consumed by training_step

        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.W_edge = nn.Linear(1, hidden)

        self.layers = nn.ModuleList([
            EdgeGATConv(hidden, hidden, heads=heads, concat=False,
                        edge_dim=hidden, dropout=dropout, add_self_loops=False)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            (nn.BatchNorm1d(hidden) if use_bn else nn.LayerNorm(hidden))
            for _ in range(n_layers)
        ])
        self.acts = nn.ModuleList([nn.GELU() for _ in range(n_layers)])
        self.drops = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])

        # FiLM: per-layer query conditioning.  input_features = raw query dim (768).
        self.film = nn.ModuleList([nn.Linear(input_features, 2 * hidden)
                                   for _ in range(n_layers)])

        self.gumbel_selector = GumbelEdgeSelector(
            node_dim=hidden,
            hidden_dim=max(32, hidden // 2),
            temperature=gumbel_temp,
        )
        self.decoder = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

        # DAE head: reconstruct input features from GNN-aggregated hidden states.
        # Active only during training when mask_ratio > 0.
        if mask_ratio > 0:
            self.feat_decoder = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, input_features),
            )
        else:
            self.feat_decoder = None

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        for conv in self.layers:
            conv.reset_parameters()
        for f in self.film:
            nn.init.zeros_(f.weight)
            nn.init.zeros_(f.bias)
        nn.init.constant_(self.gumbel_selector.scorer[-1].bias, 1.0)

    def forward(self, x, edge_index, query, edge_weight=None):
        """
        x:           [N, F]  z_q ⊙ z_d per node
        edge_index:  [2, E]  corpus-graph edges (no self-loops)
        query:       [F]     raw query embedding (for FiLM)
        edge_weight: [E]     corpus-graph prior weights

        Side-effect (training only):
          self._recon_loss  — scalar MSE over masked dims; add feat_recon_reg * this to loss.
        """
        n = x.shape[0]
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=x.device)

        self._recon_loss = None

        # --- SLAPS-style feature masking (training only) ---
        if self.training and self.feat_decoder is not None and self.mask_ratio > 0:
            mask = torch.rand(n, x.shape[1], device=x.device) < self.mask_ratio  # [N, F]
            x_in = x.clone()
            x_in[mask] = 0.0
        else:
            x_in = x
            mask = None

        h = self.encoder(x_in)                                      # [N, hidden]

        # Structure learning: prune corpus-graph edges before message passing
        if edge_index.shape[1] > 0:
            edge_index, edge_weight, _ = self.gumbel_selector(h, edge_index, edge_weight)

        # Self-loops always kept (weight 1.0), added after pruning
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=n)
        edge_attr = self.W_edge(edge_weight.float().view(-1, 1))    # [E', hidden]

        q = query.view(1, -1)
        for conv, norm, act, drop, film in zip(
                self.layers, self.norms, self.acts, self.drops, self.film):
            y = norm(conv(h, edge_index, edge_attr=edge_attr))
            gamma, beta = film(q).chunk(2, dim=-1)
            y = (1.0 + gamma) * y + beta
            y = drop(act(y))
            h = h + y

        # --- Reconstruction loss over masked positions ---
        # The graph structure (learned via gumbel_selector) mediates reconstruction,
        # so this loss propagates gradients to selector for every edge, not just
        # those adjacent to relevant documents.
        if mask is not None and mask.any():
            x_recon = self.feat_decoder(h)                          # [N, F]
            self._recon_loss = F.mse_loss(x_recon[mask], x[mask])  # scalar

        return self.decoder(h)                                      # [N, 1]
