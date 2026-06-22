"""
Edge-GAT re-ranker: a query-aware message-passing GNN over the query-induced subgraph.

Design (after the "graph must add what attention can't" revision):
  - node features: z_q ⊙ z_d (hadamard) -> each node carries its own TCT relevance
    (sum = z_q·z_d = TCT score), the same strong starting point the self-attention
    re-ranker uses.
  - EDGE features = the corpus-graph EDGE WEIGHTS (cosine sim for the semantic graph,
    BM25 sim for the lexical graph). This is the signal a fully-connected self-attention
    re-ranker cannot see, and it is what makes semantic vs lexical graphs differ.
  - QUERY conditioning via FiLM (per-layer, per-dimension gamma/beta from the query),
    so the query modulates every node directly -- no single "hub" node bottleneck
    (which behaved like a global mean-pool and washed out top-rank discrimination).
  - normalization + GELU + dropout + residual between layers.
  - decoder: hidden -> 1 scalar relevance score per node. NO pooling.

A self-attention re-ranker is GAT on a fully-connected graph with no edge features;
this model instead uses the sparse corpus graph WITH informative (weighted) edges plus
FiLM query conditioning.

Drop-in: forward(x, edge_index, query, edge_weight) is invoked from
src/utils.py:compute_output for conv_type == 'edgegat'.
"""
import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops

from src.edge_gat import EdgeGATConv


class EdgeGATReranker(nn.Module):
    def __init__(self, input_features, config, device="cpu"):
        super().__init__()
        self.config = config
        self.dev = device
        hidden = config.hidden_dim
        n_layers = config.n_layers
        heads = max(1, getattr(config, "heads", 1))
        dropout = getattr(config, "dropout_prob", 0.0)
        use_bn = bool(getattr(config, "norm", False))

        # encoder: query-document interaction features -> hidden
        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        # scalar edge weight -> edge-feature vector (edge_dim the conv expects)
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

        # FiLM: per layer, query -> (gamma, beta). Zero-init => starts as identity
        # (gamma=0 -> scale (1+gamma)=1, beta=0), then learns query dependence.
        self.film = nn.ModuleList([nn.Linear(input_features, 2 * hidden)
                                   for _ in range(n_layers)])

        self.decoder = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        for conv in self.layers:
            conv.reset_parameters()
        for f in self.film:               # identity-init FiLM
            nn.init.zeros_(f.weight)
            nn.init.zeros_(f.bias)

    def forward(self, x, edge_index, query, edge_weight=None):
        """x: [N, F] interaction features; edge_index: [2, E]; query: [F];
        edge_weight: [E] corpus-graph weights (None -> all ones)."""
        n = x.shape[0]
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=x.device)
        # self-loops carry weight 1.0 (a node is maximally similar to itself)
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=n)
        edge_attr = self.W_edge(edge_weight.float().view(-1, 1))   # [E, hidden]

        q = query.view(1, -1)
        h = self.encoder(x)
        for conv, norm, act, drop, film in zip(
                self.layers, self.norms, self.acts, self.drops, self.film):
            y = norm(conv(h, edge_index, edge_attr=edge_attr))
            gamma, beta = film(q).chunk(2, dim=-1)                 # [1, hidden] each
            y = (1.0 + gamma) * y + beta                           # FiLM query conditioning
            y = drop(act(y))
            h = h + y                                              # residual
        return self.decoder(h)                                     # [N, 1]
