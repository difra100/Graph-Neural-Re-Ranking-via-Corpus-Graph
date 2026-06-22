# mypy: ignore-errors
import torch
import torch_geometric
from typing import Optional
from torch import nn
from torch.nn import functional as F  # noqa:N812
from torch.nn import Module, Parameter, init, Linear, ModuleList, Sequential, LeakyReLU
from torch_geometric.nn.conv import MessagePassing, GATConv
from torch_geometric.utils import degree, softmax as pyg_softmax
from gfmrag.gnn.edge_gat import *

import math




def dirichlet_energy_normalized(x, edge_index):
    row, col = edge_index
    num_nodes = x.size(0)
    deg = degree(row, num_nodes=num_nodes).clamp(min=1)   # avoid div/0
    x_norm = x / deg.sqrt().unsqueeze(-1)                 # D^{-1/2} x
    diff = x_norm[row] - x_norm[col]
    return (diff ** 2).sum(dim=-1).mean().item()

class NaiveAggr(MessagePassing):
    r"""
    Simple graph convolution which compute a transformation of
    neighboring nodes:  sum_{j \in N(u)} Vx_j
    """

    def __init__(self, 
                 in_channels, 
                 edge_channels: int = 0):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.edge_channels = edge_channels
        self.lin = Linear(in_channels, in_channels, bias=False)
        self.edge_lin = None
        if edge_channels > 0:
            self.edge_lin = Linear(edge_channels, in_channels)
        self.reset_parameters()

    def forward(self, x, edge_index=None, edge_attr=None):
        out = self.propagate(
            x=self.lin(x), edge_index=edge_index, edge_attr=edge_attr
        )
        return out

    def message(
        self, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if edge_attr is None:
            return x_j
        elif self.edge_lin is None:
            if len(edge_attr.shape) == 1:
                return edge_attr.view(-1, 1) * x_j
            else:
                raise ValueError()
        else:
            return x_j + self.edge_lin(edge_attr)
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.edge_lin is not None: self.edge_lin.reset_parameters()

    def __repr__(self) -> str:
        return f"self.__class__.__name__(in_channels: {self.in_channels}, edge_channels: {self.edge_channels})"


conv_names = ["NaiveAggr", "GCNConv"]


class KGGCNConv(MessagePassing):

    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='add')
        self.W = nn.Linear(in_features=in_dim, out_features=out_dim, bias=False)
        self.O = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self.edge_W = nn.Linear(in_features=in_dim, out_features=out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.W.reset_parameters()
        self.O.reset_parameters()
        self.edge_W.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None, query_emb=None):
        out = self.W(x)
        row, col = edge_index
        valid_mask = (row >= 0) & (col >= 0)
        row = row[valid_mask]
        col = col[valid_mask]
        edge_index = edge_index[:, valid_mask]
        edge_attr = edge_attr[valid_mask] if edge_attr is not None else None
        edge_weight = edge_weight[valid_mask] if edge_weight is not None else None
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        if edge_attr is not None:
            edge_attr = self.edge_W(edge_attr)
        out = self.propagate(edge_index, x=out, norm=norm, edge_attr=edge_attr,
                             edge_weight=edge_weight, query_emb=query_emb)
        out = out + self.O(x)
        return out

    def message(self, x_j, norm, edge_attr, query_emb, edge_weight):
        if edge_attr is None:
            msg = x_j if query_emb is None else x_j + query_emb
        else:
            msg = x_j + edge_attr if query_emb is None else x_j + edge_attr + query_emb
        return norm.view(-1, 1) * msg


class EntityGNN(nn.Module):

    def __init__(self, dims, norm: bool | None = False, activation: bool = True, dropout: float = 0.0):
        super().__init__()
        self.dims = dims
        layers, norms, acts, drops = [], [], [], []
        for i in range(len(dims) - 1):
            in_c, out_c = dims[i], dims[i + 1]
            layers.append(KGGCNConv(in_c, out_c))
            norms.append(nn.LayerNorm(out_c) if norm else nn.Identity())
            acts.append(nn.GELU())
            drops.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        self.acts = nn.ModuleList(acts)
        self.drops = nn.ModuleList(drops)

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None, query_emb=None):
        for conv, norm, act, drop in zip(self.layers, self.norms, self.acts, self.drops):
            y = conv(x, edge_index, edge_attr=edge_attr, edge_weight=edge_weight, query_emb=query_emb)
            y = drop(act(norm(y)))
            x = y + x
        return x


class GenericGNN(nn.Module):

    def __init__(self, input_dim: int, dims, norm: bool | None = False, activation: bool = True,
                 dropout: float = 0.0, conv: str = "gcn", pool: str = "mean", num_tokens: int = 1):
        super().__init__()
        self.dims = dims
        self.pool = pool
        self.W_edge = nn.Linear(input_dim, dims[0])
        layers, norms, acts, drops = [], [], [], []
        for i in range(len(dims) - 1):
            in_c, out_c = dims[i], dims[i + 1]
            layers.append(KGGCNConv(in_c, out_c))
            norms.append(nn.BatchNorm1d(out_c) if norm else nn.Identity())
            acts.append(nn.GELU())
            drops.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        self.acts = nn.ModuleList(acts)
        self.drops = nn.ModuleList(drops)

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None,
                batched_query=None, batch=None):
        edge_attr = self.W_edge(edge_attr) if edge_attr is not None else None
        for conv, norm, act, drop in zip(self.layers, self.norms, self.acts, self.drops):
            y = conv(x, edge_index, edge_attr=edge_attr)
            y = drop(act(norm(y)))
            x = y + x
        if self.pool == "mean" and batch is not None:
            x = torch_geometric.nn.global_mean_pool(x, batch)
        elif self.pool == "max" and batch is not None:
            x = torch_geometric.nn.global_max_pool(x, batch)
        elif self.pool == "sum" and batch is not None:
            x = torch_geometric.nn.global_add_pool(x, batch)
        return x


class GNNEmbedder(torch.nn.Module):
    """GNN embedder — thin wrapper kept for backward compatibility."""

    def __init__(self, input_dim: int, dims, norm: bool | None = False, activation: bool = True,
                 dropout: float = 0.0, conv: str = "gcn", pool: str = "mean", num_tokens: int = 1,
                 **kwargs):
        super().__init__()
        self.gnn = GenericGNN(input_dim, dims, norm=norm, activation=activation,
                              dropout=dropout, conv=conv, pool=pool, num_tokens=num_tokens)
        self.dims = dims

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None,
                batched_query=None, batch=None, **kwargs):
        return self.gnn(x, edge_index, edge_attr=edge_attr, edge_weight=edge_weight,
                        batched_query=batched_query, batch=batch)


class DegNormConv(MessagePassing):
    """Degree-normalized graph convolution with additive vector edge features.

    Message: norm_ij * (x_j + edge_attr_ij), where norm_ij = (d_i * d_j)^{-0.5}.
    Edge attributes must already be projected to the same dimension as x.
    """

    def __init__(self):
        super().__init__(aggr="add")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        row, col = edge_index
        num_nodes = x.size(0)
        deg = degree(col, num_nodes=num_nodes, dtype=x.dtype).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

    def message(
        self,
        x_j: torch.Tensor,
        norm: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        msg = x_j if edge_attr is None else (x_j + edge_attr)
        return norm.view(-1, 1) * msg


class GATEdgeConv(MessagePassing):
    """GAT-style conv with additive vector edge features.

    Attention: e_ij = LeakyReLU(a^T [x_i || x_j]), softmax-normalized per target node.
    Message:   alpha_ij * (x_j + edge_attr_ij).
    Heads are averaged (concat=False) so output dim equals input dim.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 1, dropout: float = 0.0):
        super().__init__(aggr="add", node_dim=0)
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        # Learnable attention vector per head: a ∈ R^{n_heads × 2*head_dim}
        self.att = nn.Parameter(torch.empty(1, n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.att.view(n_heads, 2 * self.head_dim))
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr,
                              size=(x.size(0), x.size(0)))

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        index: torch.Tensor,
        size_i: int,
    ) -> torch.Tensor:
        E, H, D = x_i.size(0), self.n_heads, self.head_dim
        xi = x_i.view(E, H, D)
        xj = x_j.view(E, H, D)
        # attention logits and per-node softmax normalization
        alpha = (torch.cat([xi, xj], dim=-1) * self.att).sum(-1)   # [E, H]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = pyg_softmax(alpha, index, num_nodes=size_i)          # [E, H]
        alpha = self.attn_drop(alpha)
        # additive edge feature in message, then weight by attention
        msg = x_j if edge_attr is None else (x_j + edge_attr)        # [E, hidden_dim]
        return (msg.view(E, H, D) * alpha.unsqueeze(-1)).view(E, H * D)


class SMPNNBlock(nn.Module):
    """Single Pre-LN Transformer-style SMPNN block (equations 1-4 from the paper).

    Query conditioning is applied via Adaptive Layer Norm (AdaLN): the query repr
    predicts per-dimension (gamma, beta) shifts applied *after* each LayerNorm but
    *before* the GCN and FF sub-blocks.  Zero-init ensures the block starts as the
    unconditional SMPNN and gradually learns query dependence.

    Equations:
        H1 = (1 + gamma1[b]) * LayerNorm(X) + beta1[b]          # eq 1 + AdaLN
        H2 = alpha1 * SiLU(conv(W1(H1))) + X                     # eq 2
        H3 = (1 + gamma2[b]) * LayerNorm(H2) + beta2[b]          # eq 3 + AdaLN
        X' = alpha2 * SiLU(W2(H3)) + H2                          # eq 4
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
        query_dim: Optional[int] = None,
        conv_type: str = "gcn",
        n_heads: int = 1,
    ):
        super().__init__()
        self.norm_gcn = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # per-dimension scaling initialized at 1e-6 (identity-style block init, DiT)
        self.alpha1 = nn.Parameter(torch.full((hidden_dim,), 1e-6))
        self.alpha2 = nn.Parameter(torch.full((hidden_dim,), 1e-6))
        self.drop = nn.Dropout(dropout)
        if conv_type == "gat":
            self.conv = GATEdgeConv(hidden_dim, n_heads=n_heads, dropout=dropout)
        else:
            self.conv = DegNormConv()

        # AdaLN: query → (gamma, beta) for each of the two LayerNorms
        if query_dim is not None:
            self.adaLN_gcn = nn.Linear(query_dim, 2 * hidden_dim)
            self.adaLN_ff = nn.Linear(query_dim, 2 * hidden_dim)
            nn.init.zeros_(self.adaLN_gcn.weight)
            nn.init.zeros_(self.adaLN_gcn.bias)
            nn.init.zeros_(self.adaLN_ff.weight)
            nn.init.zeros_(self.adaLN_ff.bias)
        else:
            self.adaLN_gcn = None
            self.adaLN_ff = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        query_repr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # --- GCN branch (equations 1-2) ---
        h = self.norm_gcn(x)
        if self.adaLN_gcn is not None and query_repr is not None and batch is not None:
            gamma, beta = self.adaLN_gcn(query_repr).chunk(2, dim=-1)
            h = (1.0 + gamma[batch]) * h + beta[batch]
        h = self.conv(self.W1(h), edge_index, edge_attr=edge_attr)
        x = self.alpha1 * F.silu(h) + x

        # --- FF branch (equations 3-4) ---
        h = self.norm_ff(x)
        if self.adaLN_ff is not None and query_repr is not None and batch is not None:
            gamma, beta = self.adaLN_ff(query_repr).chunk(2, dim=-1)
            h = (1.0 + gamma[batch]) * h + beta[batch]
        x = self.alpha2 * F.silu(self.W2(h)) + x

        return self.drop(x)


class TopKSMPNN(torch.nn.Module):
    """SMPNN-based encoder with NodeSoftTopK pooling.

    Drop-in replacement for TopKGAT.  Uses Pre-LN Transformer-style GCN blocks
    (O(E) complexity, no attention) with query conditioning via AdaLN, and the
    same NodeSoftTopK + score-weighted or multi-token pooling interface.

    Edge attributes are projected to hidden_dim vectors and added to the source
    node features inside the message step: msg_ij = norm_ij * (x_j + W_edge * e_ij).
    """

    def __init__(
        self,
        input_dim: int,
        dims,
        norm: bool = False,
        activation: bool = True,
        dropout: float = 0.0,
        conv: str = "gcn",
        num_tokens: int = 1,
        n_heads: int = 1,
        ratio: float = 1,
        pool: str = "fact",
        k_facts: int = 100,
        use_film: bool = False,
    ):
        super(TopKSMPNN, self).__init__()
        self.dims = dims
        self.num_tokens = num_tokens
        self.pool = pool

        hidden_dim = dims[0]

        # Project edge attributes to hidden_dim vectors (same dim as node features)
        self.W_edge = nn.Linear(input_dim, hidden_dim)

        # Project query into hidden dim
        self.query_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # SMPNN blocks — constant hidden_dim throughout (dims[0])
        # conv="gcn" uses DegNormConv; conv="gat" uses GATEdgeConv with n_heads
        self.blocks = nn.ModuleList([
            SMPNNBlock(hidden_dim, dropout=dropout, query_dim=hidden_dim,
                       conv_type=conv, n_heads=n_heads)
            for _ in range(len(dims) - 1)
        ])

        # NodeSoftTopK — identical interface to TopKGAT
        self.pool_layer = NodeSoftTopK(
            dropout=dropout,
            ratio=-1,
            k=k_facts,
            mlp_dim=hidden_dim * 2,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        batched_query: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        batched_subgraphs_equiv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch_size = batch.max().item() + 1

        # Project edge attributes to hidden_dim vectors for additive message passing
        edge_h = self.W_edge(edge_attr) if edge_attr is not None else None

        # Project query to hidden dim for AdaLN conditioning
        query_repr = self.query_proj(batched_query)  # [B, hidden_dim]

        # SMPNN blocks with AdaLN query conditioning
        for block in self.blocks:
            x = block(x, edge_index, edge_attr=edge_h, query_repr=query_repr, batch=batch)

        # NodeSoftTopK pooling — identical to TopKGAT
        x_filtered, scores, idx = self.pool_layer(x, query_repr, batch)

        if self.num_tokens == 1:
            from torch_geometric.utils import softmax as pyg_softmax
            sel_batch = batch[idx]
            weights = pyg_softmax(scores[idx], sel_batch, num_nodes=batch_size)
            z_pooled = torch_geometric.nn.global_add_pool(
                x_filtered * weights.unsqueeze(-1),
                sel_batch,
                size=batch_size,
            )
        else:
            graph_ids = batch[idx]
            counts = torch.bincount(graph_ids, minlength=batch_size)
            max_k = counts.max().item()
            local_pos = torch.cat([torch.arange(c, device=idx.device) for c in counts])
            z_pooled = torch.zeros(
                batch_size, max_k, x_filtered.size(-1),
                device=x_filtered.device, dtype=x_filtered.dtype,
            ).index_put((graph_ids, local_pos), x_filtered)

        return z_pooled


class AntiSymmetricConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        edge_channels: int = 0,
        num_iters: int = 1,
        gamma: float = 0.1,
        epsilon: float = 0.1,
        activ_fun: str = "tanh",
        bias: bool = True,
    ) -> None:

        super().__init__(aggr="add")
        self.W = Parameter(torch.empty((in_channels, in_channels)))
        self.bias = Parameter(torch.empty(in_channels)) if bias else None

        self.conv = NaiveAggr(in_channels, edge_channels=edge_channels)
        
        self.in_channels = in_channels
        self.edge_channels = edge_channels
        self.num_iters = num_iters
        self.gamma = gamma
        self.epsilon = epsilon
        self.activation = getattr(torch, 'tanh')
        self.activ_fun = activ_fun

        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        self.antisymmetric_W = (
            self.W
            - self.W.T
            - self.gamma * torch.eye(self.in_channels, device=self.W.device)
        )
        for i in range(self.num_iters):
            neigh_x = self.conv(x, edge_index=edge_index, 
                                **({'edge_attr': edge_attr}))
            conv = x @ self.antisymmetric_W.T + neigh_x

            if self.bias is not None:
                conv += self.bias

            x = x + self.epsilon * self.activation(conv)

        return x

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        self.conv.reset_parameters()

class GAT(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        dims,
        norm: bool | None = False,
        activation: bool = True,
        dropout: float = 0.0,
        conv: str = "gat",
        pool: str = "mean",
        num_tokens: int = 1,
        num_heads: int = 1,
        **kwargs
    ):
        super(GAT, self).__init__()
        self.dims = dims
        self.norm_type = norm
        self.activation = activation
        self.pool = pool
        self.num_heads = num_heads
        self.W_edge = nn.Linear(input_dim, dims[0])

        layers: list[nn.Module] = []
        norms: list[nn.Module] = []
        acts: list[nn.Module] = []
        drops: list[nn.Module] = []
        
        for i in range(len(self.dims) - 1):
            in_c, out_c = self.dims[i], self.dims[i + 1]
            layers.append(EdgeGATConv(in_c, out_c, heads=num_heads, concat=False, edge_dim = dims[0]))
            # per-layer norm
            if norm:
                norms.append(nn.BatchNorm1d(out_c))
            else:
                norms.append(nn.Identity())
            # activation
            acts.append(self._make_activation())
            # dropout
            drops.append(nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity())

        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        self.acts = nn.ModuleList(acts)
        self.drops = nn.ModuleList(drops)

    def _make_activation(self) -> nn.Module:
        return nn.GELU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        batched_query: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        edge_attr = self.W_edge(edge_attr) if edge_attr is not None else None
        emb_list = []
        for conv, norm, act, drop, in_dim, out_dim in zip(
            self.layers,
            self.norms,
            self.acts,
            self.drops,
            self.dims[:-1],
            self.dims[1:],
        ):
            y = conv(x, edge_index, edge_attr=edge_attr)
            # Normalization
            y = norm(y)

            # Activation + Dropout
            y = act(y)
            y = drop(y)

            x = y + x
            emb_list.append(x)

        if self.pool == "mean" and batch is not None:
            x = torch_geometric.nn.global_mean_pool(x, batch)
        elif self.pool == "max" and batch is not None:
            x = torch_geometric.nn.global_max_pool(x, batch)
        elif self.pool == "sum" and batch is not None:
            x = torch_geometric.nn.global_add_pool(x, batch)
        elif self.pool == 'lateint':
            
            query_emb = self.q_enc(batched_query)  # (batch_size, n_token, emb_dim)
            
            max_scores = self.pool_layer(query_emb[batch], emb_list)
            
            x = max_scores[:, None] * x
            sum_x = torch_geometric.nn.global_add_pool(x, batch)
            sum_scores = torch_geometric.nn.global_add_pool(max_scores.unsqueeze(-1), batch)
            x = sum_x / (sum_scores + 1e-10)

        return x





class NodeSoftTopK(nn.Module):

    def __init__(self, dropout=0.1, ratio=1, temperature=1.0, k=None, mlp_dim=0):
        super(NodeSoftTopK, self).__init__()
        self.ratio = ratio
        self.k = k
        self.dim_mlp = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.LayerNorm(mlp_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, 1)  # raw scores, unbounded positive and negative
        )

    def forward(self, x, query_enc, batch, y0=None):
        query_per_node = query_enc[batch]
        mlp_input = torch.cat([x, query_per_node], dim=-1)
        scores = self.dim_mlp(mlp_input).squeeze(-1)
        # No normalization — scores are used raw for topk (detached)
        # and raw for STE gradient

        idx = self._topk(scores.detach(), batch)

        hard_mask = torch.zeros(x.size(0), device=x.device)
        hard_mask[idx] = 1.0

        ste_mask = hard_mask - scores.detach() + scores

        x_pooled = (x * ste_mask.unsqueeze(-1))[idx]

        return x_pooled, scores, idx

    def _topk(self, scores, batch):
        num_elements = torch_geometric.nn.global_add_pool(
            batch.new_ones(scores.size(0)), batch
        )
        batch_size = num_elements.size(0)

        if self.k is None:
            k_per_graph = (self.ratio * num_elements.float()).ceil().long()
        else:
            k_per_graph = torch.clamp(
                torch.full((batch_size,), self.k, dtype=torch.long, device=scores.device),
                max=num_elements
            )

        selected = []
        for b in range(batch_size):
            mask = batch == b
            local_idx = torch.where(mask)[0]
            ki = k_per_graph[b].item()
            if ki == 0 or local_idx.size(0) == 0:
                continue
            ki = min(ki, local_idx.size(0))
            _, top_local = scores[mask].topk(ki, largest=True, sorted=False)
            selected.append(local_idx[top_local])

        return torch.cat(selected) if selected else torch.tensor([], dtype=torch.long, device=scores.device)


class TopKGAT(torch.nn.Module):
    """
    GNN with query-as-virtual-node for query-aware message passing.
    
    The query embedding is injected as a virtual node connected to all real nodes,
    so every message passing step is query-conditioned. No separate q_enc needed —
    the query influence propagates through the graph naturally.
    
    Fact selection uses SoftTopK + STE with averaged pooling (no softmax weighting).
    """
    def __init__(
        self,
        input_dim: int,
        dims,
        norm: bool | None = False,
        activation: bool = True,
        dropout: float = 0.0,
        conv: str = "gat",
        num_tokens: int = 1,
        n_heads: int = 1,
        ratio: float = 1,
        pool: str = "fact",
        k_facts: int = 100,
        use_film: bool = False,
    ):
        super(TopKGAT, self).__init__()
        self.dims = dims
        self.num_heads = n_heads
        self.pool = pool
        self.use_film = use_film

        # Project edge attributes to hidden dim
        self.W_edge = nn.Linear(input_dim, dims[0])
        nn.init.xavier_uniform_(self.W_edge.weight)
        nn.init.zeros_(self.W_edge.bias)

        # Project query into node embedding space
        self.query_proj = nn.Sequential(
            nn.Linear(input_dim, dims[0]),
            nn.LayerNorm(dims[0]),
            nn.GELU(),
        )

        # GNN layers
        layers, norms, acts, drops = [], [], [], []
        for i in range(len(dims) - 1):
            in_c, out_c = dims[i], dims[i + 1]
            layers.append(EdgeGATConv(in_c, out_c, heads=n_heads, concat=False, edge_dim=dims[0]))
            norms.append(nn.LayerNorm(out_c) if not norm else nn.BatchNorm1d(out_c))
            acts.append(nn.GELU())
            drops.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())

        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        self.acts = nn.ModuleList(acts)
        self.drops = nn.ModuleList(drops)

        # FiLM: one (gamma, beta) pair per GNN layer.
        # gamma/beta have shape [dims[i+1]] and are produced from the query repr.
        if self.use_film:
            self.film_gamma = nn.ModuleList([
                nn.Linear(dims[0], dims[i + 1]) for i in range(len(dims) - 1)
            ])
            self.film_beta = nn.ModuleList([
                nn.Linear(dims[0], dims[i + 1]) for i in range(len(dims) - 1)
            ])
            # Init gamma to 1 and beta to 0 so FiLM starts as identity
            for lin in self.film_gamma:
                nn.init.zeros_(lin.weight)
                nn.init.ones_(lin.bias)
            for lin in self.film_beta:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

        # SoftTopK for node selection — input is [node_emb, query_emb] = 2*dims[-1]
        self.pool_layer = NodeSoftTopK(
            dropout=dropout,
            ratio=-1,
            k=k_facts,
            mlp_dim=dims[-1] * 2,
        )

        self.num_tokens = num_tokens

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        batched_query: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        batched_subgraphs_equiv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        batch_size = batch.max().item() + 1

        # Project edge attributes
        if edge_attr is not None:
            edge_attr = self.W_edge(edge_attr)
            edge_attr = F.layer_norm(edge_attr, edge_attr.shape[-1:])

        # Initialize virtual node from query — one per graph in batch
        # virtual_node: [batch_size, dims[0]]
        query_repr = self.query_proj(batched_query)

        # Message passing with virtual node injection
        for layer_idx, (conv, norm, act, drop) in enumerate(zip(
            self.layers, self.norms, self.acts, self.drops
        )):

            y = conv(x, edge_index, edge_attr=edge_attr)
            y = norm(y)
            y = act(y)
            y = drop(y)
            x = x + y  # residual on original x

            if self.use_film:
                gamma = self.film_gamma[layer_idx](query_repr)  # [batch_size, out_c]
                beta = self.film_beta[layer_idx](query_repr)    # [batch_size, out_c]
                x = gamma[batch] * x + beta[batch]


        x_filtered, scores, idx = self.pool_layer(
            x, query_repr, batch
        )

        if self.num_tokens == 1:
            # Score-weighted pooling: softmax of query-aware scores over selected nodes,
            # then weighted sum per graph. Preserves the ranking information computed by
            # the NodeSoftTopK MLP instead of discarding it with a flat mean.
            from torch_geometric.utils import softmax as pyg_softmax
            sel_batch = batch[idx]
            weights = pyg_softmax(scores[idx], sel_batch, num_nodes=batch_size)  # [total_sel]
            z_pooled = torch_geometric.nn.global_add_pool(
                x_filtered * weights.unsqueeze(-1),
                sel_batch,
                size=batch_size,
            )
        else:
            # Scatter into [B, max_k, D] with zero-padding for graphs that have
            # fewer selected nodes than k_facts (e.g. small graphs).
            graph_ids = batch[idx]
            counts = torch.bincount(graph_ids, minlength=batch_size)
            max_k = counts.max().item()
            local_pos = torch.cat([torch.arange(c, device=idx.device) for c in counts])
            # Use out-of-place index_put so autograd can track gradients back
            # through x_filtered. In-place assignment (z[i]=v on a leaf tensor)
            # silently breaks the computation graph.
            z_pooled = torch.zeros(
                batch_size, max_k, x_filtered.size(-1),
                device=x_filtered.device, dtype=x_filtered.dtype,
            ).index_put((graph_ids, local_pos), x_filtered)

        return z_pooled