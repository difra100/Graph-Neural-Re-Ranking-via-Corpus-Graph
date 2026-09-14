"""Cahn-Hilliard relevance transport scorer.

Two distinct spaces
-------------------
  Semantic space   x_i ∈ ℝ^d  — fixed TCT-ColBERT features, never updated
  Thermodynamic space  z_i ∈ ℝ — scalar relevance mass; the transported quantity

The model learns how semantics shape the dynamics of z:

    Step 1  z_i^(0) = f(x_i, q)           initial relevance prior
    Step 2  z_i ↔ z_j                      exchange through the corpus graph
    Step 3  z_i*                           equilibrium
    Step 4  rank(i) = z_i*                 z IS the score — no MLP on top

Theoretical framework
---------------------
Each document carries a scalar *relevance mass* z_i.  The free energy is:

    F(Z; q) = Σ_i [z_i U_i + W(z_i)]  +  λ/2 Σ_{ij} (z_i − z_j)²

where:
  U_i     – external query-doc potential:  learned MLP on semantic features.
             Acts as an energy "valley": relevant documents are guided toward
             lower U_i, so they accumulate mass.
  W(z_i)  – double-well: W(z) = γ z²(1−z)²,  W'(z) = 2γ z(1−z)(1−2z).
             Two free-energy minima at z=0 and z=1 enable *phase separation*:
               z < 0.5  →  W' > 0  →  μ rises  →  node sheds mass  (→ z=0)
               z > 0.5  →  W' < 0  →  μ falls  →  node gains mass  (→ z=1)
             Without W the system is purely convex — smooth diffusion, no
             winner-clustering, weak ranking sharpness.
  λ(Lz)  –  graph smoothness: contextual relevance from high-z neighbours.

Chemical potential and flux:
    μ_i = ∂F/∂z_i = U_i + W'(z_i) + λ(L̃z)_i
    dz_i/dt = Σ_j w_{ij}(μ_j − μ_i)

With unit-weight symmetric edges (both directions in edge_index, w_{ij}=1),
the flux Σ_j(μ_j−μ_i) = −(Lμ)_i.  We normalise L̃ = L/d_max so that
ρ(L̃) ≤ 2 for any graph, ensuring Euler stability for α·λ < 0.25.
Row sums of L̃ remain zero, so Σ_i z_i is conserved in the linear limit (γ=0).

Soft mass budget (relaxed conservation)
----------------------------------------
We initialise z via sigmoid + mean-normalisation (E[z^(0)] = 1).  This breaks
exact conservation but gives the system a *soft mass budget*: the training
signal learns to redistribute total mass toward relevant documents.  This is
strictly more expressive than enforcing Σ z_i = const, because phase
separation requires freedom to create large z-asymmetries.

Numerical stability
--------------------
With L̃ = L/d_max:  ρ(L̃²) ≤ 4.  Euler stability requires α·λ·4 < 1.
At initialisation: α = exp(−4) ≈ 0.018, λ = 1, giving α·λ·4 ≈ 0.07 ≪ 1 ✓.
U is bounded to (−1, 1) via tanh, so |L̃U| ≤ 2; with α=0.018 max Δz/step ≈ 0.036.
Backward: tanh gradient ∈ (0,1] — no gradient explosion for any query.
"""

import torch
import torch.nn as nn


class PhysicsTransportGNN(nn.Module):
    """Cahn-Hilliard relevance transport scorer.

    z_i* is the ranking score — this module is a complete, standalone scorer.
    It does not output features for a downstream MLP.

    Parameters
    ----------
    input_dim : dimension of the query-conditioned node features (raw, without
                BM25 rank).  BM25 rank is appended internally.
    n_steps   : Euler integration steps toward thermodynamic equilibrium
    dropout   : dropout in the external-potential MLP
    """

    def __init__(self, input_dim: int, n_steps: int = 6, dropout: float = 0.1):
        super().__init__()
        self.n_steps  = n_steps
        aug_dim       = input_dim + 1          # +1 for BM25 rank position

        # U_i: external query-doc potential (energy landscape shaping)
        # Input: augmented features [semantic + BM25 rank]
        self.potential_net = nn.Sequential(
            nn.Linear(aug_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # z_i^(0): initial relevance mass (learned prior from semantics)
        self.mass_init = nn.Linear(aug_dim, 1)

        # λ > 0: graph-smoothness coupling
        # α > 0: Euler step size
        #   init α = exp(-4) ≈ 0.018: with ‖Lμ‖ ≤ 2 (U normalised),
        #   max Δz per step ≈ 0.036 → z stays in (0.8, 1.2) for n_steps=6,
        #   keeping most z above the clamp boundary → gradients flow.
        # γ > 0: double-well phase-separation strength  (init γ ≈ 0.14)
        self.log_lambda = nn.Parameter(torch.zeros(1))
        self.log_alpha  = nn.Parameter(torch.full((1,), -4.0))
        self.log_gamma  = nn.Parameter(torch.full((1,), -2.0))

    # ------------------------------------------------------------------
    @staticmethod
    def _double_well_prime(z: torch.Tensor) -> torch.Tensor:
        """W'(z) = 2 z(1−z)(1−2z)  for  W(z) = z²(1−z)².

        Phase separation:
          z < 0.5  →  W' > 0  →  μ rises  →  sheds mass  →  z → 0
          z > 0.5  →  W' < 0  →  μ falls  →  gains mass  →  z → 1
          z > 1    →  W' > 0  →  restoring force toward z = 1

        Analytic everywhere; stable minima at z=0 and z=1.
        """
        return 2.0 * z * (1.0 - z) * (1.0 - 2.0 * z)

    # ------------------------------------------------------------------
    @staticmethod
    def _laplacian(f: torch.Tensor, src, dst, deg: torch.Tensor,
                   d_max: torch.Tensor, K: int) -> torch.Tensor:
        """Max-degree-normalised Laplacian: (L̃f)_i = (d_i f_i − Σ_j f_j) / d_max.

        Bounds ρ(L̃) ≤ 2 regardless of graph structure (Euler stability).
        Row sums remain zero (1ᵀL̃ = 0): antisymmetric flux conserves Σ z_i
        in the linear limit.
        """
        nbr = torch.zeros(K, device=f.device, dtype=f.dtype)
        nbr.scatter_add_(0, src, f[dst])
        return (deg * f - nbr) / d_max

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight=None) -> torch.Tensor:
        """
        x          : [K, input_dim]   query-conditioned node features
        edge_index : [2, E]            (src, dst) — both directions present
        Returns    : [K, 1]            equilibrium relevance masses z*
                                       (directly used as ranking scores)
        """
        K = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        lam   = self.log_lambda.exp()
        alpha = self.log_alpha.exp()
        gamma = self.log_gamma.exp()

        # ── Augment semantics with BM25 rank position ──────────────────
        # Rank is mapped to [0, 1] (rank 0 = BM25-best candidate).
        # Without normalisation rank ∈ [0, K−1] up to 999, which would
        # dominate the first linear layer and prevent mass_init from
        # learning meaningful feature weights.
        rank  = torch.arange(K, device=x.device, dtype=x.dtype).unsqueeze(-1)
        rank  = rank / max(K - 1, 1.0)                        # [0, 1]
        x_aug = torch.cat([x, rank], dim=-1)                  # [K, input_dim+1]

        # ── Pre-compute static quantities (outside Euler loop) ──────────

        # U_i: external query potential derived from semantic features.
        # tanh bounds U to (−1, 1) without dividing by std:
        #   • std-based normalisation has gradient ∝ 1/std², which diverges
        #     to ±∞ for queries where U is nearly constant (std → 0),
        #     corrupting Adam state after one backward pass.
        #   • tanh gradient is (1 − tanh²) ∈ (0, 1] everywhere — bounded.
        #   • |(L̃U)_i| ≤ 2|U|_∞ ≤ 2 keeps the Euler step small.
        #   • Lμ only depends on differences in U; the sign convention is
        #     physical (large negative U → low energy → accumulate mass).
        U = torch.tanh(self.potential_net(x_aug).squeeze(-1))  # [K] ∈ (−1, 1)

        # Node degree: both edge directions present → scatter gives |N(i)|
        deg = torch.zeros(K, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, src,
                         torch.ones(src.size(0), device=x.device, dtype=x.dtype))
        d_max = deg.max().clamp(min=1.0)                      # Laplacian normaliser

        # ── Soft mass budget: z^(0) with E[z] = 1 ─────────────────────
        z = torch.sigmoid(self.mass_init(x_aug).squeeze(-1))  # [K] ∈ (0, 1)
        z = z / z.mean().clamp(min=1e-8)                      # mean = 1

        # ── Euler integration of Cahn-Hilliard dynamics ────────────────
        Lmu = torch.zeros_like(z)
        for _ in range(self.n_steps):
            # μ_i = U_i + γ W'(z_i) + λ (L̃z)_i
            #        ↑ external   ↑ phase sep   ↑ contextual propagation
            Lz  = self._laplacian(z, src, dst, deg, d_max, K)
            Wp  = self._double_well_prime(z)
            mu  = U + gamma * Wp + lam * Lz                  # [K]

            # dz_i/dt = −(L̃μ)_i  (antisymmetric; approx. conserves Σ z_i)
            Lmu = self._laplacian(mu, src, dst, deg, d_max, K)
            z   = (z - alpha * Lmu).clamp(min=0.0)          # [K]  mass ≥ 0

        # ── Physics diagnostics (detached — monitoring only) ───────────
        # Populated every forward so the lightning module can log them.
        with torch.no_grad():
            z_d   = z.detach()
            Lz_d  = self._laplacian(z_d, src, dst, deg, d_max, K)
            self.diagnostics = {
                # M(t) = mean relevance mass (init ≈1; explosion/collapse → instability)
                'total_mass':          z_d.mean().item(),
                # V(t) = variance (→0 = oversmoothing; growing = phase separation)
                'variance':            z_d.var().item(),
                # S(t) = z^T L̃ z / K (graph energy; high → sharp relevance boundaries)
                'graph_smoothness':    (z_d * Lz_d).sum().item() / K,
                # R(t) = ||L̃μ|| / √K (convergence residual; →0 at equilibrium)
                'potential_mismatch':  Lmu.detach().norm().item() / (K ** 0.5),
            }

        # z* IS the ranking score — return directly, no downstream MLP
        return z.unsqueeze(-1)                               # [K, 1]
