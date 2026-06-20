"""
gnn.py
──────
GAT GNN for inter-sensor dependency modelling (Blue Agent — Stream 2).

Graph Construction:
  - 65 nodes (one per sensor/actuator, after preprocessing)
  - Edges: Pearson correlation ≥ 0.7 between sensor pairs (436 edges from normal data)
  - Node features: [mean, std, min, max, range] per sliding window (5-dim per node)

Architecture (PyG if available, else pure-PyTorch fallback):
  - GATConv Layer 1: (5 → 16, heads=8, edge_dim=1), ReLU, dropout=0.1 → 128-dim concat
  - GATConv Layer 2: (128 → 5, heads=1), reconstruction head → 5-dim output
  - Anomaly score = mean per-node MSE(input, reconstructed), aggregated per graph

Training:
  - Reconstruction-based on normal-only graph snapshots
  - MSE loss, Adam optimizer, ReduceLROnPlateau scheduler, early stopping

Output:
  - gnn_score: float per window (higher = more anomalous)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("rakshak.gnn")

# ─── Detect PyTorch Geometric ────────────────────────────────────────────────
try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data, Batch
    _PYG_AVAILABLE = True
    logger.info("PyTorch Geometric detected — using GATConv")
except ImportError:
    _PYG_AVAILABLE = False
    logger.info("PyTorch Geometric NOT found — using pure-PyTorch GAT fallback")


# ══════════════════════════════════════════════════════════════════════════════
#  Pure-PyTorch GAT Layer (fallback when PyG is unavailable)
# ══════════════════════════════════════════════════════════════════════════════

class GATLayerFallback(nn.Module):
    """
    Manual GAT attention layer (no PyG required).
    Implements the Veličković et al. 2018 attention mechanism via sparse matmul.

    For 65-node graphs this runs efficiently on CPU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # Linear transform: input → (heads × out_channels)
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention parameters: a_src and a_dst (2-way attention)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))

        # Optional edge feature projection
        if edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.edge_proj = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.edge_proj is not None:
            nn.init.xavier_uniform_(self.edge_proj.weight)

    def forward(
        self,
        x: torch.Tensor,          # (N, in_channels)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: Optional[torch.Tensor] = None,  # (E, edge_dim)
    ) -> torch.Tensor:
        N = x.size(0)
        H, C = self.heads, self.out_channels

        # 1) Linear transform → (N, H, C)
        x_t = self.lin(x).view(N, H, C)

        # 2) Compute attention logits
        # alpha_src: (N, H), alpha_dst: (N, H)
        alpha_src = (x_t * self.att_src).sum(dim=-1)  # (N, H)
        alpha_dst = (x_t * self.att_dst).sum(dim=-1)  # (N, H)

        src, dst = edge_index[0], edge_index[1]  # (E,)

        # Edge attention: alpha_src[i] + alpha_dst[j] for edge (i→j)
        alpha = alpha_src[src] + alpha_dst[dst]  # (E, H)

        # Optional edge feature contribution
        if self.edge_proj is not None and edge_attr is not None:
            e_feat = self.edge_proj(edge_attr).view(-1, H, C)
            alpha = alpha + (e_feat * self.att_src).sum(dim=-1)

        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        # Softmax per destination node
        alpha = torch.exp(alpha - alpha.max())  # (E, H) numerically stable
        # Aggregate denominator per dst node
        denom = torch.zeros(N, H, device=x.device)
        denom.scatter_add_(0, dst.unsqueeze(1).expand(-1, H), alpha)
        alpha = alpha / (denom[dst] + 1e-9)  # (E, H)

        # Apply dropout to attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 3) Aggregate messages: out[j] = Σ_{i∈N(j)} alpha[i,j] * x_t[i]
        # alpha: (E, H), x_t[src]: (E, H, C)
        msg = alpha.unsqueeze(-1) * x_t[src]  # (E, H, C)
        out = torch.zeros(N, H, C, device=x.device)
        # Scatter add for each destination
        idx = dst.view(-1, 1, 1).expand_as(msg)
        out.scatter_add_(0, idx, msg)

        if self.concat:
            out = out.view(N, H * C)   # (N, H*C)
        else:
            out = out.mean(dim=1)      # (N, C)

        return out


# ══════════════════════════════════════════════════════════════════════════════
#  SensorGAT — main model (uses PyG or fallback transparently)
# ══════════════════════════════════════════════════════════════════════════════

class SensorGAT(nn.Module):
    """
    Two-layer GAT for sensor-graph reconstruction anomaly detection.

    The model encodes each graph snapshot via attention over the Pearson-
    correlation sensor graph and reconstructs the 5-dim node features.
    Anomaly score = mean per-node MSE(input, reconstructed).

    Parameters
    ----------
    node_feature_dim : int
        Dimensionality of input node features (default 5: mean/std/min/max/range).
    hidden_dim : int
        Hidden dimension per attention head (default 16).
    num_heads : int
        Number of attention heads in layer 1 (default 8).
    dropout : float
        Attention dropout (default 0.1).
    edge_dim : int or None
        Dimensionality of edge attributes (1 for correlation weight, or None).
    """

    def __init__(
        self,
        node_feature_dim: int = 5,
        hidden_dim: int = 16,
        num_heads: int = 8,
        dropout: float = 0.1,
        edge_dim: Optional[int] = 1,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.use_pyg = _PYG_AVAILABLE

        mid_dim = hidden_dim * num_heads  # 128

        if _PYG_AVAILABLE:
            self.conv1 = GATConv(
                in_channels=node_feature_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_dim,
            )
            self.conv2 = GATConv(
                in_channels=mid_dim,
                out_channels=node_feature_dim,
                heads=1,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim,
            )
        else:
            self.conv1 = GATLayerFallback(
                in_channels=node_feature_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_dim,
            )
            self.conv2 = GATLayerFallback(
                in_channels=mid_dim,
                out_channels=node_feature_dim,
                heads=1,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim,
            )

        self.norm1 = nn.LayerNorm(mid_dim)
        self.norm2 = nn.LayerNorm(node_feature_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,           # (N_nodes, node_feature_dim) per graph
        edge_index: torch.Tensor,   # (2, E)
        edge_attr: Optional[torch.Tensor] = None,  # (E, 1)
    ) -> torch.Tensor:
        """Forward pass — returns reconstructed node features (N_nodes, node_feature_dim)."""
        # Layer 1
        if self.use_pyg:
            h = self.conv1(x, edge_index, edge_attr)
        else:
            h = self.conv1(x, edge_index, edge_attr)
        h = self.norm1(h)
        h = F.elu(h)
        h = self.dropout(h)

        # Layer 2 — reconstruction
        if self.use_pyg:
            out = self.conv2(h, edge_index, edge_attr)
        else:
            out = self.conv2(h, edge_index, edge_attr)
        out = self.norm2(out)
        return out

    def anomaly_score(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Per-graph anomaly score: mean MSE over all node features.

        Parameters
        ----------
        x : Tensor (N_nodes, node_feature_dim)
        edge_index : Tensor (2, E)
        edge_attr : Tensor (E, 1) or None

        Returns
        -------
        score : scalar Tensor
        """
        with torch.no_grad():
            x_hat = self(x, edge_index, edge_attr)
            per_node_mse = ((x - x_hat) ** 2).mean(dim=-1)  # (N_nodes,)
            return per_node_mse.mean()


# ══════════════════════════════════════════════════════════════════════════════
#  Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def _build_batched_graph(
    node_features_batch: torch.Tensor,   # (B, N_nodes, node_feature_dim)
    edge_index: torch.Tensor,            # (2, E)
    edge_weights: torch.Tensor,          # (E, 1)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stack B individual graphs into one large disconnected super-graph.

    Returns
    -------
    x_batch     : (B × N_nodes, node_feature_dim)
    ei_batch    : (2, B × E)  — each copy offset by n_nodes × graph_idx
    ea_batch    : (B × E, 1)
    """
    B = node_features_batch.shape[0]
    N = node_features_batch.shape[1]  # 65

    # Flatten node features: (B×N, F)
    x_batch = node_features_batch.reshape(B * N, -1)

    # Tile edge index with offsets
    offsets = torch.arange(B, device=edge_index.device).unsqueeze(1) * N  # (B, 1)
    ei_tiled = edge_index.unsqueeze(0).expand(B, -1, -1)   # (B, 2, E)
    ei_batch = (ei_tiled + offsets.unsqueeze(-1)).reshape(2, -1)   # (2, B*E)

    # Tile edge attributes
    ea_batch = edge_weights.unsqueeze(0).expand(B, -1, -1).reshape(-1, edge_weights.shape[-1])  # (B*E, 1)

    return x_batch, ei_batch, ea_batch


def train_gat(
    model: SensorGAT,
    node_features_train: np.ndarray,   # (N_windows, 65, 5)
    edge_index: np.ndarray,            # (2, E)
    edge_weights: np.ndarray,          # (E,)
    node_features_val: np.ndarray,     # (N_windows, 65, 5)
    learning_rate: float = 0.001,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 256,             # graphs per batch
    subsample: int = 3,                # take every Nth window (speed)
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """
    Train SensorGAT on normal-only graph snapshots using batched forward passes.

    The full dataset is split into mini-batches of `batch_size` graphs each.
    All graphs in a batch are processed as a single disconnected super-graph,
    giving ~100× speedup over the graph-by-graph loop.

    Parameters
    ----------
    subsample : int
        Use every `subsample`-th window for speed (3 = ~3× faster).

    Returns
    -------
    history : dict with 'train_loss' and 'val_loss' lists.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Convert graph topology to tensors
    ei = torch.tensor(edge_index, dtype=torch.long, device=device)     # (2, E)
    ea = torch.tensor(edge_weights, dtype=torch.float32, device=device).unsqueeze(1)  # (E,1)

    # Subsample windows for speed
    X_tr = node_features_train[::subsample]   # (N/3, 65, 5)
    X_vl = node_features_val[::subsample]

    # Convert to torch
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_vl_t = torch.tensor(X_vl, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, min_lr=1e-5
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    N_tr = len(X_tr_t)
    N_vl = len(X_vl_t)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        # Shuffle indices
        perm = torch.randperm(N_tr)
        X_tr_t = X_tr_t[perm]

        n_batches = 0
        for start in range(0, N_tr, batch_size):
            batch = X_tr_t[start : start + batch_size].to(device)  # (B, 65, 5)
            B = batch.shape[0]

            # Build super-graph for this batch
            x_b, ei_b, ea_b = _build_batched_graph(batch, ei, ea)

            # Forward
            x_hat_b = model(x_b, ei_b, ea_b)  # (B*65, 5)

            loss = F.mse_loss(x_hat_b, x_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * B
            n_batches += 1

        avg_train = epoch_loss / N_tr

        # Validation (batched)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for start in range(0, N_vl, batch_size):
                batch = X_vl_t[start : start + batch_size].to(device)
                B = batch.shape[0]
                x_b, ei_b, ea_b = _build_batched_graph(batch, ei, ea)
                x_hat_b = model(x_b, ei_b, ea_b)
                val_loss += F.mse_loss(x_hat_b, x_b).item() * B

        avg_val = val_loss / N_vl

        scheduler.step(avg_val)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train={avg_train:.6f}  val={avg_val:.6f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"[{elapsed:.1f}s/ep, {n_batches} batches]"
            )

        # Early stopping — use 1e-4 tolerance to avoid micro-jitter resetting counter
        if avg_val < best_val_loss - 1e-4:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.to("cpu")
    return history



# ══════════════════════════════════════════════════════════════════════════════
#  Score computation & threshold tuning
# ══════════════════════════════════════════════════════════════════════════════

def compute_gat_scores(
    model: SensorGAT,
    node_features: np.ndarray,   # (N_windows, 65, 5)
    edge_index: np.ndarray,      # (2, E)
    edge_weights: np.ndarray,    # (E,)
    batch_size: int = 512,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Compute per-window GAT anomaly scores.

    Returns
    -------
    scores : np.ndarray of shape (N_windows,)
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device).eval()
    ei = torch.tensor(edge_index, dtype=torch.long, device=device)
    ea = torch.tensor(edge_weights, dtype=torch.float32, device=device).unsqueeze(1)
    X = torch.tensor(node_features, dtype=torch.float32, device=device)

    scores = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size].to(device)  # (B, 65, 5)
            B = batch.shape[0]
            N = batch.shape[1]  # 65

            x_b, ei_b, ea_b = _build_batched_graph(batch, ei, ea)
            x_hat_b = model(x_b, ei_b, ea_b)  # (B*65, 5)

            # Per-node MSE, then mean per graph
            mse_per_node = ((x_b - x_hat_b) ** 2).mean(dim=-1)   # (B*65,)
            mse_per_graph = mse_per_node.view(B, N).mean(dim=-1)  # (B,)
            scores.extend(mse_per_graph.cpu().tolist())

    return np.array(scores)


def tune_gat_threshold(
    val_scores: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Set detection threshold at `percentile`-th percentile of val reconstruction errors."""
    tau = float(np.percentile(val_scores, percentile))
    logger.info(f"  GAT threshold τ = {tau:.6f} (at {percentile}th percentile of val)")
    return tau


# ══════════════════════════════════════════════════════════════════════════════
#  Blue Agent wrapper (AI-search compatible interface)
# ══════════════════════════════════════════════════════════════════════════════

class GATBlueAgent:
    """
    Stateful Blue Agent wrapper around SensorGAT for AI-search attacker evaluation.

    Usage
    -----
    agent = GATBlueAgent(model, X_test_node_features, edge_index, edge_weights, threshold)
    agent.set_context(idx)                        # set window index
    detected, score = agent(perturbed_readings)   # perturbed_readings shape (65,) or (60, 65)
    """

    def __init__(
        self,
        model: SensorGAT,
        X_test_node_features: np.ndarray,  # (N, 65, 5)
        edge_index: np.ndarray,
        edge_weights: np.ndarray,
        threshold: float,
        X_test_windows: Optional[np.ndarray] = None,  # (N, 60, 65) for rebuilding node feats
    ):
        self.model = model.eval()
        self.X_test_nf = X_test_node_features   # (N, 65, 5)
        self.X_test_windows = X_test_windows    # (N, 60, 65) optional
        self.ei = torch.tensor(edge_index, dtype=torch.long)
        self.ea = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
        self.threshold = threshold
        self._idx: int = 0

    def set_context(self, idx: int) -> None:
        """Set the active test window index (called by ai_search.run_all_attackers)."""
        self._idx = int(idx)

    def _node_features_from_window(self, window: np.ndarray) -> np.ndarray:
        """Compute 5-dim node features from a (60, 65) window."""
        # window: (T, F)  → per-sensor stats → (F, 5)
        nf = np.stack([
            window.mean(axis=0),
            window.std(axis=0),
            window.min(axis=0),
            window.max(axis=0),
            window.max(axis=0) - window.min(axis=0),
        ], axis=1)   # (65, 5)
        return nf

    def __call__(self, readings: np.ndarray) -> Tuple[bool, float]:
        """
        Evaluate detection on (possibly perturbed) sensor readings.

        Parameters
        ----------
        readings : np.ndarray
            Shape (65,) — the perturbed sensor reading for the current timestep,
            or (60, 65) — full window override.

        Returns
        -------
        (detected, score) : (bool, float)
        """
        readings = np.asarray(readings, dtype=np.float32)

        if readings.ndim == 1:
            # Build node features from stored test window, with last timestep replaced
            if self.X_test_windows is not None:
                window = self.X_test_windows[self._idx].copy()  # (60, 65)
                window[-1] = readings
            else:
                # Fallback: just use stored node features (no perturbation on node features)
                window = None

            if window is not None:
                nf = self._node_features_from_window(window)
            else:
                nf = self.X_test_nf[self._idx].copy()
        elif readings.ndim == 2:
            nf = self._node_features_from_window(readings)
        else:
            raise ValueError(f"Unexpected readings shape: {readings.shape}")

        x = torch.tensor(nf, dtype=torch.float32)
        score = self.model.anomaly_score(x, self.ei, self.ea).item()
        detected = bool(score > self.threshold)
        return detected, score


# ══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Quick smoke test: load preprocessed data, train 1-epoch GAT, report score."""
    import json
    from src.stat_utils import set_all_seeds

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    PROOF = Path("data/proof")
    MODELS = Path("models")
    MODELS.mkdir(exist_ok=True)

    set_all_seeds(42)

    # Load data
    logger.info("Loading preprocessed node features...")
    X_train_nf = np.load(PROOF / "node_features_train.npy")   # (N, 65, 5)
    X_val_nf   = np.load(PROOF / "node_features_val.npy")
    edge_index  = np.load(PROOF / "edge_index.npy")            # (2, E)
    edge_weights = np.load(PROOF / "edge_weights.npy")         # (E,)

    logger.info(f"  Train snapshots: {X_train_nf.shape}")
    logger.info(f"  Val   snapshots: {X_val_nf.shape}")
    logger.info(f"  Graph: {edge_index.shape[1]} edges")

    model = SensorGAT(
        node_feature_dim=5,
        hidden_dim=16,
        num_heads=8,
        dropout=0.1,
        edge_dim=1,
    )

    history = train_gat(
        model,
        X_train_nf,
        edge_index,
        edge_weights,
        X_val_nf,
        epochs=2,       # smoke test
        patience=10,
        subsample=10,   # very fast
    )

    val_scores = compute_gat_scores(model, X_val_nf, edge_index, edge_weights)
    tau = tune_gat_threshold(val_scores)

    logger.info(f"Smoke test passed. Val score mean: {val_scores.mean():.6f}, τ={tau:.6f}")
    torch.save(model.state_dict(), MODELS / "gnn.pt")
    logger.info("Saved models/gnn.pt")


if __name__ == "__main__":
    main()
