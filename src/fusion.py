"""
fusion.py
─────────
Blue Agent score fusion: combines LSTM-AE (Stream 1) and GAT GNN (Stream 2).

Fusion formula:
  combined_score = α × lstm_score + (1 − α) × gnn_score
  anomaly_flag   = combined_score > τ

Parameters:
  - α (alpha):     fusion weight, tuned on validation set (grid search 0.0–1.0, step=0.05)
  - τ (threshold): decision boundary, set at optimal F1 point on val data

Variants for ablation (4 configs):
  - LSTM-only   (α = 1.0)
  - GAT-only    (α = 0.0)
  - Both, no fusion (max of individual scores, τ tuned)
  - Full fused  (α = optimal)

Key classes / functions:
  - FusionBlueAgent  : combined detector, AI-search compatible
  - tune_fusion()    : grid-search α & τ on validation set
  - run_ablation()   : benchmark all 4 configs over 5 seeds, return mean±std
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score
)

from src.stat_utils import set_all_seeds

logger = logging.getLogger("rakshak.fusion")


# ══════════════════════════════════════════════════════════════════════════════
#  Score normalisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_scores(scores, eps: float = 1e-9) -> np.ndarray:
    """Min-max normalise scores to [0, 1] for comparability. Accepts arrays or tuples."""
    scores = np.asarray(scores, dtype=np.float64).ravel() if not isinstance(scores, np.ndarray) else scores
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo + eps)


# ══════════════════════════════════════════════════════════════════════════════
#  Fusion decision function
# ══════════════════════════════════════════════════════════════════════════════

def fuse_scores(
    lstm_scores: np.ndarray,   # (N,)
    gnn_scores: np.ndarray,    # (N,)
    alpha: float,              # weight for LSTM stream (1-alpha = GNN)
    threshold: float,
    mode: str = "weighted",    # "weighted" | "max"
) -> np.ndarray:
    """
    Combine LSTM and GNN scores into a binary anomaly prediction.

    Parameters
    ----------
    lstm_scores : np.ndarray (N,)
    gnn_scores  : np.ndarray (N,)
    alpha       : float in [0, 1] — weight on LSTM score
    threshold   : float — decision boundary
    mode        : "weighted" → α×lstm + (1−α)×gnn
                  "max"      → max(lstm_norm, gnn_norm) (no-fusion baseline)

    Returns
    -------
    y_pred : np.ndarray of bool (N,)
    """
    lstm_n = _normalise_scores(lstm_scores)
    gnn_n  = _normalise_scores(gnn_scores)

    if mode == "max":
        combined = np.maximum(lstm_n, gnn_n)
    else:  # weighted
        combined = alpha * lstm_n + (1 - alpha) * gnn_n

    return combined > threshold, combined


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Compute F1, Precision, Recall, AUC-ROC."""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.5
    return {
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "auc_roc":   auc,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Fusion Hyperparameter Tuning
# ══════════════════════════════════════════════════════════════════════════════

def tune_fusion(
    lstm_scores_val: np.ndarray,
    gnn_scores_val: np.ndarray,
    y_val: np.ndarray,
    alpha_range: Tuple[float, float] = (0.0, 1.0),
    alpha_step: float = 0.05,
    threshold_range: Tuple[float, float] = (0.0, 1.0),
    threshold_step: float = 0.01,
) -> Tuple[float, float, float, Dict]:
    """
    Grid-search optimal (alpha, threshold) on the validation set to maximise F1.

    Parameters
    ----------
    lstm_scores_val, gnn_scores_val : np.ndarray (N_val,)
        Raw anomaly scores from LSTM-AE and GAT.
    y_val : np.ndarray (N_val,) — true binary labels (0=normal, 1=anomaly)
    alpha_range, alpha_step : sweep over fusion weights
    threshold_range, threshold_step : sweep over thresholds

    Returns
    -------
    best_alpha, best_threshold, best_f1, best_metrics_dict
    """
    lstm_n = _normalise_scores(lstm_scores_val)
    gnn_n  = _normalise_scores(gnn_scores_val)

    alphas     = np.arange(alpha_range[0], alpha_range[1] + alpha_step / 2, alpha_step)
    thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step / 2, threshold_step)

    best_f1     = -1.0
    best_alpha  = 0.5
    best_tau    = 0.5
    best_metrics: Dict = {}
    sweep_results: List[Dict] = []

    y_true = y_val.astype(int)

    for alpha in alphas:
        combined = alpha * lstm_n + (1 - alpha) * gnn_n
        for tau in thresholds:
            y_pred = (combined > tau).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            sweep_results.append({"alpha": float(alpha), "tau": float(tau), "f1": float(f1)})
            if f1 > best_f1:
                best_f1    = f1
                best_alpha = float(alpha)
                best_tau   = float(tau)
                best_metrics = compute_metrics(y_val, y_pred.astype(bool), combined)

    logger.info(
        f"  Fusion tuning complete: best α={best_alpha:.2f}, τ={best_tau:.3f}, "
        f"F1={best_f1:.4f}"
    )
    return best_alpha, best_tau, best_f1, best_metrics


# ══════════════════════════════════════════════════════════════════════════════
#  FusionBlueAgent — AI-search compatible combined detector
# ══════════════════════════════════════════════════════════════════════════════

class FusionBlueAgent:
    """
    Combined LSTM-AE + GAT Blue Agent for adversarial evaluation.

    Wraps both individual Blue Agents and combines their scores.
    Implements the same stateful interface as LSTMBlueAgent and GATBlueAgent.

    Parameters
    ----------
    lstm_agent : LSTMBlueAgent
    gnn_agent  : GATBlueAgent
    alpha      : float — fusion weight (tuned on validation)
    threshold  : float — decision threshold (tuned on validation)
    mode       : "weighted" or "max"
    """

    def __init__(
        self,
        lstm_agent: Any,
        gnn_agent: Any,
        alpha: float,
        threshold: float,
        mode: str = "weighted",
    ):
        self.lstm_agent = lstm_agent
        self.gnn_agent  = gnn_agent
        self.alpha     = float(alpha)
        self.threshold = float(threshold)
        self.mode      = mode
        self._idx: int = 0

    def set_context(self, idx: int) -> None:
        """Forward context to both sub-agents."""
        self._idx = int(idx)
        self.lstm_agent.set_context(idx)
        self.gnn_agent.set_context(idx)

    def __call__(self, readings: np.ndarray) -> Tuple[bool, float]:
        """
        Evaluate fused detection on (possibly perturbed) sensor readings.

        Returns
        -------
        (detected, combined_score) : (bool, float)
        """
        _, lstm_score = self.lstm_agent(readings)
        _, gnn_score  = self.gnn_agent(readings)

        # Use pre-fitted normalisation ranges if available; else raw scores
        if self.mode == "max":
            combined = max(lstm_score, gnn_score)
        else:
            combined = self.alpha * lstm_score + (1 - self.alpha) * gnn_score

        detected = bool(combined > self.threshold)
        return detected, combined


# ══════════════════════════════════════════════════════════════════════════════
#  Ablation Study
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(
    lstm_scores_test: np.ndarray,   # (N_test,)
    gnn_scores_test: np.ndarray,    # (N_test,)
    lstm_scores_val: np.ndarray,
    gnn_scores_val: np.ndarray,
    y_test: np.ndarray,             # binary labels
    y_val: np.ndarray,
    optimal_alpha: float,
    optimal_tau: float,
) -> Dict[str, Dict]:
    """
    Benchmark 4 ablation configurations on the test set.

    Configs
    -------
    1. LSTM-only   (α=1.0, τ tuned on val LSTM scores)
    2. GAT-only    (α=0.0, τ tuned on val GNN scores)
    3. Both, no fusion (max, τ=0.5)
    4. Full fused  (α=optimal, τ=optimal)

    Returns
    -------
    results : Dict[config_name, metrics_dict]
    """
    from sklearn.metrics import f1_score as _f1

    def _tune_single(scores_val: np.ndarray, y: np.ndarray) -> float:
        """Tune threshold for a single-stream score."""
        best_t, best_f = 0.5, 0.0
        thresholds = np.arange(0.0, 1.01, 0.005)
        norm = _normalise_scores(scores_val)
        for t in thresholds:
            y_p = (norm > t).astype(int)
            f = _f1(y.astype(int), y_p, zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        return best_t

    results: Dict[str, Dict] = {}

    # --- Config 1: LSTM-only ---
    tau_lstm = _tune_single(lstm_scores_val, y_val)
    lstm_n_test = _normalise_scores(lstm_scores_test)
    y_pred_lstm = lstm_n_test > tau_lstm
    results["lstm_only"] = compute_metrics(y_test, y_pred_lstm, lstm_n_test)
    results["lstm_only"]["alpha"] = 1.0
    results["lstm_only"]["tau"]   = tau_lstm

    # --- Config 2: GAT-only ---
    tau_gnn = _tune_single(gnn_scores_val, y_val)
    gnn_n_test = _normalise_scores(gnn_scores_test)
    y_pred_gnn = gnn_n_test > tau_gnn
    results["gat_only"] = compute_metrics(y_test, y_pred_gnn, gnn_n_test)
    results["gat_only"]["alpha"] = 0.0
    results["gat_only"]["tau"]   = tau_gnn

    # --- Config 3: Both, no fusion (max) ---
    combined_max = np.maximum(lstm_n_test, gnn_n_test)
    y_pred_max = combined_max > 0.5
    results["no_fusion_max"] = compute_metrics(y_test, y_pred_max, combined_max)
    results["no_fusion_max"]["alpha"] = None
    results["no_fusion_max"]["tau"]   = 0.5

    # --- Config 4: Full fused (optimal α and τ) ---
    lstm_n_test_f = _normalise_scores(lstm_scores_test)
    gnn_n_test_f  = _normalise_scores(gnn_scores_test)
    combined_fused = optimal_alpha * lstm_n_test_f + (1 - optimal_alpha) * gnn_n_test_f
    y_pred_fused = combined_fused > optimal_tau
    results["full_fused"] = compute_metrics(y_test, y_pred_fused, combined_fused)
    results["full_fused"]["alpha"] = optimal_alpha
    results["full_fused"]["tau"]   = optimal_tau

    for name, metrics in results.items():
        logger.info(
            f"  Ablation [{name}]: "
            f"F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
            f"R={metrics['recall']:.4f}  AUC={metrics['auc_roc']:.4f}"
        )

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-seed ablation runner
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation_multiseed(
    lstm_score_fn,   # callable(seed) → (lstm_scores_val, lstm_scores_test)
    gnn_score_fn,    # callable(seed) → (gnn_scores_val, gnn_scores_test)
    y_val: np.ndarray,
    y_test: np.ndarray,
    seeds: List[int] = (42, 123, 456, 789, 1024),
) -> Dict[str, Dict]:
    """
    Run ablation study across multiple seeds and aggregate mean±std.

    Parameters
    ----------
    lstm_score_fn : callable that takes seed and returns (val_scores, test_scores)
    gnn_score_fn  : callable that takes seed and returns (val_scores, test_scores)

    Returns
    -------
    aggregated : Dict with per-config aggregated metrics
    """
    all_results: Dict[str, List[Dict]] = {
        "lstm_only": [], "gat_only": [], "no_fusion_max": [], "full_fused": []
    }

    for seed in seeds:
        set_all_seeds(seed)
        lstm_val, lstm_test = lstm_score_fn(seed)
        gnn_val, gnn_test   = gnn_score_fn(seed)

        # Tune fusion on val
        best_alpha, best_tau, _, _ = tune_fusion(
            lstm_val, gnn_val, y_val,
            alpha_step=0.1,       # coarser for speed in multi-seed loop
            threshold_step=0.05,
        )

        seed_results = run_ablation(
            lstm_test, gnn_test, lstm_val, gnn_val,
            y_test, y_val, best_alpha, best_tau
        )

        for cfg, metrics in seed_results.items():
            all_results[cfg].append(metrics)

    # Aggregate
    aggregated: Dict[str, Dict] = {}
    metric_keys = ["f1", "precision", "recall", "auc_roc"]
    for cfg, per_seed_list in all_results.items():
        agg: Dict[str, Any] = {}
        for k in metric_keys:
            vals = [r[k] for r in per_seed_list]
            mean, std = float(np.mean(vals)), float(np.std(vals))
            agg[k] = {
                "scores": vals,
                "mean": round(mean, 6),
                "std":  round(std, 6),
                "formatted": f"{mean:.3f}±{std:.3f}",
            }
        aggregated[cfg] = agg

    return aggregated


# ══════════════════════════════════════════════════════════════════════════════
#  Persistence helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_fusion_params(
    alpha: float,
    threshold: float,
    metrics: Dict,
    path: str = "models/fusion_params.json",
) -> None:
    """Save tuned fusion parameters to JSON."""
    payload = {
        "alpha":     alpha,
        "threshold": threshold,
        "metrics":   metrics,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"  Saved fusion params → {path}")


def load_fusion_params(path: str = "models/fusion_params.json") -> Dict:
    """Load previously saved fusion parameters."""
    with open(path) as f:
        return json.load(f)


def save_ablation_results(
    results: Dict,
    path: str = "results/tables/ablation_results.json",
) -> None:
    """Persist aggregated ablation results to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved ablation results → {path}")
