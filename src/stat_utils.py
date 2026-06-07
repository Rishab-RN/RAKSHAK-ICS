"""
RAKSHAK-ICS — Statistical Testing & Multi-Seed Evaluation Utilities
====================================================================

Provides reproducible seed management, multi-seed train/eval harness,
paired statistical testing, and effect-size computation for comparing
anomaly-detection models across multiple random initialisations.
"""

from __future__ import annotations

import os
import random
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    """
    Set seeds for full reproducibility across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        The random seed to use everywhere.

    Notes
    -----
    - Sets ``PYTHONHASHSEED`` for hash-based reproducibility.
    - Configures ``torch.backends.cudnn`` for deterministic behaviour
      (may reduce GPU performance slightly).
    """
    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}")

    # Python builtins
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False      # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Multi-seed evaluation harness
# ---------------------------------------------------------------------------

def run_with_seeds(
    train_fn: Callable[..., Any],
    eval_fn: Callable[..., Dict[str, float]],
    seeds: Optional[List[int]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run training and evaluation across multiple seeds and aggregate results.

    Parameters
    ----------
    train_fn : callable
        ``train_fn(seed=seed, **kwargs) -> model``
        A function that trains a model given a seed and returns the model
        (or any artefact that ``eval_fn`` can consume).
    eval_fn : callable
        ``eval_fn(model, **kwargs) -> dict[str, float]``
        A function that evaluates the trained model and returns a dictionary
        mapping metric names to scalar scores.
    seeds : list of int, optional
        Seeds to iterate over.  Defaults to ``[42, 123, 456, 789, 1024]``.
    **kwargs
        Extra keyword arguments forwarded to both ``train_fn`` and ``eval_fn``.

    Returns
    -------
    dict
        Structure::

            {
                "per_seed": [
                    {"seed": 42, "metrics": {"f1": 0.93, ...}, "train_time": 12.3},
                    ...
                ],
                "aggregated": {
                    "f1": {
                        "scores": [0.93, 0.91, ...],
                        "mean": 0.92,
                        "std": 0.01,
                        "formatted": "0.920±0.010"
                    },
                    ...
                },
                "total_time": 61.5,
                "seeds": [42, 123, 456, 789, 1024],
            }

    Raises
    ------
    ValueError
        If ``seeds`` is empty or ``eval_fn`` returns inconsistent metric keys.
    RuntimeError
        If ``train_fn`` or ``eval_fn`` raises during any seed.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    if not seeds:
        raise ValueError("At least one seed must be provided.")

    per_seed_results: List[Dict[str, Any]] = []
    all_metric_names: Optional[List[str]] = None
    metric_accumulator: Dict[str, List[float]] = {}

    total_start = time.time()

    for seed in seeds:
        set_all_seeds(seed)

        # ---- Train --------------------------------------------------------
        t0 = time.time()
        try:
            model = train_fn(seed=seed, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"train_fn failed for seed={seed}: {exc}"
            ) from exc
        train_time = time.time() - t0

        # ---- Evaluate -----------------------------------------------------
        try:
            metrics: Dict[str, float] = eval_fn(model, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"eval_fn failed for seed={seed}: {exc}"
            ) from exc

        if not isinstance(metrics, dict):
            raise TypeError(
                f"eval_fn must return a dict, got {type(metrics).__name__}"
            )

        # Validate consistent keys
        current_keys = sorted(metrics.keys())
        if all_metric_names is None:
            all_metric_names = current_keys
            for k in all_metric_names:
                metric_accumulator[k] = []
        elif current_keys != all_metric_names:
            raise ValueError(
                f"Inconsistent metric keys across seeds. "
                f"Expected {all_metric_names}, got {current_keys} (seed={seed})."
            )

        for k, v in metrics.items():
            metric_accumulator[k].append(float(v))

        per_seed_results.append({
            "seed": seed,
            "metrics": metrics,
            "train_time": round(train_time, 3),
        })

    total_time = time.time() - total_start

    # ---- Aggregate --------------------------------------------------------
    aggregated: Dict[str, Dict[str, Any]] = {}
    for metric_name, scores in metric_accumulator.items():
        arr = np.array(scores)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        aggregated[metric_name] = {
            "scores": scores,
            "mean": round(mean, 6),
            "std": round(std, 6),
            "formatted": format_result(mean, std),
        }

    return {
        "per_seed": per_seed_results,
        "aggregated": aggregated,
        "total_time": round(total_time, 3),
        "seeds": seeds,
    }


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def paired_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.001,
) -> Dict[str, Any]:
    """
    Paired two-sided t-test between two models' per-seed scores.

    Parameters
    ----------
    scores_a : array-like, shape (n_seeds,)
        Metric scores for model A (one per seed).
    scores_b : array-like, shape (n_seeds,)
        Metric scores for model B (one per seed).
    alpha : float
        Significance level (default: 0.001 for strict ICS evaluation).

    Returns
    -------
    dict
        ``t_statistic`` : float
        ``p_value``      : float
        ``significant``  : bool — whether p < alpha
        ``effect_size``  : float — Cohen's d
        ``alpha``        : float — the threshold used
        ``n``            : int — number of paired observations
        ``mean_diff``    : float — mean of (A − B)
        ``ci_95``        : tuple — 95% confidence interval for mean difference

    Raises
    ------
    ValueError
        If arrays have different lengths or fewer than 2 observations.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f"Score arrays must have the same shape. "
            f"Got {a.shape} and {b.shape}."
        )
    n = len(a)
    if n < 2:
        raise ValueError(
            f"Need at least 2 paired observations for a t-test, got {n}."
        )

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(a, b)

    # Effect size (Cohen's d for paired samples)
    d = cohens_d(a, b)

    # 95% CI for the mean difference
    diffs = a - b
    mean_diff = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(n))
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return {
        "t_statistic": round(float(t_stat), 6),
        "p_value": float(p_val),
        "significant": bool(p_val < alpha),
        "effect_size": round(d, 6),
        "alpha": alpha,
        "n": n,
        "mean_diff": round(mean_diff, 6),
        "ci_95": (round(ci_lower, 6), round(ci_upper, 6)),
    }


def cohens_d(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> float:
    """
    Compute Cohen's d effect size for paired samples.

    Uses the pooled standard deviation as the denominator, following
    the standard definition for independent-like comparison. For paired
    data, this gives a conservative (smaller) effect size estimate.

    Parameters
    ----------
    scores_a, scores_b : array-like
        Score arrays to compare.

    Returns
    -------
    float
        Cohen's d.  Positive values indicate A > B on average.

    Interpretation
    --------------
    |d| < 0.2  — negligible
    |d| < 0.5  — small
    |d| < 0.8  — medium
    |d| >= 0.8 — large
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)

    n_a, n_b = len(a), len(b)

    if n_a < 2 or n_b < 2:
        warnings.warn(
            "Cohen's d requires at least 2 observations per group. "
            "Returning 0.0.",
            stacklevel=2,
        )
        return 0.0

    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)

    # Pooled standard deviation
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0.0:
        # All values identical — no variance
        return 0.0

    return float((mean_a - mean_b) / pooled_std)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_result(mean: float, std: float, decimals: int = 3) -> str:
    """
    Format a result as ``'mean±std'``.

    Parameters
    ----------
    mean : float
        Mean value.
    std : float
        Standard deviation.
    decimals : int
        Number of decimal places (default: 3).

    Returns
    -------
    str
        Formatted string, e.g. ``'0.923±0.012'``.
    """
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)}±{fmt.format(std)}"


# ---------------------------------------------------------------------------
# Convenience: summary table
# ---------------------------------------------------------------------------

def comparison_summary(
    name_a: str,
    name_b: str,
    results_a: Dict[str, Any],
    results_b: Dict[str, Any],
    alpha: float = 0.001,
) -> Dict[str, Any]:
    """
    Build a comparison summary between two models' aggregated results.

    Parameters
    ----------
    name_a, name_b : str
        Model names for reporting.
    results_a, results_b : dict
        Output of ``run_with_seeds`` for each model.
    alpha : float
        Significance level for paired t-tests.

    Returns
    -------
    dict
        Per-metric comparison including t-test results, formatted means, and
        an overall ``winner`` string.
    """
    agg_a = results_a["aggregated"]
    agg_b = results_b["aggregated"]

    common_metrics = sorted(set(agg_a.keys()) & set(agg_b.keys()))
    if not common_metrics:
        raise ValueError("No common metrics between the two result sets.")

    comparisons: Dict[str, Any] = {}
    wins_a, wins_b = 0, 0

    for metric in common_metrics:
        sa = np.array(agg_a[metric]["scores"])
        sb = np.array(agg_b[metric]["scores"])
        test = paired_ttest(sa, sb, alpha=alpha)

        better = name_a if test["mean_diff"] > 0 else name_b
        if test["significant"]:
            if test["mean_diff"] > 0:
                wins_a += 1
            else:
                wins_b += 1

        comparisons[metric] = {
            f"{name_a}_formatted": agg_a[metric]["formatted"],
            f"{name_b}_formatted": agg_b[metric]["formatted"],
            "better": better,
            "significant": test["significant"],
            **test,
        }

    # Determine overall winner
    if wins_a > wins_b:
        winner = name_a
    elif wins_b > wins_a:
        winner = name_b
    else:
        winner = "tie"

    return {
        "metrics": comparisons,
        "significant_wins": {name_a: wins_a, name_b: wins_b},
        "winner": winner,
        "alpha": alpha,
    }
