"""
baselines.py
────────────
Classical ML baseline models for ICS anomaly detection.

Models (Units III–V):
  1. Decision Tree         — interpretable, fast, shallow depth benchmark
  2. Random Forest         — bagged ensemble, robust to noise
  3. K-Nearest Neighbours  — instance-based, good for local patterns
  4. Naive Bayes           — probabilistic, assumes feature independence
  5. Isolation Forest      — unsupervised, isolation-based anomaly scoring
  6. K-Means Clustering    — unsupervised, cluster-distance anomaly scoring
  7. XGBoost               — gradient boosted ensemble (added per project plan)

Usage:
  from src.baselines import run_all_baselines
  results = run_all_baselines("data/proof/")
"""

import logging
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  Synthetic Label Generation (SWaT A9 is clean — no attack labels)
# ══════════════════════════════════════════════════════════════════════

def generate_reconstruction_labels(
    data: np.ndarray,
    threshold_percentile: float = 95.0,
    window_size: int = 60,
) -> np.ndarray:
    """Generate pseudo-anomaly labels from reconstruction error.

    Since SWaT A9 is a CLEAN dataset with no attack labels, we create
    synthetic anomalies for baseline evaluation by:
      1. Computing per-sample feature variance across a sliding window
      2. Flagging samples above the threshold_percentile as anomalous

    This gives baselines a target to classify against, allowing fair
    comparison with the Blue Agent (which also uses reconstruction error).

    Args:
        data: 2D array (n_samples, n_features), scaled values.
        threshold_percentile: Percentile above which samples are anomalous.
        window_size: Rolling window for variance computation.

    Returns:
        Binary labels: 0=normal, 1=anomaly.
    """
    # Compute rolling variance per feature, then average across features
    df = pd.DataFrame(data)
    rolling_var = df.rolling(window=window_size, min_periods=1).var().mean(axis=1)

    threshold = np.percentile(rolling_var.dropna().values, threshold_percentile)
    labels = (rolling_var.values > threshold).astype(int)

    anomaly_rate = labels.mean()
    logger.info(
        f"Generated labels: {labels.sum()} anomalies / {len(labels)} samples "
        f"({anomaly_rate:.2%}), threshold={threshold:.6f}"
    )
    return labels


def inject_synthetic_anomalies(
    data: np.ndarray,
    anomaly_fraction: float = 0.05,
    perturbation_scale: float = 3.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject synthetic anomalies into clean data for baseline evaluation.

    Creates point anomalies by adding Gaussian perturbations to a random
    subset of samples. This is a standard approach for evaluating anomaly
    detectors on clean-only datasets.

    Args:
        data: 2D array (n_samples, n_features), scaled [0,1].
        anomaly_fraction: Fraction of samples to perturb.
        perturbation_scale: Std of Gaussian perturbation (relative to feature std).
        seed: Random seed for reproducibility.

    Returns:
        perturbed_data: Data with injected anomalies.
        labels: Binary labels (0=normal, 1=injected anomaly).
    """
    rng = np.random.RandomState(seed)
    n_samples, n_features = data.shape
    n_anomalies = int(n_samples * anomaly_fraction)

    # Select random indices for anomaly injection
    anomaly_indices = rng.choice(n_samples, size=n_anomalies, replace=False)

    # Create perturbed copy
    perturbed = data.copy()
    feature_stds = np.std(data, axis=0) + 1e-8  # avoid division by zero

    for idx in anomaly_indices:
        # Perturb a random subset of features (2-10 features)
        n_perturb = rng.randint(2, min(10, n_features) + 1)
        features_to_perturb = rng.choice(n_features, size=n_perturb, replace=False)

        for f in features_to_perturb:
            noise = rng.normal(0, perturbation_scale * feature_stds[f])
            perturbed[idx, f] += noise

    # Clip to valid range
    perturbed = np.clip(perturbed, 0, 1)

    labels = np.zeros(n_samples, dtype=int)
    labels[anomaly_indices] = 1

    logger.info(
        f"Injected {n_anomalies} synthetic anomalies ({anomaly_fraction:.1%}) "
        f"with perturbation_scale={perturbation_scale}"
    )
    return perturbed, labels


# ══════════════════════════════════════════════════════════════════════
#  Evaluation Metrics
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute standard anomaly detection metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_score: Anomaly scores (for AUC-ROC). If None, uses y_pred.

    Returns:
        Dict with f1, precision, recall, auc_roc.
    """
    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    # AUC-ROC requires both classes present
    if len(np.unique(y_true)) == 2:
        scores = y_score if y_score is not None else y_pred.astype(float)
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, scores))
        except ValueError:
            metrics["auc_roc"] = 0.0
    else:
        metrics["auc_roc"] = 0.0

    return metrics


# ══════════════════════════════════════════════════════════════════════
#  Individual Model Wrappers
# ══════════════════════════════════════════════════════════════════════

class BaselineModel:
    """Base class for all baseline models."""

    name: str = "base"
    model_type: str = "supervised"  # "supervised" or "unsupervised"

    def __init__(self, **kwargs):
        self.model = None
        self.threshold = None
        self.train_time = 0.0

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        raise NotImplementedError

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score(self, X_test: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        return self.predict(X_test).astype(float)


class DecisionTreeBaseline(BaselineModel):
    name = "decision_tree"
    model_type = "supervised"

    def __init__(self, max_depth: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
        )

    def fit(self, X_train, y_train=None):
        t0 = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - t0

    def predict(self, X_test):
        return self.model.predict(X_test)

    def score(self, X_test):
        proba = self.model.predict_proba(X_test)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]


class RandomForestBaseline(BaselineModel):
    name = "random_forest"
    model_type = "supervised"

    def __init__(self, n_estimators: int = 100, max_depth: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, X_train, y_train=None):
        t0 = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - t0

    def predict(self, X_test):
        return self.model.predict(X_test)

    def score(self, X_test):
        proba = self.model.predict_proba(X_test)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]


class KNNBaseline(BaselineModel):
    name = "knn"
    model_type = "supervised"

    def __init__(self, n_neighbors: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train=None):
        t0 = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - t0

    def predict(self, X_test):
        return self.model.predict(X_test)

    def score(self, X_test):
        proba = self.model.predict_proba(X_test)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]


class NaiveBayesBaseline(BaselineModel):
    name = "naive_bayes"
    model_type = "supervised"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = GaussianNB()

    def fit(self, X_train, y_train=None):
        t0 = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - t0

    def predict(self, X_test):
        return self.model.predict(X_test)

    def score(self, X_test):
        proba = self.model.predict_proba(X_test)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]


class IsolationForestBaseline(BaselineModel):
    name = "isolation_forest"
    model_type = "unsupervised"

    def __init__(self, n_estimators: int = 100, contamination: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, X_train, y_train=None):
        t0 = time.time()
        # Unsupervised: fit on normal data only
        if y_train is not None:
            normal_mask = y_train == 0
            self.model.fit(X_train[normal_mask])
        else:
            self.model.fit(X_train)
        self.train_time = time.time() - t0

    def predict(self, X_test):
        # IsolationForest returns -1 for anomalies, 1 for normal
        preds = self.model.predict(X_test)
        return (preds == -1).astype(int)

    def score(self, X_test):
        # Negative decision function = more anomalous
        return -self.model.decision_function(X_test)


class KMeansBaseline(BaselineModel):
    name = "kmeans"
    model_type = "unsupervised"

    def __init__(self, n_clusters: int = 5, threshold_percentile: float = 95, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.threshold_percentile = threshold_percentile
        self.model = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=42,
        )

    def fit(self, X_train, y_train=None):
        t0 = time.time()
        if y_train is not None:
            normal_mask = y_train == 0
            self.model.fit(X_train[normal_mask])
        else:
            self.model.fit(X_train)
        # Set threshold from training distances
        train_distances = self.model.transform(X_train).min(axis=1)
        self.threshold = np.percentile(train_distances, self.threshold_percentile)
        self.train_time = time.time() - t0

    def predict(self, X_test):
        distances = self.model.transform(X_test).min(axis=1)
        return (distances > self.threshold).astype(int)

    def score(self, X_test):
        return self.model.transform(X_test).min(axis=1)


class XGBoostBaseline(BaselineModel):
    name = "xgboost"
    model_type = "supervised"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                tree_method="gpu_hist" if use_gpu else "hist",
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            )
        except ImportError:
            logger.warning("XGBoost not installed, falling back to RandomForest")
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=-1,
                random_state=42,
            )

    def fit(self, X_train, y_train=None):
        t0 = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - t0

    def predict(self, X_test):
        return self.model.predict(X_test)

    def score(self, X_test):
        proba = self.model.predict_proba(X_test)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]


# ══════════════════════════════════════════════════════════════════════
#  Model Registry
# ══════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: Dict[str, type] = {
    "decision_tree": DecisionTreeBaseline,
    "random_forest": RandomForestBaseline,
    "knn": KNNBaseline,
    "naive_bayes": NaiveBayesBaseline,
    "isolation_forest": IsolationForestBaseline,
    "kmeans": KMeansBaseline,
    "xgboost": XGBoostBaseline,
}


def get_model(name: str, **kwargs) -> BaselineModel:
    """Get a baseline model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)


# ══════════════════════════════════════════════════════════════════════
#  Single Model Train + Eval (for use with run_with_seeds)
# ══════════════════════════════════════════════════════════════════════

def train_baseline(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    **kwargs,
) -> BaselineModel:
    """Train a single baseline model.

    Compatible with stat_utils.run_with_seeds() as train_fn.
    """
    np.random.seed(seed)
    model = get_model(model_name, **kwargs)
    model.fit(X_train, y_train)
    logger.info(f"Trained {model_name} in {model.train_time:.2f}s (seed={seed})")
    return model


def eval_baseline(
    model: BaselineModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **kwargs,
) -> Dict[str, float]:
    """Evaluate a trained baseline model.

    Compatible with stat_utils.run_with_seeds() as eval_fn.
    """
    y_pred = model.predict(X_test)
    y_score = model.score(X_test)
    metrics = compute_metrics(y_test, y_pred, y_score)
    return metrics


# ══════════════════════════════════════════════════════════════════════
#  Run All Baselines
# ══════════════════════════════════════════════════════════════════════

def flatten_windows(windows: np.ndarray) -> np.ndarray:
    """Flatten 3D sliding windows (N, T, F) to 2D (N, T*F) for sklearn.

    Sklearn models expect 2D input. We flatten each window into a single
    feature vector by concatenating all timesteps.
    """
    if windows.ndim == 3:
        n, t, f = windows.shape
        return windows.reshape(n, t * f)
    return windows


def run_all_baselines(
    data_dir: str = "data/proof/",
    config: Optional[dict] = None,
    anomaly_fraction: float = 0.05,
    models: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run all baseline models with multi-seed evaluation.

    Since SWaT A9 is clean (no attack labels), we inject synthetic
    anomalies into the test set for evaluation.

    Args:
        data_dir: Directory with processed .npy files.
        config: Optional config dict with baseline params.
        anomaly_fraction: Fraction of synthetic anomalies to inject.
        models: List of model names to run. None = all.
        seeds: Seeds for multi-seed evaluation.

    Returns:
        Dict with per-model results including mean±std metrics.
    """
    from src.preprocess import load_processed_data
    from src.stat_utils import run_with_seeds, format_result

    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    if models is None:
        models = list(MODEL_REGISTRY.keys())

    # Load processed data
    logger.info("Loading processed data...")
    data = load_processed_data(data_dir)
    X_train_windows = data["X_train"]
    X_test_windows = data["X_test"]

    # Flatten windows for sklearn (N, T, F) → (N, T*F)
    X_train_flat = flatten_windows(X_train_windows)
    X_test_flat = flatten_windows(X_test_windows)

    logger.info(f"Train: {X_train_flat.shape}, Test: {X_test_flat.shape}")

    all_results = {}

    for model_name in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {model_name}...")
        logger.info(f"{'='*60}")

        try:
            seed_results = run_with_seeds(
                train_fn=lambda seed, **kw: _train_with_injection(
                    model_name, X_train_flat, X_test_flat,
                    anomaly_fraction, seed,
                ),
                eval_fn=lambda model, **kw: _eval_with_injection(
                    model, model_name, X_test_flat,
                    anomaly_fraction, kw.get("seed", 42),
                ),
                seeds=seeds,
            )
            all_results[model_name] = seed_results

            # Log summary
            agg = seed_results["aggregated"]
            logger.info(
                f"{model_name}: "
                f"F1={agg['f1']['formatted']}, "
                f"AUC={agg['auc_roc']['formatted']}"
            )

        except Exception as e:
            logger.error(f"Failed to run {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}

    return all_results


def _train_with_injection(
    model_name: str,
    X_train_flat: np.ndarray,
    X_test_flat: np.ndarray,
    anomaly_fraction: float,
    seed: int,
) -> Tuple[BaselineModel, np.ndarray, np.ndarray]:
    """Train helper that injects synthetic anomalies."""
    # For supervised models: inject anomalies into training data
    model = get_model(model_name)

    if model.model_type == "supervised":
        X_train_aug, y_train = inject_synthetic_anomalies(
            X_train_flat, anomaly_fraction, seed=seed
        )
        model.fit(X_train_aug, y_train)
    else:
        # Unsupervised: train on clean data only
        model.fit(X_train_flat)

    # Generate test labels with same seed for consistency
    _, y_test = inject_synthetic_anomalies(
        X_test_flat, anomaly_fraction, seed=seed + 1000
    )
    return (model, X_test_flat, y_test)


def _eval_with_injection(
    result_tuple,
    model_name: str,
    X_test_flat: np.ndarray,
    anomaly_fraction: float,
    seed: int,
) -> Dict[str, float]:
    """Eval helper for injected anomalies."""
    model, X_test, y_test = result_tuple

    # For unsupervised models on injected data
    X_test_aug, y_test_aug = inject_synthetic_anomalies(
        X_test_flat, anomaly_fraction, seed=seed + 1000
    )
    y_pred = model.predict(X_test_aug)
    y_score = model.score(X_test_aug)
    return compute_metrics(y_test_aug, y_pred, y_score)


# ══════════════════════════════════════════════════════════════════════
#  Results Formatting & Saving
# ══════════════════════════════════════════════════════════════════════

def format_results_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Format multi-model results as a comparison DataFrame.

    Args:
        results: Output of run_all_baselines().

    Returns:
        DataFrame with models as rows, metrics as columns (mean±std).
    """
    rows = []
    for model_name, result in results.items():
        if "error" in result:
            rows.append({
                "Model": model_name,
                "F1": "ERROR",
                "Precision": "ERROR",
                "Recall": "ERROR",
                "AUC-ROC": "ERROR",
            })
            continue

        agg = result["aggregated"]
        rows.append({
            "Model": model_name,
            "F1": agg.get("f1", {}).get("formatted", "—"),
            "Precision": agg.get("precision", {}).get("formatted", "—"),
            "Recall": agg.get("recall", {}).get("formatted", "—"),
            "AUC-ROC": agg.get("auc_roc", {}).get("formatted", "—"),
        })

    return pd.DataFrame(rows)


def save_results(
    results: Dict[str, Any],
    output_dir: str = "results/tables/",
    filename: str = "baseline_results.json",
):
    """Save baseline results to JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean = json.loads(json.dumps(results, default=_convert))

    path = out / filename
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info(f"Results saved to {path}")


# ══════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="RAKSHAK-ICS: Run ML Baseline Models"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/proof/",
        help="Directory with processed .npy files",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "--anomaly-fraction", type=float, default=0.05,
        help="Fraction of synthetic anomalies (default: 0.05)",
    )
    args = parser.parse_args()

    results = run_all_baselines(
        data_dir=args.data_dir,
        models=args.models,
        anomaly_fraction=args.anomaly_fraction,
    )

    # Print results table
    table = format_results_table(results)
    print(f"\n{'='*70}")
    print("ML Baseline Results (mean±std across 5 seeds)")
    print(f"{'='*70}")
    print(table.to_string(index=False))
    print(f"{'='*70}")

    # Save
    save_results(results)
