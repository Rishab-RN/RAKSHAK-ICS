"""
Data preprocessing pipeline for SWaT and HAI datasets.

This module implements the full data pipeline for RAKSHAK-ICS:
  - Load and concatenate raw SWaT A9 CSV files
  - Clean: handle 'Bad Input' values, drop timestamps, remove constant columns
  - Encode alarm columns (Active=1, Inactive=0)
  - Normalise: fit StandardScaler/MinMaxScaler on TRAIN-NORMAL only
  - Sliding windows: (num_windows, window_size, n_features) for LSTM/autoencoder input
  - Graph construction: Pearson correlation >= threshold -> edge_index for GNN
  - Node features: [mean, std, min, max, range] per sensor per window
  - Temporal train/val/test split (NO shuffle — time-series data)
  - Save privacy-safe dataset statistics (no raw values)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import yaml
import torch
import warnings

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Default configuration (used when no YAML config file is found)
# ──────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "data": {
        "raw_dir": "data/swat/",
        "train_files": ["dataset1.csv", "dataset2.csv"],
        "test_files": ["dataset3.csv"],
        "proof_dir": "data/proof/",
    },
    "preprocessing": {
        "window_size": 60,
        "stride": 1,
        "scaler": "standard",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "correlation_threshold": 0.7,
        "remove_constant_cols": True,
        "bad_input_strategy": "ffill",  # 'ffill', 'interpolate', 'drop'
        "min_non_null_ratio": 0.5,  # drop columns with < 50% non-null after cleaning
    },
}

# ──────────────────────────────────────────────────────────────────────
# Known SWaT A9 column patterns (from actual dataset inspection)
# ──────────────────────────────────────────────────────────────────────

# Alarm columns always contain 'Active'/'Inactive'/'Bad Input' strings
ALARM_COLUMNS = [
    "LS201.Alarm", "LS202.Alarm", "LSL203.Alarm", "LSLL203.Alarm",
    "PSH301.Alarm", "DPSH301.Alarm",
    "LS401.Alarm",
    "PSH501.Alarm", "PSL501.Alarm",
    "LSH601.Alarm", "LSL601.Alarm", "LSH602.Alarm", "LSL602.Alarm",
    "LSH603.Alarm", "LSL603.Alarm",
]

# Speed columns
SPEED_COLUMNS = ["P501.Speed", "P502.Speed"]

# Process state columns (integer-valued)
STATE_COLUMNS = [f"P{i}_STATE" for i in range(1, 7)]

TIMESTAMP_COLUMN = "t_stamp"

# Timestamp format in SWaT A9 CSVs
TIMESTAMP_FORMAT = "%d-%b-%Y %H:%M:%S"


# ══════════════════════════════════════════════════════════════════════
#  Column Classification
# ══════════════════════════════════════════════════════════════════════

def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Classify SWaT dataframe columns into semantic categories.

    Returns a dict with keys:
        'continuous'  — .Pv sensor readings (float)
        'binary'      — .Status actuator states (0/1 or 1/2)
        'alarm'       — .Alarm columns (Active/Inactive)
        'speed'       — .Speed columns (float)
        'state'       — P*_STATE process state columns (int)
        'timestamp'   — timestamp column(s)
        'constant'    — columns with zero variance (detected later)
    """
    columns = list(df.columns)

    timestamp_cols = [c for c in columns if c.lower() in ("t_stamp", "timestamp")]

    alarm_cols = [c for c in columns if c.endswith(".Alarm")]
    speed_cols = [c for c in columns if c.endswith(".Speed")]
    status_cols = [c for c in columns if c.endswith(".Status")]
    state_cols = [c for c in columns if c in STATE_COLUMNS or
                  (c.startswith("P") and c.endswith("_STATE"))]

    # Continuous = .Pv columns (sensor readings)
    pv_cols = [c for c in columns if c.endswith(".Pv")]

    # Anything remaining that isn't classified
    classified = set(timestamp_cols + alarm_cols + speed_cols +
                     status_cols + state_cols + pv_cols)
    other_cols = [c for c in columns if c not in classified]

    # Detect constant columns (requires numeric data — do after cleaning)
    # For now, return empty; will be populated after cleaning
    constant_cols: List[str] = []

    categories = {
        "timestamp": timestamp_cols,
        "continuous": pv_cols,
        "binary": status_cols,
        "alarm": alarm_cols,
        "speed": speed_cols,
        "state": state_cols,
        "constant": constant_cols,
        "other": other_cols,
    }

    logger.info(
        f"Column classification: "
        f"{len(pv_cols)} continuous, {len(status_cols)} binary/status, "
        f"{len(alarm_cols)} alarm, {len(speed_cols)} speed, "
        f"{len(state_cols)} state, {len(timestamp_cols)} timestamp, "
        f"{len(other_cols)} other"
    )

    return categories


# ══════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_swat_data(
    data_dir: str = "data/swat/",
    files: Optional[List[str]] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Load and concatenate SWaT CSV files.

    Handles:
      - Mixed types in columns (numbers + 'Bad Input' strings)
      - Automatic file discovery if files not specified
      - Empty trailing rows

    Args:
        data_dir: Path to directory containing CSV files.
        files: List of CSV filenames. If None, auto-discovers dataset*.csv.
        config: Optional config dict (unused here but reserved for future use).

    Returns:
        Concatenated DataFrame with all CSV data.
    """
    data_path = Path(data_dir)

    if files is None:
        # Auto-discover dataset files
        csv_files = sorted(data_path.glob("dataset*.csv"))
        if not csv_files:
            csv_files = sorted(data_path.glob("*.csv"))
        files = [f.name for f in csv_files]
        logger.info(f"Auto-discovered {len(files)} CSV files in {data_dir}")

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            f"Expected dataset1.csv, dataset2.csv, dataset3.csv"
        )

    dfs = []
    for fname in files:
        filepath = data_path / fname
        if not filepath.exists():
            logger.warning(f"File not found, skipping: {filepath}")
            continue

        logger.info(f"Loading {filepath} ...")

        # Read all columns as strings first to handle mixed types safely,
        # then selectively convert. This avoids pandas DtypeWarning.
        df = pd.read_csv(
            filepath,
            dtype=str,        # read everything as string to avoid mixed-type issues
            na_values=[""],   # treat empty strings as NaN
            keep_default_na=True,
            low_memory=False,
        )

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Drop completely empty rows (e.g., trailing blank line)
        df.dropna(how="all", inplace=True)

        logger.info(f"  {fname}: {df.shape[0]} rows, {df.shape[1]} columns")
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No valid CSV files loaded from {data_dir}")

    # Concatenate all files
    combined = pd.concat(dfs, axis=0, ignore_index=True)
    logger.info(f"Combined dataset: {combined.shape[0]} rows, {combined.shape[1]} columns")

    return combined


# ══════════════════════════════════════════════════════════════════════
#  Data Cleaning
# ══════════════════════════════════════════════════════════════════════

def _encode_alarm_columns(df: pd.DataFrame, alarm_cols: List[str]) -> pd.DataFrame:
    """Encode alarm columns: Active=1, Inactive=0, Bad Input=NaN.

    Handles case-insensitive matching and strips whitespace.
    """
    for col in alarm_cols:
        if col not in df.columns:
            continue
        # Normalize strings
        series = df[col].astype(str).str.strip().str.lower()
        # Map to numeric
        mapping = {"active": 1.0, "inactive": 0.0}
        df[col] = series.map(mapping)  # unmapped values (Bad Input, nan) -> NaN
    return df


def _replace_bad_input(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 'Bad Input' string values with NaN across all columns."""
    # Vectorized replacement: any cell containing 'Bad Input' -> NaN
    df = df.replace(r"(?i)^bad\s*input$", np.nan, regex=True)
    return df


def clean_dataframe(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Clean raw SWaT data through a multi-step pipeline.

    Steps:
        1. Store and drop timestamp column
        2. Encode alarm columns (Active=1, Inactive=0)
        3. Replace all 'Bad Input' values with NaN
        4. Convert all remaining columns to numeric (coerce errors to NaN)
        5. Handle NaN: forward fill -> backward fill -> drop rows with remaining NaN
        6. Remove constant columns (std == 0)
        7. Drop columns with too many NaN values (< min_non_null_ratio)

    Args:
        df: Raw DataFrame from load_swat_data().
        config: Optional config dict with preprocessing parameters.

    Returns:
        Cleaned DataFrame with all numeric values, no NaN, no constant columns.
    """
    if config is None:
        config = DEFAULT_CONFIG
    preproc = config.get("preprocessing", DEFAULT_CONFIG["preprocessing"])

    df = df.copy()

    # --- Step 1: Drop timestamp column ---
    ts_col = None
    for candidate in ["t_stamp", "Timestamp", "timestamp"]:
        if candidate in df.columns:
            ts_col = candidate
            break

    if ts_col is not None:
        logger.info(f"Dropping timestamp column: '{ts_col}'")
        df.drop(columns=[ts_col], inplace=True)

    # --- Step 2: Encode alarm columns before general numeric conversion ---
    alarm_cols = [c for c in df.columns if c.endswith(".Alarm")]
    if alarm_cols:
        logger.info(f"Encoding {len(alarm_cols)} alarm columns (Active=1, Inactive=0)")
        df = _encode_alarm_columns(df, alarm_cols)

    # --- Step 3: Replace 'Bad Input' with NaN ---
    logger.info("Replacing 'Bad Input' values with NaN")
    df = _replace_bad_input(df)

    # --- Step 4: Convert all columns to numeric ---
    logger.info("Converting all columns to numeric (coercing errors to NaN)")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Step 5: Drop columns with too many NaN values ---
    min_ratio = preproc.get("min_non_null_ratio", 0.5)
    non_null_ratios = df.notna().mean()
    sparse_cols = non_null_ratios[non_null_ratios < min_ratio].index.tolist()
    if sparse_cols:
        logger.warning(
            f"Dropping {len(sparse_cols)} columns with < {min_ratio*100:.0f}% "
            f"non-null values: {sparse_cols}"
        )
        df.drop(columns=sparse_cols, inplace=True)

    # --- Step 6: Handle remaining NaN values ---
    nan_before = df.isna().sum().sum()
    if nan_before > 0:
        logger.info(f"Handling {nan_before} NaN values with forward/backward fill")

        strategy = preproc.get("bad_input_strategy", "ffill")
        if strategy == "interpolate":
            df = df.interpolate(method="linear", limit_direction="both")
        else:
            # Forward fill, then backward fill
            df = df.ffill()
            df = df.bfill()

        # If any NaN still remain (e.g., entire column was NaN), drop those rows
        nan_after = df.isna().sum().sum()
        if nan_after > 0:
            logger.warning(
                f"Dropping {df.isna().any(axis=1).sum()} rows with remaining NaN values"
            )
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)

    # --- Step 7: Remove constant columns (std == 0) ---
    if preproc.get("remove_constant_cols", True):
        stds = df.std()
        constant_cols = stds[stds == 0].index.tolist()
        if constant_cols:
            logger.info(
                f"Removing {len(constant_cols)} constant columns: {constant_cols}"
            )
            df.drop(columns=constant_cols, inplace=True)

    logger.info(
        f"Cleaning complete: {df.shape[0]} rows, {df.shape[1]} columns, "
        f"dtypes: {df.dtypes.value_counts().to_dict()}"
    )

    return df


# ══════════════════════════════════════════════════════════════════════
#  Normalization / Scaling
# ══════════════════════════════════════════════════════════════════════

def fit_scaler(
    train_data: np.ndarray,
    method: str = "standard",
) -> Union[StandardScaler, MinMaxScaler]:
    """Fit a scaler on training data ONLY (prevents data leakage).

    Args:
        train_data: 2D array of shape (n_samples, n_features).
        method: 'standard' for StandardScaler, 'minmax' for MinMaxScaler.

    Returns:
        Fitted scaler object.
    """
    if method == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler method: {method}. Use 'standard' or 'minmax'.")

    scaler.fit(train_data)
    logger.info(
        f"Fitted {method} scaler on training data: "
        f"{train_data.shape[0]} samples, {train_data.shape[1]} features"
    )
    return scaler


def apply_scaler(data: np.ndarray, scaler) -> np.ndarray:
    """Apply a fitted scaler to data.

    Args:
        data: 2D array of shape (n_samples, n_features).
        scaler: Fitted StandardScaler or MinMaxScaler.

    Returns:
        Scaled data with same shape.
    """
    return scaler.transform(data).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
#  Sliding Windows
# ══════════════════════════════════════════════════════════════════════

def create_sliding_windows(
    data: np.ndarray,
    window_size: int = 60,
    stride: int = 1,
) -> np.ndarray:
    """Create sliding windows from a 2D time-series array.

    Uses np.lib.stride_tricks for memory-efficient windowing (creates
    a view when possible, copies only when necessary).

    Args:
        data: 2D array of shape (n_timesteps, n_features).
        window_size: Number of timesteps per window.
        stride: Step size between consecutive windows.

    Returns:
        3D array of shape (n_windows, window_size, n_features).

    Raises:
        ValueError: If data has fewer rows than window_size.
    """
    n_timesteps, n_features = data.shape

    if n_timesteps < window_size:
        raise ValueError(
            f"Data has {n_timesteps} timesteps but window_size is {window_size}. "
            f"Need at least {window_size} timesteps."
        )

    # Number of windows
    n_windows = (n_timesteps - window_size) // stride + 1

    # Use stride_tricks for efficient windowing
    # Shape: (n_windows, window_size, n_features)
    # Strides: (stride * row_stride, row_stride, feature_stride)
    row_stride = data.strides[0]
    feat_stride = data.strides[1]

    windows = np.lib.stride_tricks.as_strided(
        data,
        shape=(n_windows, window_size, n_features),
        strides=(stride * row_stride, row_stride, feat_stride),
    )

    # Copy to own memory to avoid memory issues with stride tricks
    windows = np.array(windows, dtype=np.float32)

    return windows


# ══════════════════════════════════════════════════════════════════════
#  Graph Construction
# ══════════════════════════════════════════════════════════════════════

def build_sensor_graph(
    data: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a sensor correlation graph from training data.

    Computes the absolute Pearson correlation matrix between all sensor
    features and creates edges between sensors whose |correlation| >= threshold.

    Self-loops are excluded. The graph is undirected (edges in both directions).

    Args:
        data: 2D array of shape (n_samples, n_features) — training data.
        feature_names: List of feature/column names, length = n_features.
        threshold: Minimum |correlation| to create an edge.

    Returns:
        edge_index: (2, num_edges) int64 array — COO format node pairs.
        edge_weights: (num_edges,) float32 array — absolute correlation values.
    """
    n_features = data.shape[1]
    assert len(feature_names) == n_features, (
        f"feature_names length ({len(feature_names)}) != n_features ({n_features})"
    )

    # Compute Pearson correlation matrix
    # Use pandas for robust handling of constant columns (returns NaN for those)
    df = pd.DataFrame(data, columns=feature_names)
    corr_matrix = df.corr(method="pearson").values

    # Replace NaN correlations (from constant columns) with 0
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Take absolute values
    abs_corr = np.abs(corr_matrix)

    # Zero out diagonal (no self-loops)
    np.fill_diagonal(abs_corr, 0.0)

    # Find edges where |correlation| >= threshold
    source, target = np.where(abs_corr >= threshold)

    # Edge weights = absolute correlation values
    weights = abs_corr[source, target].astype(np.float32)

    edge_index = np.stack([source, target], axis=0).astype(np.int64)

    logger.info(
        f"Sensor graph: {n_features} nodes, {edge_index.shape[1]} edges "
        f"(threshold={threshold:.2f}), "
        f"avg edge weight={weights.mean():.4f}" if len(weights) > 0 else
        f"Sensor graph: {n_features} nodes, 0 edges (threshold={threshold:.2f})"
    )

    # Log most connected sensors
    if edge_index.shape[1] > 0:
        degree = np.bincount(edge_index[0], minlength=n_features)
        top_k = min(5, n_features)
        top_indices = np.argsort(degree)[-top_k:][::-1]
        top_sensors = [(feature_names[i], int(degree[i])) for i in top_indices]
        logger.info(f"Most connected sensors: {top_sensors}")

    return edge_index, weights


def compute_node_features(windows: np.ndarray) -> np.ndarray:
    """Compute statistical node features per sensor per window.

    For each window and each sensor (feature), computes 5 summary statistics
    across the time dimension:
        [mean, std, min, max, range]

    Args:
        windows: 3D array of shape (num_windows, window_size, num_sensors).

    Returns:
        4D-reshaped to 3D array of shape (num_windows, num_sensors, 5).
        Feature order: [mean, std, min, max, range].
    """
    # windows shape: (N, T, F)  where N=num_windows, T=window_size, F=num_sensors
    # Compute statistics along axis=1 (time dimension)
    w_mean = np.mean(windows, axis=1)   # (N, F)
    w_std = np.std(windows, axis=1)     # (N, F)
    w_min = np.min(windows, axis=1)     # (N, F)
    w_max = np.max(windows, axis=1)     # (N, F)
    w_range = w_max - w_min             # (N, F)

    # Stack to (N, F, 5)
    node_features = np.stack([w_mean, w_std, w_min, w_max, w_range], axis=-1)

    return node_features.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
#  Dataset Statistics (Privacy-Safe)
# ══════════════════════════════════════════════════════════════════════

def save_dataset_stats(
    data: pd.DataFrame,
    output_path: str = "data/proof/dataset_stats.json",
    feature_names: Optional[List[str]] = None,
    scaler=None,
) -> dict:
    """Save aggregated dataset statistics without any raw data values.

    This file is safe to commit to version control — it contains only
    shapes, column names, and aggregated statistics (means, stds, etc.).

    Args:
        data: Cleaned DataFrame (post-cleaning, pre-scaling).
        output_path: Path to save the JSON stats file.
        feature_names: List of feature column names.
        scaler: Fitted scaler object (to extract mean/std or min/max).

    Returns:
        Dictionary of saved statistics.
    """
    stats = {
        "dataset": "SWaT A9 (Clean)",
        "total_samples": int(data.shape[0]),
        "total_features": int(data.shape[1]),
        "feature_names": feature_names or list(data.columns),
        "column_dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
        "per_feature_stats": {},
    }

    # Per-feature aggregated stats (no raw values!)
    for col in data.columns:
        col_data = data[col]
        stats["per_feature_stats"][col] = {
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "median": float(col_data.median()),
            "nunique": int(col_data.nunique()),
            "is_binary": bool(col_data.nunique() <= 3),
        }

    # Scaler parameters
    if scaler is not None:
        if isinstance(scaler, StandardScaler):
            stats["scaler"] = {
                "type": "standard",
                "means": scaler.mean_.tolist(),
                "stds": scaler.scale_.tolist(),
            }
        elif isinstance(scaler, MinMaxScaler):
            stats["scaler"] = {
                "type": "minmax",
                "data_min": scaler.data_min_.tolist(),
                "data_max": scaler.data_max_.tolist(),
                "feature_range": list(scaler.feature_range),
            }

    # Save to file
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Dataset stats saved to {out_path} ({out_path.stat().st_size} bytes)")

    return stats


# ══════════════════════════════════════════════════════════════════════
#  Save / Load Processed Data
# ══════════════════════════════════════════════════════════════════════

def save_processed_data(
    results: Dict[str, object],
    output_dir: str = "data/proof/",
) -> str:
    """Save all processed tensors and metadata to disk.

    Saves:
      - X_train.npy, X_val.npy, X_test.npy            (sliding windows)
      - node_features_train.npy, ..._val.npy, ..._test.npy
      - edge_index.npy, edge_weights.npy               (graph)
      - feature_names.json, column_categories.json      (metadata)
      - scaler.pkl                                      (fitted scaler)

    All files are derived from NDA-protected raw data, so they should
    remain gitignored. Only dataset_stats.json is safe to commit.

    Args:
        results: Dict returned by SWaTPreprocessor.run().
        output_dir: Directory to save processed files.

    Returns:
        Path to the output directory.
    """
    import pickle

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Numpy arrays ──────────────────────────────────────────────────
    for key in ["X_train", "X_val", "X_test",
                "node_features_train", "node_features_val", "node_features_test",
                "edge_index", "edge_weights"]:
        if key in results and results[key] is not None:
            fpath = out / f"{key}.npy"
            np.save(fpath, results[key])
            logger.info(f"Saved {key}: {results[key].shape} → {fpath}")

    # ── Scaler ────────────────────────────────────────────────────────
    if "scaler" in results and results["scaler"] is not None:
        scaler_path = out / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(results["scaler"], f)
        logger.info(f"Saved scaler → {scaler_path}")

    # ── Metadata (JSON-safe) ──────────────────────────────────────────
    if "feature_names" in results:
        with open(out / "feature_names.json", "w") as f:
            json.dump(results["feature_names"], f, indent=2)
        logger.info(f"Saved feature_names ({len(results['feature_names'])} features)")

    if "column_categories" in results:
        with open(out / "column_categories.json", "w") as f:
            json.dump(results["column_categories"], f, indent=2)
        logger.info(f"Saved column_categories")

    logger.info(f"All processed data saved to {out}/")
    return str(out)


def load_processed_data(
    data_dir: str = "data/proof/",
) -> Dict[str, object]:
    """Load previously saved processed data from disk.

    This allows downstream modules (LSTM-AE, GAT, baselines) to
    skip the full preprocessing pipeline and load ready-to-use tensors.

    Args:
        data_dir: Directory containing saved .npy files.

    Returns:
        Dict with same keys as SWaTPreprocessor.run() output.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    import pickle

    d = Path(data_dir)
    if not d.exists():
        raise FileNotFoundError(
            f"Processed data directory not found: {d}. "
            f"Run preprocessing first: python -m src.preprocess --save"
        )

    results = {}

    # ── Numpy arrays ──────────────────────────────────────────────────
    for key in ["X_train", "X_val", "X_test",
                "node_features_train", "node_features_val", "node_features_test",
                "edge_index", "edge_weights"]:
        fpath = d / f"{key}.npy"
        if fpath.exists():
            results[key] = np.load(fpath)
            logger.info(f"Loaded {key}: {results[key].shape}")
        else:
            logger.warning(f"Missing {fpath}")

    # ── Scaler ────────────────────────────────────────────────────────
    scaler_path = d / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            results["scaler"] = pickle.load(f)
        logger.info(f"Loaded scaler: {type(results['scaler']).__name__}")

    # ── Metadata ──────────────────────────────────────────────────────
    fn_path = d / "feature_names.json"
    if fn_path.exists():
        with open(fn_path, "r") as f:
            results["feature_names"] = json.load(f)
        logger.info(f"Loaded feature_names ({len(results['feature_names'])})")

    cc_path = d / "column_categories.json"
    if cc_path.exists():
        with open(cc_path, "r") as f:
            results["column_categories"] = json.load(f)

    return results


# ══════════════════════════════════════════════════════════════════════
#  Main Pipeline Class
# ══════════════════════════════════════════════════════════════════════

class SWaTPreprocessor:
    """Complete preprocessing pipeline for SWaT dataset.

    Orchestrates all preprocessing steps:
        1. Load raw CSV files
        2. Classify columns by type
        3. Clean data (Bad Input, NaN, constant columns)
        4. Temporal train/val/test split
        5. Fit scaler on training data only
        6. Create sliding windows
        7. Build sensor correlation graph
        8. Compute node features
        9. Save dataset statistics

    Usage:
        preprocessor = SWaTPreprocessor("configs/default.yaml")
        results = preprocessor.run()
        X_train = results['X_train']  # (N, window_size, features)
    """

    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = self._load_config(config_path)
        self.scaler = None
        self.feature_names: Optional[List[str]] = None
        self.column_categories: Optional[Dict[str, List[str]]] = None
        self.edge_index: Optional[np.ndarray] = None
        self.edge_weights: Optional[np.ndarray] = None
        self.timestamps: Optional[pd.Series] = None

    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML file, falling back to defaults.

        Args:
            path: Path to YAML config file.

        Returns:
            Merged configuration dictionary.
        """
        config = DEFAULT_CONFIG.copy()
        config_path = Path(path)

        if config_path.exists():
            logger.info(f"Loading config from {config_path}")
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config:
                config = self._deep_merge(config, yaml_config)
        else:
            logger.info(
                f"Config file not found at {config_path}, using defaults. "
                f"This is fine for initial setup."
            )

        return config

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Deep merge override into base dict (override wins on conflicts).

        Args:
            base: Base configuration dictionary.
            override: Override dictionary (values take precedence).

        Returns:
            Merged dictionary.
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = SWaTPreprocessor._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def run(self, save_stats: bool = True) -> Dict[str, object]:
        """Execute the full preprocessing pipeline.

        Args:
            save_stats: Whether to save dataset statistics to JSON.

        Returns:
            Dictionary with keys:
                'X_train', 'X_val', 'X_test' — (N, window_size, features) float32 arrays
                'node_features_train', 'node_features_val', 'node_features_test'
                    — (N, sensors, 5) float32 arrays
                'edge_index' — (2, num_edges) int64 array
                'edge_weights' — (num_edges,) float32 array
                'feature_names' — list of feature column names
                'scaler' — fitted scaler object
                'column_categories' — dict of column type -> column names
        """
        logger.info("=" * 70)
        logger.info("Starting SWaT preprocessing pipeline...")
        logger.info("=" * 70)

        # ── 1. Load data ──────────────────────────────────────────────
        all_files = (
            self.config["data"]["train_files"] +
            self.config["data"]["test_files"]
        )
        df = load_swat_data(
            self.config["data"]["raw_dir"],
            all_files,
            self.config,
        )
        logger.info(f"Loaded data: {df.shape}")

        # ── 2. Classify columns ───────────────────────────────────────
        self.column_categories = classify_columns(df)

        # ── 3. Save timestamps before cleaning ────────────────────────
        for ts_candidate in ["t_stamp", "Timestamp", "timestamp"]:
            if ts_candidate in df.columns:
                self.timestamps = df[ts_candidate].copy()
                break

        # ── 4. Clean ──────────────────────────────────────────────────
        df_clean = clean_dataframe(df, self.config)
        self.feature_names = list(df_clean.columns)
        logger.info(
            f"After cleaning: {df_clean.shape}, "
            f"features: {len(self.feature_names)}"
        )

        # ── 5. Temporal split (NO shuffle for time series!) ───────────
        n = len(df_clean)
        train_ratio = self.config["preprocessing"]["train_ratio"]
        val_ratio = self.config["preprocessing"]["val_ratio"]

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_data = df_clean.iloc[:train_end].values.astype(np.float64)
        val_data = df_clean.iloc[train_end:val_end].values.astype(np.float64)
        test_data = df_clean.iloc[val_end:].values.astype(np.float64)

        logger.info(
            f"Temporal split: "
            f"Train={train_data.shape[0]}, "
            f"Val={val_data.shape[0]}, "
            f"Test={test_data.shape[0]}"
        )

        # ── 6. Fit scaler on training data ONLY ──────────────────────
        self.scaler = fit_scaler(
            train_data,
            self.config["preprocessing"]["scaler"],
        )
        train_scaled = apply_scaler(train_data, self.scaler)
        val_scaled = apply_scaler(val_data, self.scaler)
        test_scaled = apply_scaler(test_data, self.scaler)

        # ── 7. Create sliding windows ─────────────────────────────────
        ws = self.config["preprocessing"]["window_size"]
        stride = self.config["preprocessing"]["stride"]

        X_train = create_sliding_windows(train_scaled, ws, stride)
        X_val = create_sliding_windows(val_scaled, ws, stride)
        X_test = create_sliding_windows(test_scaled, ws, stride)

        logger.info(
            f"Windows — Train: {X_train.shape}, "
            f"Val: {X_val.shape}, Test: {X_test.shape}"
        )

        # ── 8. Build sensor graph from training data ──────────────────
        self.edge_index, self.edge_weights = build_sensor_graph(
            train_scaled,
            self.feature_names,
            self.config["preprocessing"]["correlation_threshold"],
        )
        logger.info(f"Graph: {self.edge_index.shape[1]} edges")

        # ── 9. Compute node features ─────────────────────────────────
        nf_train = compute_node_features(X_train)
        nf_val = compute_node_features(X_val)
        nf_test = compute_node_features(X_test)

        logger.info(
            f"Node features — Train: {nf_train.shape}, "
            f"Val: {nf_val.shape}, Test: {nf_test.shape}"
        )

        # ── 10. Save stats ────────────────────────────────────────────
        if save_stats:
            proof_dir = self.config["data"]["proof_dir"]
            stats_path = os.path.join(proof_dir, "dataset_stats.json")
            save_dataset_stats(
                df_clean, stats_path,
                self.feature_names, self.scaler,
            )

        logger.info("=" * 70)
        logger.info("SWaT preprocessing pipeline complete!")
        logger.info("=" * 70)

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "node_features_train": nf_train,
            "node_features_val": nf_val,
            "node_features_test": nf_test,
            "edge_index": self.edge_index,
            "edge_weights": self.edge_weights,
            "feature_names": self.feature_names,
            "scaler": self.scaler,
            "column_categories": self.column_categories,
        }


# ══════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser(
        description="RAKSHAK-ICS: SWaT Data Preprocessing Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory from config",
    )
    parser.add_argument(
        "--no-stats", action="store_true",
        help="Skip saving dataset statistics",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save processed tensors (.npy), scaler (.pkl), and metadata to data/proof/",
    )
    args = parser.parse_args()

    preprocessor = SWaTPreprocessor(args.config)

    if args.data_dir:
        preprocessor.config["data"]["raw_dir"] = args.data_dir

    results = preprocessor.run(save_stats=not args.no_stats)

    # Save processed data to disk if requested
    if args.save:
        proof_dir = preprocessor.config["data"].get("proof_dir", "data/proof/")
        save_processed_data(results, proof_dir)

    print(f"\n{'='*50}")
    print(f"Preprocessing complete!")
    print(f"{'='*50}")
    print(f"  Train windows:     {results['X_train'].shape}")
    print(f"  Val windows:       {results['X_val'].shape}")
    print(f"  Test windows:      {results['X_test'].shape}")
    print(f"  Node features:     {results['node_features_train'].shape}")
    print(f"  Graph edges:       {results['edge_index'].shape[1]}")
    print(f"  Features:          {len(results['feature_names'])}")
    print(f"  Feature names:     {results['feature_names'][:5]}...")
    print(f"  Scaler type:       {type(results['scaler']).__name__}")
    if args.save:
        print(f"  Saved to:          {proof_dir}")
    print(f"{'='*50}")
