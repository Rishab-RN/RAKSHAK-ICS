#!/usr/bin/env python3
"""
================================================================================
RAKSHAK-ICS — Complete EDA & Preprocessing Pipeline
================================================================================
Author  : RAKSHAK-ICS Team
Purpose : Exploratory Data Analysis and Preprocessing for SWaT ICS Datasets
Target  : Anomaly Detection (LSTM-AE, Isolation Forest, Random Forest, FDI sim)
================================================================================

Pipeline Tasks:
  1. Dataset Profiling
  2. Feature Categorization
  3. Data Quality Analysis
  4. Data Cleaning
  5. Feature Engineering
  6. Scaling (StandardScaler + MinMaxScaler)
  7. Exploratory Data Analysis (Visualizations)
  8. Anomaly Detection Readiness Assessment
  9. Output Generation (CSVs, reports, plots)
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
import textwrap
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Resolve project root relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "swat")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

DATASET_FILES = [
    os.path.join(DATA_DIR, "dataset1.csv"),
    os.path.join(DATA_DIR, "dataset2.csv"),
    os.path.join(DATA_DIR, "dataset3.csv"),
]

# Plotting aesthetics
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.figsize": (14, 8),
})

# Color palette for consistent visuals
PALETTE = {
    "sensor": "#2196F3",
    "status": "#4CAF50",
    "alarm": "#FF9800",
    "state": "#9C27B0",
    "primary": "#1976D2",
    "danger": "#E53935",
    "success": "#43A047",
    "warning": "#FB8C00",
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
        os.makedirs(d, exist_ok=True)
    print("✅ Output directories verified.")


def print_section(title: str):
    """Print a formatted section header to stdout."""
    width = 72
    print(f"\n{'═' * width}")
    print(f"  {title.upper()}")
    print(f"{'═' * width}")


def fmt_bytes(nbytes: int) -> str:
    """Human-readable byte size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if nbytes < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} TB"


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — DATASET PROFILING
# ─────────────────────────────────────────────────────────────────────────────

def load_datasets() -> dict[str, pd.DataFrame]:
    """
    Load all three SWaT CSV files into a dictionary of DataFrames.
    Uses low_memory=False to avoid mixed-type inference warnings.
    """
    print_section("Task 1 — Loading Datasets")
    datasets = {}
    for fpath in DATASET_FILES:
        name = os.path.splitext(os.path.basename(fpath))[0]
        print(f"  Loading {name}...", end=" ")
        df = pd.read_csv(fpath, low_memory=False)
        datasets[name] = df
        print(f"✅ ({df.shape[0]:,} rows × {df.shape[1]} cols)")
    return datasets


def profile_dataset(df: pd.DataFrame, name: str) -> dict:
    """
    Generate a comprehensive profile for a single dataset.

    Returns a dictionary with shape, dtypes, memory usage,
    duplicate count, and unique-value summaries for categorical columns.
    """
    mem_usage = df.memory_usage(deep=True).sum()
    duplicates = df.duplicated().sum()

    # Identify categorical columns (object dtype)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_uniques = {}
    for col in cat_cols:
        unique_vals = df[col].unique()
        cat_uniques[col] = {
            "n_unique": len(unique_vals),
            "values": unique_vals[:15].tolist(),  # Show first 15
        }

    profile = {
        "name": name,
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.value_counts().to_dict(),
        "memory_usage": mem_usage,
        "memory_human": fmt_bytes(mem_usage),
        "duplicates": duplicates,
        "categorical_uniques": cat_uniques,
    }

    print(f"\n  📊 Profile: {name}")
    print(f"     Shape           : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"     Memory          : {fmt_bytes(mem_usage)}")
    print(f"     Duplicate rows  : {duplicates}")
    print(f"     Data types      : {dict(df.dtypes.value_counts())}")
    print(f"     Object columns  : {len(cat_cols)}")

    return profile


def profile_all_datasets(datasets: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Profile every dataset and return a dict of profiles."""
    profiles = {}
    for name, df in datasets.items():
        profiles[name] = profile_dataset(df, name)
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — FEATURE CATEGORIZATION
# ─────────────────────────────────────────────────────────────────────────────

def categorize_features(columns: list[str]) -> dict[str, list[str]]:
    """
    Automatically categorize columns based on naming conventions:
      - .Pv        → Sensor (process variable)
      - .Status    → Equipment status
      - .Alarm     → Alarm indicator
      - _STATE     → Process state
      - t_stamp    → Timestamp
      - .Speed     → Speed (treated as sensor-like)
      - Other      → Uncategorized
    """
    categories = OrderedDict({
        "sensor_pv": [],
        "speed": [],
        "status": [],
        "alarm": [],
        "process_state": [],
        "timestamp": [],
        "other": [],
    })

    for col in columns:
        if col == "t_stamp":
            categories["timestamp"].append(col)
        elif col.endswith(".Pv"):
            categories["sensor_pv"].append(col)
        elif col.endswith(".Speed"):
            categories["speed"].append(col)
        elif col.endswith(".Status"):
            categories["status"].append(col)
        elif col.endswith(".Alarm"):
            categories["alarm"].append(col)
        elif col.endswith("_STATE"):
            categories["process_state"].append(col)
        else:
            categories["other"].append(col)

    print_section("Task 2 — Feature Categorization")
    for cat, cols in categories.items():
        if cols:
            print(f"  {cat:20s} : {len(cols):3d} features")
            # Show the feature names (truncated if too many)
            display = cols[:10]
            print(f"    → {display}")
            if len(cols) > 10:
                print(f"      ... and {len(cols) - 10} more")

    return categories


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — DATA QUALITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_data_quality(df: pd.DataFrame, name: str) -> dict:
    """
    Perform thorough data quality analysis on a DataFrame:
      - Detect columns with mixed datatypes
      - Identify invalid / non-numeric values in expected-numeric columns
      - Find columns containing "Bad Input"
      - Report missing values (NaN)
      - Report constant columns (zero variance)

    Returns a quality report dictionary.
    """
    print_section(f"Task 3 — Data Quality Analysis [{name}]")

    quality = {
        "mixed_type_columns": [],
        "bad_input_columns": [],
        "invalid_value_columns": {},
        "constant_columns": [],
        "missing_values": {},
    }

    # ── Mixed datatype detection ──
    # A column has mixed types if it contains both numeric-parseable and
    # non-numeric string values (excluding NaN).
    for col in df.columns:
        if df[col].dtype == object:
            # Try converting to numeric; failures indicate mixed types
            converted = pd.to_numeric(df[col], errors="coerce")
            n_original_non_null = df[col].notna().sum()
            n_numeric = converted.notna().sum()
            if 0 < n_numeric < n_original_non_null:
                quality["mixed_type_columns"].append(col)

    # ── "Bad Input" detection ──
    for col in df.columns:
        if df[col].dtype == object:
            bad_mask = df[col].astype(str).str.strip().str.lower() == "bad input"
            if bad_mask.any():
                quality["bad_input_columns"].append({
                    "column": col,
                    "count": int(bad_mask.sum()),
                    "percentage": round(bad_mask.sum() / len(df) * 100, 2),
                })

    # ── Invalid values in object columns ──
    for col in df.columns:
        if df[col].dtype == object:
            unique_vals = df[col].dropna().unique()
            non_numeric = []
            for v in unique_vals:
                try:
                    float(v)
                except (ValueError, TypeError):
                    non_numeric.append(str(v))
            if non_numeric:
                quality["invalid_value_columns"][col] = non_numeric

    # ── Missing values ──
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        quality["missing_values"] = missing.to_dict()

    # ── Constant columns (zero variance) ──
    for col in df.columns:
        if df[col].dtype != object and col != "t_stamp":
            if df[col].nunique() <= 1:
                quality["constant_columns"].append(col)

    # ── Print summary ──
    print(f"  Mixed-type columns    : {len(quality['mixed_type_columns'])}")
    if quality["mixed_type_columns"]:
        print(f"    → {quality['mixed_type_columns']}")

    print(f"  'Bad Input' columns   : {len(quality['bad_input_columns'])}")
    for item in quality["bad_input_columns"]:
        print(f"    → {item['column']}: {item['count']} occurrences ({item['percentage']}%)")

    print(f"  Invalid-value columns : {len(quality['invalid_value_columns'])}")
    for col, vals in quality["invalid_value_columns"].items():
        print(f"    → {col}: {vals}")

    print(f"  Missing value columns : {len(quality['missing_values'])}")
    print(f"  Constant columns      : {len(quality['constant_columns'])}")
    if quality["constant_columns"]:
        print(f"    → {quality['constant_columns']}")

    return quality


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 — DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_alarm_column(series: pd.Series) -> pd.Series:
    """
    Convert alarm columns to numeric:
      - 'Active'    → 1  (alarm is triggered)
      - 'Inactive'  → 0  (alarm is not triggered)
      - 'Bad Input' → -1 (sensor malfunction / invalid reading)
      - Numeric     → keep as-is (already numeric representation)
      - NaN         → 0  (default to inactive)

    Rationale: Preserves the three-state information (active/inactive/faulty)
    which is operationally meaningful for anomaly detection.
    """
    mapping = {
        "Active": 1,
        "active": 1,
        "Inactive": 0,
        "inactive": 0,
        "Bad Input": -1,
        "bad input": -1,
        "Bad input": -1,
    }
    return series.map(lambda x: mapping.get(str(x).strip(), x)).pipe(
        pd.to_numeric, errors="coerce"
    ).fillna(0).astype(int)


def clean_status_column(series: pd.Series) -> pd.Series:
    """
    Convert status columns to numeric:
      - Numeric values (1, 2, etc.) → keep as-is
      - Non-numeric strings         → coerce to NaN, then fill with 0

    Status columns typically contain integer codes (1=ON, 2=OFF, etc.)
    but may occasionally have string artifacts.
    """
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def clean_sensor_column(series: pd.Series) -> pd.Series:
    """
    Convert sensor (.Pv) and speed columns to float:
      - 'Bad Input' → NaN (then forward-fill + backfill)
      - Other non-numeric → NaN (then fill)

    Forward-fill is used because sensor readings are time-series and the
    most recent valid reading is the best estimate for a momentary dropout.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    # Forward-fill, then backfill for any leading NaNs
    numeric = numeric.ffill().bfill()
    # If still NaN (entire column was bad), fill with 0
    numeric = numeric.fillna(0.0)
    return numeric


def clean_dataset(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """
    Master cleaning function that applies appropriate cleaning
    strategy to each feature category.

    Steps:
      1. Parse timestamp column
      2. Clean sensor (.Pv) columns → float64
      3. Clean speed columns → float64
      4. Clean status (.Status) columns → int
      5. Clean alarm (.Alarm) columns → int (Active=1, Inactive=0, Bad=-1)
      6. Clean process state (_STATE) columns → int

    Returns a fully numeric DataFrame (except t_stamp which becomes datetime).
    """
    print_section("Task 4 — Data Cleaning")
    cleaned = df.copy()

    # ── 1. Timestamp ──
    print("  [1/6] Parsing timestamps...")
    if "t_stamp" in cleaned.columns:
        cleaned["t_stamp"] = pd.to_datetime(cleaned["t_stamp"], dayfirst=True, errors="coerce")
        n_parsed = cleaned["t_stamp"].notna().sum()
        print(f"        Parsed {n_parsed:,}/{len(cleaned):,} timestamps successfully")

    # ── 2. Sensor columns (.Pv) ──
    sensor_cols = categories.get("sensor_pv", [])
    print(f"  [2/6] Cleaning {len(sensor_cols)} sensor columns...")
    for col in sensor_cols:
        if col in cleaned.columns:
            n_bad = (cleaned[col].astype(str).str.strip().str.lower() == "bad input").sum()
            cleaned[col] = clean_sensor_column(cleaned[col])
            if n_bad > 0:
                print(f"        {col}: replaced {n_bad} 'Bad Input' values via forward-fill")

    # ── 3. Speed columns ──
    speed_cols = categories.get("speed", [])
    print(f"  [3/6] Cleaning {len(speed_cols)} speed columns...")
    for col in speed_cols:
        if col in cleaned.columns:
            cleaned[col] = clean_sensor_column(cleaned[col])

    # ── 4. Status columns (.Status) ──
    status_cols = categories.get("status", [])
    print(f"  [4/6] Cleaning {len(status_cols)} status columns...")
    for col in status_cols:
        if col in cleaned.columns:
            cleaned[col] = clean_status_column(cleaned[col])

    # ── 5. Alarm columns (.Alarm) ──
    alarm_cols = categories.get("alarm", [])
    print(f"  [5/6] Cleaning {len(alarm_cols)} alarm columns...")
    for col in alarm_cols:
        if col in cleaned.columns:
            cleaned[col] = clean_alarm_column(cleaned[col])

    # ── 6. Process state columns (_STATE) ──
    state_cols = categories.get("process_state", [])
    print(f"  [6/6] Cleaning {len(state_cols)} process state columns...")
    for col in state_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(0).astype(int)

    # ── Verify all numeric (except timestamp) ──
    non_ts_cols = [c for c in cleaned.columns if c != "t_stamp"]
    numeric_check = cleaned[non_ts_cols].apply(pd.to_numeric, errors="coerce")
    remaining_nan = numeric_check.isnull().sum().sum()
    print(f"\n  ✅ Cleaning complete.")
    print(f"     Non-numeric residuals after cleaning: {remaining_nan}")
    print(f"     Final dtypes: {dict(cleaned.dtypes.value_counts())}")

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_feature_sets(
    df: pd.DataFrame, categories: dict[str, list[str]]
) -> dict[str, pd.DataFrame]:
    """
    Create multiple feature subsets for different modelling needs:

      1. sensor_only      — Only .Pv and .Speed columns (pure numeric sensors)
      2. sensor_status    — Sensors + Status variables
      3. full_processed   — All features (sensors + status + alarms + states)

    All subsets exclude the timestamp column (kept separately).
    """
    print_section("Task 5 — Feature Engineering")

    sensor_cols = categories.get("sensor_pv", []) + categories.get("speed", [])
    status_cols = categories.get("status", [])
    alarm_cols = categories.get("alarm", [])
    state_cols = categories.get("process_state", [])

    # Filter to only columns that exist in df
    sensor_cols = [c for c in sensor_cols if c in df.columns]
    status_cols = [c for c in status_cols if c in df.columns]
    alarm_cols = [c for c in alarm_cols if c in df.columns]
    state_cols = [c for c in state_cols if c in df.columns]

    feature_sets = {
        "sensor_only": df[sensor_cols].copy(),
        "sensor_status": df[sensor_cols + status_cols].copy(),
        "full_processed": df[sensor_cols + status_cols + alarm_cols + state_cols].copy(),
    }

    for name, fset in feature_sets.items():
        print(f"  {name:20s} : {fset.shape[1]:3d} features × {fset.shape[0]:,} rows")

    return feature_sets


# ─────────────────────────────────────────────────────────────────────────────
# TASK 6 — SCALING
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(
    feature_sets: dict[str, pd.DataFrame]
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Apply both StandardScaler and MinMaxScaler to each feature set.

    StandardScaler : zero-mean, unit-variance — good for LSTM-AE, distance-based models.
    MinMaxScaler   : [0, 1] range — good for neural networks, bounded activations.

    Returns nested dict: {feature_set_name: {"standard": df, "minmax": df}}
    """
    print_section("Task 6 — Scaling")

    scaled = {}
    for name, df in feature_sets.items():
        # StandardScaler
        ss = StandardScaler()
        std_data = ss.fit_transform(df.values)
        df_std = pd.DataFrame(std_data, columns=df.columns, index=df.index)

        # MinMaxScaler
        mms = MinMaxScaler()
        mm_data = mms.fit_transform(df.values)
        df_mm = pd.DataFrame(mm_data, columns=df.columns, index=df.index)

        scaled[name] = {"standard": df_std, "minmax": df_mm}

        print(f"  {name}:")
        print(f"    StandardScaler → mean≈{std_data.mean():.4f}, std≈{std_data.std():.4f}")
        print(f"    MinMaxScaler   → min={mm_data.min():.4f}, max={mm_data.max():.4f}")

    return scaled


# ─────────────────────────────────────────────────────────────────────────────
# TASK 7 — EXPLORATORY DATA ANALYSIS (VISUALIZATIONS)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_distributions(df: pd.DataFrame, categories: dict, save_dir: str):
    """
    Plot histograms / KDE of sensor feature distributions.
    Generates a grid of subplots for all sensor (.Pv) columns.
    """
    print("  [7.1] Plotting sensor feature distributions...")
    sensor_cols = categories.get("sensor_pv", [])
    sensor_cols = [c for c in sensor_cols if c in df.columns]

    if not sensor_cols:
        print("        No sensor columns found, skipping.")
        return

    n_cols = 4
    n_rows = (len(sensor_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for idx, col in enumerate(sensor_cols):
        ax = axes[idx]
        data = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(data, bins=50, color=PALETTE["sensor"], alpha=0.7, edgecolor="white")
        ax.set_title(col.replace(".Pv", ""), fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.set_ylabel("Count", fontsize=8)

    # Hide unused subplots
    for idx in range(len(sensor_cols), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Sensor Feature Distributions (.Pv)", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sensor_distributions.png"))
    plt.close(fig)
    print("        ✅ Saved sensor_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame, categories: dict, save_dir: str) -> pd.DataFrame:
    """
    Compute and plot correlation matrix for sensor features.
    Returns the correlation DataFrame for later saving.
    """
    print("  [7.2] Plotting correlation heatmap...")
    sensor_cols = categories.get("sensor_pv", []) + categories.get("speed", [])
    sensor_cols = [c for c in sensor_cols if c in df.columns]

    numeric_df = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(20, 18))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        square=True, linewidths=0.3,
        cbar_kws={"shrink": 0.6, "label": "Pearson Correlation"},
        ax=ax, vmin=-1, vmax=1,
        xticklabels=[c.replace(".Pv", "").replace(".Speed", " Spd") for c in corr.columns],
        yticklabels=[c.replace(".Pv", "").replace(".Speed", " Spd") for c in corr.columns],
    )
    ax.set_title("Sensor Feature Correlation Matrix", fontsize=16, fontweight="bold", pad=20)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_heatmap.png"))
    plt.close(fig)
    print("        ✅ Saved correlation_heatmap.png")
    return corr


def plot_sensor_trends(df: pd.DataFrame, categories: dict, save_dir: str):
    """
    Plot time-series trends for key sensors (one per process stage P1–P6).
    Uses a representative sensor from each stage for a concise overview.
    """
    print("  [7.3] Plotting sensor trend plots...")

    # Select representative sensors from each process stage
    key_sensors = []
    stage_sensors = {
        "P1": "LIT101.Pv", "P2": "AIT201.Pv", "P3": "LIT301.Pv",
        "P4": "LIT401.Pv", "P5": "PIT501.Pv", "P6": "LIT601.Pv",
    }
    for stage, sensor in stage_sensors.items():
        if sensor in df.columns:
            key_sensors.append((stage, sensor))

    if not key_sensors:
        print("        No key sensors found, skipping.")
        return

    fig, axes = plt.subplots(len(key_sensors), 1, figsize=(18, 3.5 * len(key_sensors)),
                              sharex=True)
    if len(key_sensors) == 1:
        axes = [axes]

    colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#C62828", "#00838F"]

    for idx, (stage, sensor) in enumerate(key_sensors):
        ax = axes[idx]
        data = pd.to_numeric(df[sensor], errors="coerce")
        ax.plot(data.values, color=colors[idx % len(colors)], linewidth=0.5, alpha=0.8)
        ax.set_ylabel(sensor.replace(".Pv", ""), fontsize=10, fontweight="bold")
        ax.set_title(f"Stage {stage} — {sensor}", fontsize=11, loc="left")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample Index", fontsize=11)
    fig.suptitle("Key Sensor Trends Across Process Stages",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sensor_trends.png"))
    plt.close(fig)
    print("        ✅ Saved sensor_trends.png")


def plot_outlier_analysis(df: pd.DataFrame, categories: dict, save_dir: str):
    """
    Box-plot outlier analysis for sensor features.
    Highlights features with significant outlier populations.
    """
    print("  [7.4] Plotting outlier analysis (boxplots)...")
    sensor_cols = categories.get("sensor_pv", [])
    sensor_cols = [c for c in sensor_cols if c in df.columns]

    if not sensor_cols:
        print("        No sensor columns found, skipping.")
        return

    # Standardize for comparable boxplots
    numeric_df = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
    standardized = (numeric_df - numeric_df.mean()) / (numeric_df.std() + 1e-10)

    fig, ax = plt.subplots(figsize=(20, 8))
    bp = ax.boxplot(
        [standardized[col].dropna().values for col in sensor_cols],
        labels=[c.replace(".Pv", "") for c in sensor_cols],
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=2, alpha=0.3, color=PALETTE["danger"]),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE["sensor"])
        patch.set_alpha(0.6)

    ax.set_title("Sensor Feature Outlier Analysis (Standardized)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Standardized Value", fontsize=11)
    ax.axhline(y=3, color=PALETTE["danger"], linestyle="--", alpha=0.5, label="±3σ threshold")
    ax.axhline(y=-3, color=PALETTE["danger"], linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "outlier_boxplots.png"))
    plt.close(fig)
    print("        ✅ Saved outlier_boxplots.png")


def plot_variance_analysis(df: pd.DataFrame, categories: dict, save_dir: str):
    """
    Horizontal bar chart of feature variance (log scale).
    Low-variance features may be candidates for removal.
    High-variance features are prime candidates for anomaly detection.
    """
    print("  [7.5] Plotting variance analysis...")
    sensor_cols = categories.get("sensor_pv", []) + categories.get("speed", [])
    sensor_cols = [c for c in sensor_cols if c in df.columns]

    numeric_df = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
    variances = numeric_df.var().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(8, len(variances) * 0.35)))
    colors = [PALETTE["danger"] if v > variances.quantile(0.9) else
              PALETTE["warning"] if v > variances.quantile(0.5) else
              PALETTE["sensor"] for v in variances.values]

    bars = ax.barh(
        [c.replace(".Pv", "").replace(".Speed", " Spd") for c in variances.index],
        variances.values, color=colors, edgecolor="white", height=0.7
    )
    ax.set_xscale("log")
    ax.set_xlabel("Variance (log scale)", fontsize=11)
    ax.set_title("Feature Variance Analysis", fontsize=14, fontweight="bold")

    # Annotate very low variance
    low_var_threshold = variances.quantile(0.1)
    for i, (idx, val) in enumerate(variances.items()):
        if val <= low_var_threshold:
            ax.annotate("⚠ low", (val, i), fontsize=7, color="red",
                        xytext=(5, 0), textcoords="offset points", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "variance_analysis.png"))
    plt.close(fig)
    print("        ✅ Saved variance_analysis.png")


def plot_alarm_status_distribution(df: pd.DataFrame, categories: dict, save_dir: str):
    """
    Bar chart showing the distribution of alarm and status values.
    Useful for understanding the operational state distribution.
    """
    print("  [7.6] Plotting alarm & status distributions...")

    alarm_cols = categories.get("alarm", [])
    alarm_cols = [c for c in alarm_cols if c in df.columns]

    if not alarm_cols:
        print("        No alarm columns found, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Alarm distribution (before cleaning — use raw counts from original)
    alarm_data = df[alarm_cols].apply(pd.to_numeric, errors="coerce")
    alarm_counts = {}
    for col in alarm_cols:
        vc = alarm_data[col].value_counts()
        for val, count in vc.items():
            label = {1: "Active", 0: "Inactive", -1: "Bad Input"}.get(val, str(val))
            alarm_counts[label] = alarm_counts.get(label, 0) + count

    ax1 = axes[0]
    if alarm_counts:
        labels = list(alarm_counts.keys())
        values = list(alarm_counts.values())
        colors_a = [PALETTE["danger"] if "Active" in l else
                     PALETTE["success"] if "Inactive" in l else
                     PALETTE["warning"] for l in labels]
        ax1.bar(labels, values, color=colors_a, edgecolor="white")
        ax1.set_title("Alarm Value Distribution (All Columns)", fontweight="bold")
        ax1.set_ylabel("Total Count")
        for i, v in enumerate(values):
            ax1.text(i, v + max(values) * 0.01, f"{v:,}", ha="center", fontsize=9)

    # Status distribution
    status_cols = categories.get("status", [])
    status_cols = [c for c in status_cols if c in df.columns]

    ax2 = axes[1]
    if status_cols:
        status_data = df[status_cols].apply(pd.to_numeric, errors="coerce")
        all_status = status_data.values.flatten()
        all_status = all_status[~np.isnan(all_status)]
        unique_vals, counts = np.unique(all_status.astype(int), return_counts=True)
        ax2.bar([str(v) for v in unique_vals], counts,
                color=PALETTE["status"], edgecolor="white")
        ax2.set_title("Status Value Distribution (All Columns)", fontweight="bold")
        ax2.set_ylabel("Total Count")
        for i, v in enumerate(counts):
            ax2.text(i, v + max(counts) * 0.01, f"{v:,}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "alarm_status_distribution.png"))
    plt.close(fig)
    print("        ✅ Saved alarm_status_distribution.png")


def plot_process_state_timeline(df: pd.DataFrame, categories: dict, save_dir: str):
    """
    Step plot showing process state transitions for each stage.
    """
    print("  [7.7] Plotting process state timeline...")
    state_cols = categories.get("process_state", [])
    state_cols = [c for c in state_cols if c in df.columns]

    if not state_cols:
        print("        No state columns found, skipping.")
        return

    fig, axes = plt.subplots(len(state_cols), 1, figsize=(18, 2.5 * len(state_cols)),
                              sharex=True)
    if len(state_cols) == 1:
        axes = [axes]

    colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#C62828", "#00838F"]
    for idx, col in enumerate(state_cols):
        ax = axes[idx]
        data = pd.to_numeric(df[col], errors="coerce")
        ax.step(range(len(data)), data.values, where="post",
                color=colors[idx % len(colors)], linewidth=1)
        ax.set_ylabel(col.replace("_STATE", ""), fontsize=10, fontweight="bold")
        ax.set_title(f"Process State — {col}", fontsize=10, loc="left")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample Index", fontsize=11)
    fig.suptitle("Process State Transitions", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "process_state_timeline.png"))
    plt.close(fig)
    print("        ✅ Saved process_state_timeline.png")


def run_eda(df_cleaned: pd.DataFrame, categories: dict, save_dir: str) -> pd.DataFrame:
    """
    Run all EDA visualizations. Returns the correlation matrix.
    """
    print_section("Task 7 — Exploratory Data Analysis")
    plot_feature_distributions(df_cleaned, categories, save_dir)
    corr_matrix = plot_correlation_heatmap(df_cleaned, categories, save_dir)
    plot_sensor_trends(df_cleaned, categories, save_dir)
    plot_outlier_analysis(df_cleaned, categories, save_dir)
    plot_variance_analysis(df_cleaned, categories, save_dir)
    plot_alarm_status_distribution(df_cleaned, categories, save_dir)
    plot_process_state_timeline(df_cleaned, categories, save_dir)
    print("\n  ✅ All EDA visualizations complete.")
    return corr_matrix


# ─────────────────────────────────────────────────────────────────────────────
# TASK 8 — ANOMALY DETECTION READINESS
# ─────────────────────────────────────────────────────────────────────────────

def assess_anomaly_readiness(
    df: pd.DataFrame,
    categories: dict[str, list[str]],
    corr_matrix: pd.DataFrame,
    save_dir: str,
) -> dict:
    """
    Assess readiness for anomaly detection and FDI attack simulation:

    1. Best candidate sensors for FDI attacks
       → High variance + high correlation with other sensors
       → Attacking these would propagate disruption through the process
    2. Highly correlated sensor pairs
       → Pearson ≥ 0.7 indicates physical/process coupling
       → Useful for GNN adjacency and cross-validation of attacks
    3. Critical process variables
       → Level sensors (LIT*), flow sensors (FIT*) are most impactful

    Returns a readiness report dictionary.
    """
    print_section("Task 8 — Anomaly Detection Readiness")

    sensor_cols = categories.get("sensor_pv", []) + categories.get("speed", [])
    sensor_cols = [c for c in sensor_cols if c in df.columns and c in corr_matrix.columns]

    readiness = {
        "fdi_candidates": [],
        "highly_correlated_pairs": [],
        "critical_variables": [],
    }

    # ── 1. FDI Attack Candidates ──
    # Score = variance_rank × avg_correlation_magnitude
    numeric_df = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
    variances = numeric_df.var()
    var_rank = variances.rank(pct=True)  # Percentile rank

    avg_abs_corr = corr_matrix[sensor_cols].loc[sensor_cols].abs().mean()

    fdi_score = (var_rank * 0.4 + avg_abs_corr * 0.6).sort_values(ascending=False)

    print("  🎯 Top FDI Attack Candidate Sensors:")
    print("     (High variance + high correlation = max attack impact)\n")
    print(f"     {'Rank':<5} {'Sensor':<20} {'FDI Score':<12} {'Variance':<15} {'Avg |Corr|':<12}")
    print(f"     {'─'*5} {'─'*20} {'─'*12} {'─'*15} {'─'*12}")

    for rank, (col, score) in enumerate(fdi_score.head(15).items(), 1):
        readiness["fdi_candidates"].append({
            "rank": rank,
            "sensor": col,
            "fdi_score": round(score, 4),
            "variance": round(variances.get(col, 0), 4),
            "avg_abs_corr": round(avg_abs_corr.get(col, 0), 4),
        })
        print(f"     {rank:<5} {col:<20} {score:<12.4f} {variances.get(col, 0):<15.4f} "
              f"{avg_abs_corr.get(col, 0):<12.4f}")

    # ── 2. Highly Correlated Pairs ──
    print("\n  🔗 Highly Correlated Sensor Pairs (|r| ≥ 0.7):\n")
    high_corr_pairs = []
    corr_sensor = corr_matrix[sensor_cols].loc[sensor_cols]
    for i in range(len(sensor_cols)):
        for j in range(i + 1, len(sensor_cols)):
            r = corr_sensor.iloc[i, j]
            if abs(r) >= 0.7:
                pair = {
                    "sensor_a": sensor_cols[i],
                    "sensor_b": sensor_cols[j],
                    "correlation": round(r, 4),
                }
                high_corr_pairs.append(pair)

    high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    readiness["highly_correlated_pairs"] = high_corr_pairs

    for pair in high_corr_pairs[:20]:
        direction = "+" if pair["correlation"] > 0 else "-"
        print(f"     {pair['sensor_a']:<20} ↔ {pair['sensor_b']:<20} "
              f"r = {direction}{abs(pair['correlation']):.4f}")

    if len(high_corr_pairs) > 20:
        print(f"     ... and {len(high_corr_pairs) - 20} more pairs")

    print(f"\n     Total highly correlated pairs: {len(high_corr_pairs)}")

    # ── 3. Critical Process Variables ──
    # Level (LIT), Flow (FIT), and Pressure (PIT) sensors directly
    # reflect physical process state and are the most impactful targets.
    critical_prefixes = ["LIT", "FIT", "PIT", "AIT", "DPIT"]
    print("\n  ⚠️  Critical Process Variables (by sensor type):\n")
    for prefix in critical_prefixes:
        matching = [c for c in sensor_cols if c.startswith(prefix)]
        if matching:
            readiness["critical_variables"].append({
                "type": prefix,
                "sensors": matching,
                "count": len(matching),
            })
            print(f"     {prefix:6s} ({len(matching):2d}): {matching}")

    # ── Save FDI candidate plot ──
    if readiness["fdi_candidates"]:
        fig, ax = plt.subplots(figsize=(12, 8))
        candidates = readiness["fdi_candidates"][:15]
        labels = [c["sensor"].replace(".Pv", "").replace(".Speed", " Spd") for c in candidates]
        scores = [c["fdi_score"] for c in candidates]
        colors = [PALETTE["danger"] if s > np.mean(scores) else PALETTE["warning"]
                  for s in scores]

        bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1],
                       edgecolor="white", height=0.7)
        ax.set_xlabel("FDI Attack Suitability Score", fontsize=12)
        ax.set_title("Top Candidate Sensors for False Data Injection Attacks",
                     fontsize=14, fontweight="bold")
        ax.axvline(x=np.mean(scores), color="gray", linestyle="--",
                   alpha=0.7, label=f"Mean score = {np.mean(scores):.3f}")
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "fdi_candidates.png"))
        plt.close(fig)
        print("\n  ✅ Saved fdi_candidates.png")

    # ── Correlation network graph ──
    if high_corr_pairs:
        fig, ax = plt.subplots(figsize=(14, 10))
        n_pairs = min(30, len(high_corr_pairs))
        top_pairs = high_corr_pairs[:n_pairs]

        # Collect unique nodes
        nodes = set()
        for p in top_pairs:
            nodes.add(p["sensor_a"])
            nodes.add(p["sensor_b"])
        nodes = sorted(nodes)

        # Simple circular layout
        n_nodes = len(nodes)
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(nodes, angles)}

        # Draw edges
        for pair in top_pairs:
            x = [pos[pair["sensor_a"]][0], pos[pair["sensor_b"]][0]]
            y = [pos[pair["sensor_a"]][1], pos[pair["sensor_b"]][1]]
            alpha = min(1.0, abs(pair["correlation"]))
            color = PALETTE["primary"] if pair["correlation"] > 0 else PALETTE["danger"]
            ax.plot(x, y, color=color, alpha=alpha * 0.6, linewidth=1.5)

        # Draw nodes
        for node in nodes:
            x, y = pos[node]
            ax.scatter(x, y, s=200, c=PALETTE["sensor"], edgecolors="white",
                       linewidth=1.5, zorder=5)
            label = node.replace(".Pv", "").replace(".Speed", " Spd")
            ax.annotate(label, (x, y), fontsize=7, ha="center", va="bottom",
                        xytext=(0, 10), textcoords="offset points", fontweight="bold")

        ax.set_title(f"Sensor Correlation Network (top {n_pairs} pairs, |r| ≥ 0.7)",
                     fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "correlation_network.png"))
        plt.close(fig)
        print("  ✅ Saved correlation_network.png")

    return readiness


# ─────────────────────────────────────────────────────────────────────────────
# TASK 9 — OUTPUT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def save_cleaned_dataset(df: pd.DataFrame, path: str):
    """Save the fully cleaned and combined dataset to CSV."""
    print(f"  Saving cleaned_dataset.csv ({df.shape[0]:,} rows × {df.shape[1]} cols)...")
    df.to_csv(path, index=False)
    size = os.path.getsize(path)
    print(f"  ✅ Saved ({fmt_bytes(size)})")


def save_feature_summary(
    categories: dict, quality: dict, readiness: dict, path: str
):
    """
    Generate feature_summary.csv with per-column metadata:
    column name, category, data type, quality issues, FDI score, etc.
    """
    rows = []
    # Build FDI score lookup
    fdi_lookup = {}
    for c in readiness.get("fdi_candidates", []):
        fdi_lookup[c["sensor"]] = c["fdi_score"]

    # Build bad-input lookup
    bad_input_lookup = {}
    for item in quality.get("bad_input_columns", []):
        bad_input_lookup[item["column"]] = item["count"]

    # Flatten categories into column → category mapping
    col_category = {}
    for cat_name, cols in categories.items():
        for col in cols:
            col_category[col] = cat_name

    all_cols = []
    for cols in categories.values():
        all_cols.extend(cols)

    for col in all_cols:
        row = {
            "column": col,
            "category": col_category.get(col, "unknown"),
            "has_bad_input": col in bad_input_lookup,
            "bad_input_count": bad_input_lookup.get(col, 0),
            "is_mixed_type": col in quality.get("mixed_type_columns", []),
            "fdi_score": fdi_lookup.get(col, None),
            "invalid_values": "; ".join(quality.get("invalid_value_columns", {}).get(col, [])),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(path, index=False)
    print(f"  ✅ Saved feature_summary.csv ({len(rows)} features)")


def save_correlation_matrix(corr: pd.DataFrame, path: str):
    """Save the full correlation matrix to CSV."""
    corr.to_csv(path)
    print(f"  ✅ Saved correlation_matrix.csv ({corr.shape[0]}×{corr.shape[1]})")


def generate_preprocessing_report(
    profiles: dict,
    categories: dict,
    quality_reports: dict,
    readiness: dict,
    output_path: str,
):
    """
    Generate a comprehensive Markdown report summarizing the entire
    preprocessing pipeline: profiling, quality, cleaning, and readiness.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Count features per category
    cat_counts = {k: len(v) for k, v in categories.items() if v}

    # Aggregate quality issues
    total_bad_input_cols = set()
    total_mixed_cols = set()
    for name, q in quality_reports.items():
        for item in q.get("bad_input_columns", []):
            total_bad_input_cols.add(item["column"])
        total_mixed_cols.update(q.get("mixed_type_columns", []))

    report = f"""# RAKSHAK-ICS — Preprocessing Report

**Generated**: {timestamp}
**Pipeline**: Complete EDA & Preprocessing for SWaT ICS Datasets

---

## 1. Dataset Overview

| Property | Dataset 1 | Dataset 2 | Dataset 3 |
|----------|-----------|-----------|-----------|
"""
    for prop_name, prop_key in [("Rows", "shape"), ("Columns", "shape"),
                                  ("Memory", "memory_human"), ("Duplicates", "duplicates")]:
        row = f"| {prop_name} |"
        for ds_name in ["dataset1", "dataset2", "dataset3"]:
            p = profiles.get(ds_name, {})
            if prop_key == "shape":
                val = p.get("shape", ("?", "?"))
                val = f"{val[0]:,}" if prop_name == "Rows" else str(val[1])
            elif prop_key == "memory_human":
                val = p.get("memory_human", "?")
            else:
                val = str(p.get(prop_key, "?"))
            row += f" {val} |"
        report += row + "\n"

    report += f"""
---

## 2. Feature Categorization

| Category | Count | Features |
|----------|-------|----------|
"""
    for cat, cols in categories.items():
        if cols:
            display = ", ".join(cols[:5])
            if len(cols) > 5:
                display += f", ... (+{len(cols)-5} more)"
            report += f"| {cat} | {len(cols)} | {display} |\n"

    report += f"""
**Total features**: {sum(cat_counts.values())}

---

## 3. Data Quality Summary

- **Mixed-type columns**: {len(total_mixed_cols)}
"""
    if total_mixed_cols:
        report += f"  - Columns: {', '.join(sorted(total_mixed_cols))}\n"

    report += f"- **'Bad Input' columns**: {len(total_bad_input_cols)}\n"
    if total_bad_input_cols:
        report += f"  - Columns: {', '.join(sorted(total_bad_input_cols))}\n"

    # Per-dataset quality details
    for ds_name, q in quality_reports.items():
        report += f"\n### {ds_name}\n"
        report += f"- Bad Input columns: {len(q.get('bad_input_columns', []))}\n"
        for item in q.get("bad_input_columns", []):
            report += f"  - `{item['column']}`: {item['count']} ({item['percentage']}%)\n"
        report += f"- Mixed-type columns: {len(q.get('mixed_type_columns', []))}\n"
        report += f"- Constant columns: {len(q.get('constant_columns', []))}\n"
        if q.get("constant_columns"):
            report += f"  - {', '.join(q['constant_columns'])}\n"

    report += f"""
---

## 4. Data Cleaning Summary

### Transformations Applied

| Feature Type | Cleaning Strategy |
|-------------|-------------------|
| Sensor (.Pv) | Convert to float64; 'Bad Input' → NaN → forward-fill + backfill |
| Speed (.Speed) | Convert to float64; same strategy as sensors |
| Status (.Status) | Convert to int; non-numeric → 0 |
| Alarm (.Alarm) | Active → 1, Inactive → 0, Bad Input → -1 |
| State (_STATE) | Convert to int |
| Timestamp | Parse to datetime (dayfirst=True) |

---

## 5. Feature Engineering

Three feature subsets were created:

| Subset | Description | Features |
|--------|-------------|----------|
| sensor_only | .Pv + .Speed columns | {len(categories.get('sensor_pv', [])) + len(categories.get('speed', []))} |
| sensor_status | Sensors + .Status | {len(categories.get('sensor_pv', [])) + len(categories.get('speed', [])) + len(categories.get('status', []))} |
| full_processed | All numeric features | {sum(len(v) for k, v in categories.items() if k != 'timestamp')} |

---

## 6. Scaling

Two scaling methods applied to each feature subset:

| Scaler | Method | Use Case |
|--------|--------|----------|
| StandardScaler | z = (x - μ) / σ | LSTM-AE, distance-based models |
| MinMaxScaler | x' = (x - min) / (max - min) | Neural networks, bounded activations |

---

## 7. Anomaly Detection Readiness

### Top FDI Attack Candidate Sensors

| Rank | Sensor | FDI Score | Variance | Avg |Corr| |
|------|--------|-----------|----------|------------|
"""
    for c in readiness.get("fdi_candidates", [])[:10]:
        report += (f"| {c['rank']} | {c['sensor']} | {c['fdi_score']:.4f} | "
                   f"{c['variance']:.4f} | {c['avg_abs_corr']:.4f} |\n")

    report += f"""
### Highly Correlated Sensor Pairs (|r| ≥ 0.7)

Total pairs found: **{len(readiness.get('highly_correlated_pairs', []))}**

| Sensor A | Sensor B | Correlation |
|----------|----------|-------------|
"""
    for pair in readiness.get("highly_correlated_pairs", [])[:15]:
        report += f"| {pair['sensor_a']} | {pair['sensor_b']} | {pair['correlation']:.4f} |\n"

    report += """
### Critical Process Variables

"""
    for cv in readiness.get("critical_variables", []):
        report += f"- **{cv['type']}** ({cv['count']}): {', '.join(cv['sensors'])}\n"

    report += f"""
---

## 8. Generated Outputs

| File | Description |
|------|-------------|
| `cleaned_dataset.csv` | Fully cleaned, combined dataset (all 3 datasets) |
| `preprocessing_report.md` | This report |
| `feature_summary.csv` | Per-feature metadata and quality flags |
| `correlation_matrix.csv` | Full sensor correlation matrix |
| `results/figures/*.png` | All EDA visualizations |

---

## 9. Visualizations Generated

1. `sensor_distributions.png` — Histograms of all sensor features
2. `correlation_heatmap.png` — Pearson correlation matrix heatmap
3. `sensor_trends.png` — Time-series trends for key sensors (P1–P6)
4. `outlier_boxplots.png` — Standardized boxplots for outlier detection
5. `variance_analysis.png` — Feature variance ranking (log scale)
6. `alarm_status_distribution.png` — Distribution of alarm/status values
7. `process_state_timeline.png` — Process state transitions
8. `fdi_candidates.png` — FDI attack suitability ranking
9. `correlation_network.png` — Sensor correlation network graph

---

*Report generated by RAKSHAK-ICS EDA Pipeline v1.0*
"""

    with open(output_path, "w") as f:
        f.write(report)
    print(f"  ✅ Saved preprocessing_report.md")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Execute the complete EDA and preprocessing pipeline:
      1. Setup directories
      2. Load datasets
      3. Profile datasets
      4. Categorize features
      5. Analyze data quality
      6. Clean data (combined across all datasets)
      7. Engineer feature subsets
      8. Scale features
      9. Run EDA visualizations
     10. Assess anomaly detection readiness
     11. Save all outputs
    """
    pipeline_start = datetime.now()
    print("=" * 72)
    print("  RAKSHAK-ICS — Complete EDA & Preprocessing Pipeline")
    print(f"  Started: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # ── Step 0: Setup ──
    ensure_dirs()

    # ── Step 1: Load ──
    datasets = load_datasets()

    # ── Step 2: Profile ──
    profiles = profile_all_datasets(datasets)

    # ── Step 3: Categorize (use first dataset's columns — all are identical) ──
    first_df = list(datasets.values())[0]
    categories = categorize_features(first_df.columns.tolist())

    # ── Step 4: Quality Analysis (per dataset) ──
    quality_reports = {}
    for name, df in datasets.items():
        quality_reports[name] = analyze_data_quality(df, name)

    # ── Step 5: Combine datasets with a source label ──
    print_section("Combining Datasets")
    for name, df in datasets.items():
        df["_source"] = name
    combined_raw = pd.concat(datasets.values(), ignore_index=True)
    print(f"  Combined shape: {combined_raw.shape[0]:,} rows × {combined_raw.shape[1]} cols")

    # ── Step 6: Clean ──
    combined_cleaned = clean_dataset(combined_raw, categories)

    # ── Step 7: Feature Engineering ──
    feature_sets = engineer_feature_sets(combined_cleaned, categories)

    # ── Step 8: Scaling ──
    scaled_sets = scale_features(feature_sets)

    # ── Step 9: EDA Visualizations ──
    corr_matrix = run_eda(combined_cleaned, categories, FIGURES_DIR)

    # ── Step 10: Anomaly Detection Readiness ──
    readiness = assess_anomaly_readiness(
        combined_cleaned, categories, corr_matrix, FIGURES_DIR
    )

    # ── Step 11: Save Outputs ──
    print_section("Task 9 — Saving Outputs")

    # 11a. Cleaned dataset (drop internal _source column for the output)
    save_cleaned_dataset(
        combined_cleaned.drop(columns=["_source"], errors="ignore"),
        os.path.join(DATA_DIR, "cleaned_dataset.csv"),
    )

    # 11b. Feature summary
    # Use the first dataset's quality report as representative
    first_quality = list(quality_reports.values())[0]
    save_feature_summary(
        categories, first_quality, readiness,
        os.path.join(RESULTS_DIR, "feature_summary.csv"),
    )

    # 11c. Correlation matrix
    save_correlation_matrix(corr_matrix, os.path.join(RESULTS_DIR, "correlation_matrix.csv"))

    # 11d. Preprocessing report
    generate_preprocessing_report(
        profiles, categories, quality_reports, readiness,
        os.path.join(RESULTS_DIR, "preprocessing_report.md"),
    )

    # ── Also save scaled versions ──
    print("\n  Saving scaled datasets...")
    scaled_dir = os.path.join(DATA_DIR, "scaled")
    os.makedirs(scaled_dir, exist_ok=True)
    for set_name, scalers in scaled_sets.items():
        for scaler_name, sdf in scalers.items():
            fname = f"{set_name}_{scaler_name}.csv"
            sdf.to_csv(os.path.join(scaled_dir, fname), index=False)
            print(f"    ✅ {fname}")

    # ── Pipeline Summary ──
    elapsed = (datetime.now() - pipeline_start).total_seconds()
    print_section("Pipeline Complete")
    print(f"  Total time   : {elapsed:.1f} seconds")
    print(f"  Rows processed: {combined_cleaned.shape[0]:,}")
    print(f"  Features      : {combined_cleaned.shape[1]}")
    print(f"  Plots saved   : {len([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')])} figures")
    print(f"\n  📁 Outputs:")
    print(f"     {os.path.join(DATA_DIR, 'cleaned_dataset.csv')}")
    print(f"     {os.path.join(DATA_DIR, 'scaled/')}")
    print(f"     {os.path.join(RESULTS_DIR, 'feature_summary.csv')}")
    print(f"     {os.path.join(RESULTS_DIR, 'correlation_matrix.csv')}")
    print(f"     {os.path.join(RESULTS_DIR, 'preprocessing_report.md')}")
    print(f"     {FIGURES_DIR}/*.png")
    print()


if __name__ == "__main__":
    main()
