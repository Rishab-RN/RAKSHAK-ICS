"""
preprocess.py
─────────────
Data pipeline for SWaT and HAI datasets.

Responsibilities:
  - Load raw CSVs from data/swat/ and data/hai/
  - Clean: drop timestamps, handle NaN, remove constant columns
  - Normalise: fit StandardScaler on TRAIN-NORMAL only
  - Sliding windows: (window_size × n_features) for LSTM input
  - Graph construction: Pearson correlation ≥ 0.7 → adjacency for GNN
  - Train / Val / Test split with stratified attack labels
  - Save processed tensors to data/proof/
"""
