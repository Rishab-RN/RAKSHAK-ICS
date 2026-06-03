"""
lstm_ae.py
──────────
LSTM Autoencoder for time-series anomaly detection (Blue Agent — Stream 1).

Architecture:
  Encoder: Linear(65→64) → LSTM(64→32, 2 layers, dropout=0.1) → Linear(32→16) → z
  Decoder: Linear(16→32) → LSTM(32→64, 2 layers, dropout=0.1) → Linear(64→65)

Training:
  - Normal-only data (SWaT A9 clean dataset)
  - MSE reconstruction loss
  - Anomaly score = per-sample MSE (higher = more anomalous)
  - Threshold τ set at 95th percentile of validation reconstruction error
  - Early stopping: patience=10, ReduceLROnPlateau(factor=0.5, patience=5)

Output:
  - lstm_score: float per window
  - sensor_errors: per-sensor reconstruction error for top-5 attribution
"""
