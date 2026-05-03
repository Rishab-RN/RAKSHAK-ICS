"""
lstm_ae.py
──────────
LSTM Autoencoder for time-series anomaly detection (Blue Agent — Stream 1).

Architecture:
  Encoder: Linear(51→64) → LSTM(64→32, 2 layers) → Linear(32→16) → z
  Decoder: Linear(16→32) → LSTM(32→64, 2 layers) → Linear(64→51)

Training:
  - Normal-only data, MSE reconstruction loss
  - Anomaly score = per-sample MSE (higher = more anomalous)
  - Threshold τ set at 95th percentile of validation reconstruction error

Output:
  - lstm_score: float per window
  - sensor_errors: per-sensor reconstruction error for top-5 attribution
"""
