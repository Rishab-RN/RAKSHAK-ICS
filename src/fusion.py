"""
fusion.py
─────────
Blue Agent score fusion: combines LSTM-AE and GraphSAGE outputs.

Fusion formula:
  combined_score = α × lstm_score + (1 − α) × gnn_score
  anomaly_flag   = combined_score > τ

Parameters:
  - α (alpha):     fusion weight, tuned on validation set (grid search 0.0–1.0)
  - τ (threshold): decision boundary, set at optimal F1 point on val data

Additional outputs:
  - Top-5 sensor attribution from LSTM reconstruction error ranking
  - Confidence: distance of combined_score from threshold τ

Variants for ablation:
  - LSTM-only  (α = 1.0)
  - GNN-only   (α = 0.0)
  - Fused       (α = optimal)
"""
