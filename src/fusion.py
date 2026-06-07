"""
fusion.py
─────────
Blue Agent score fusion: combines LSTM-AE and GAT GNN outputs.

Fusion formula:
  combined_score = α × lstm_score + (1 − α) × gnn_score
  anomaly_flag   = combined_score > τ

Parameters:
  - α (alpha):     fusion weight, tuned on validation set (grid search 0.0–1.0, step=0.05)
  - τ (threshold): decision boundary, set at optimal F1 point on val data

Additional outputs:
  - Top-5 sensor attribution from LSTM reconstruction error ranking
  - Confidence: distance of combined_score from threshold τ

Variants for ablation (4 configs):
  - LSTM-only   (α = 1.0)
  - GAT-only    (α = 0.0)
  - Both, no fusion (max of individual scores)
  - Full fused  (α = optimal)

Ablation wrapper:
  - run_ablation(configs, seeds=[42,123,456,789,1024]) → mean±std per config
"""
