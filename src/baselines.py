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

Each model:
  - Trains on normal-only data (for unsupervised) or labelled data (supervised)
  - Evaluates: F1, Precision, Recall, AUC-ROC, Confusion Matrix
  - Results saved to results/baselines/
"""
