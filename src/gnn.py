"""
gnn.py
──────
GAT GNN for inter-sensor dependency modelling (Blue Agent — Stream 2).

Graph Construction:
  - 65 nodes (one per sensor/actuator, after preprocessing)
  - Edges: Pearson correlation ≥ 0.7 between sensor pairs (436 edges from normal data)
  - Node features: [mean, std, min, max, range] per sliding window (5-dim per node)

Architecture:
  - GATConv Layer 1: (5 → 16, heads=8, edge_dim=1), ReLU, dropout=0.1
  - GATConv Layer 2: (128 → 5, heads=1), reconstruction head
  - Anomaly score = per-node reconstruction MSE, aggregated per graph

Training:
  - Reconstruction-based on normal-only graph snapshots
  - Anomaly score = MSE between input node features and reconstructed features

Output:
  - gnn_score: float per window (higher = more anomalous)
"""
