"""
gnn.py
──────
GraphSAGE GNN for inter-sensor dependency modelling (Blue Agent — Stream 2).

Graph Construction:
  - 51 nodes (one per sensor/actuator)
  - Edges: Pearson correlation ≥ 0.7 between sensor pairs (from normal data)
  - Node features: [μ, σ] per sliding window (2-dim per node)

Architecture:
  - SAGEConv Layer 1: (2 → 32), mean aggregation, ReLU
  - SAGEConv Layer 2: (32 → 32), mean aggregation
  - Global mean pooling → Linear(32 → 1) → sigmoid → gnn_score

Training:
  - Binary cross-entropy on labelled normal/attack windows
  - Or: one-class approach using normal-only graph embeddings

Output:
  - gnn_score: float per window (probability of anomaly)
"""
