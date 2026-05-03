"""
evaluate.py
───────────
Unified evaluation framework for all RAKSHAK-ICS components.

Evaluation matrix:
  ┌────────────────────┬──────────────────────────────────────────┐
  │ Category           │ What is compared                         │
  ├────────────────────┼──────────────────────────────────────────┤
  │ ML Baselines       │ DT, RF, KNN, NB, IsoForest, K-Means     │
  │ Blue Agent Ablation│ LSTM-only, GNN-only, Fused (α-sweep)     │
  │ AI Attack Baselines│ Random, IDDFS, A*, AlphaBeta depth-3     │
  │ Adversarial Eval   │ Blue vs Standard SWaT attacks             │
  │                    │ Blue vs AI Baseline attackers              │
  │                    │ Blue vs DQN Red Agent                      │
  │ Cross-domain       │ SWaT-trained Blue → HAI (no retraining)   │
  └────────────────────┴──────────────────────────────────────────┘

Metrics:
  - F1 Score (macro), Precision, Recall
  - AUC-ROC curve
  - Confusion Matrix
  - Per-attack-type detection rate (36 SWaT attack scenarios)
  - Evasion rate (for adversarial evaluation)

Outputs:
  - results/tables/    → CSV comparison tables
  - results/figures/   → matplotlib/seaborn plots
  - results/latex/     → LaTeX-formatted tables for report
"""
