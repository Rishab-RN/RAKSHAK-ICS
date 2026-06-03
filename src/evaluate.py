"""
evaluate.py
───────────
Unified evaluation framework for all RAKSHAK-ICS components.

Evaluation matrix:
  ┌────────────────────┬──────────────────────────────────────────┐
  │ Category           │ What is compared                         │
  ├────────────────────┼──────────────────────────────────────────┤
  │ ML Baselines       │ DT, RF, KNN, NB, IsoForest, K-Means, XGB│
  │ DL Baselines       │ Anomaly Transformer, USAD                │
  │ Blue Agent Ablation│ LSTM-only, GAT-only, no-fusion, Fused    │
  │ AI Attack Baselines│ Random, IDDFS, A*, AlphaBeta depth-3     │
  │ Adversarial Eval   │ Blue vs Standard SWaT attacks            │
  │                    │ Blue vs AI Baseline attackers             │
  │                    │ Blue vs DQN Red Agent (multi-sensor)     │
  │ Cross-domain       │ SWaT-trained Blue → HAI (no retraining)  │
  └────────────────────┴──────────────────────────────────────────┘

Metrics:
  - F1 Score (macro), Precision, Recall — all as mean±std (5 seeds)
  - AUC-ROC curve
  - Confusion Matrix
  - AHG = F1(standard) − F1(under_attacker) — Adversarial Hardening Gap
  - Per-attack-type detection rate
  - Evasion rate (for adversarial evaluation)
  - Paired t-test (p<0.001) for all key comparisons
  - Cohen's d effect size

Outputs:
  - results/tables/    → CSV comparison tables
  - results/figures/   → matplotlib/seaborn plots
  - results/latex/     → LaTeX-formatted tables for report
"""
