# 🛡️ RAKSHAK-ICS

**Adversarial Anomaly Detection for Industrial Control Systems**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Research Contributions](#-key-research-contributions)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Datasets](#-datasets)
- [Setup & Installation](#-setup--installation)
- [Module Reference](#-module-reference)
- [Dashboard](#-dashboard)
- [8-Week Progress Checklist](#-8-week-progress-checklist)
- [Results](#-results)
- [AIML Curriculum Mapping](#-aiml-curriculum-mapping)
- [References](#-references)

---

## 🔍 Overview

Industrial Control Systems (ICS) — water treatment plants, power grids, manufacturing lines — are increasingly targeted by sophisticated cyber-attacks. Traditional signature-based detection fails against novel, adversarial attacks.

**RAKSHAK-ICS** addresses this through an adversarial training framework:

1. **Blue Agent (Defender)**: A fused LSTM Autoencoder + GAT GNN that learns normal sensor behavior and inter-sensor dependencies to detect anomalies.
2. **Red Agent (Attacker)**: A multi-sensor DQN reinforcement learning agent that learns to craft coordinated evasive perturbations against the frozen Blue Agent.
3. **AI Search Baselines**: Classical AI search algorithms (Random, IDDFS, A*, Alpha-Beta) formulated as attacker strategies.
4. **ML Baselines**: 6 classical ML models (DT, RF, KNN, NB, IsoForest, K-Means) + XGBoost ensemble for comparison.
5. **DL Baselines**: Anomaly Transformer (ICLR 2022) and USAD (KDD 2020) for journal-grade comparison.
6. **Live Dashboard**: React + FastAPI real-time monitoring with WebSocket sensor streaming at 10Hz.

The adversarial loop forces the Blue Agent to become robust against attacks it has never seen, while the Red Agent discovers novel evasion strategies.

---

## 🏆 Key Research Contributions

| # | Novelty | Description |
|---|---------|-------------|
| 1 | **First LSTM+GAT dual-stream for ICS** | Combines temporal (LSTM-AE) and spatial (GAT) anomaly detection streams with learnable fusion |
| 2 | **First multi-sensor DQN Red Agent on SWaT** | MultiDiscrete action space attacks 2 sensors simultaneously, modelling coordinated FDI attacks |
| 3 | **Formal AHG metric** | Adversarial Hardening Gap = F1(standard) − F1(under_attacker), quantifying adversarial robustness |
| 4 | **Adversarial ICS study with Indian policy context** | CERT-In alignment for critical infrastructure protection |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ RAKSHAK-ICS                                                           │
├─────────────────────────────────────────────────────────────────────────┤
│ DATA LAYER                                                            │
│ SWaT A9 (65 features, 87K rows) → MinMaxScale → Sliding Windows(60×65)│
│ HAI (86 channels) → Same scaler (cross-domain, no refit)              │
├─────────────────────────────────────────────────────────────────────────┤
│ AI SEARCH LAYER (Unit I–II)                                           │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐              │
│ │ Random   │ │ IDDFS    │ │ A*Search │ │ AlphaBeta    │              │
│ │ (BFS)    │ │ depth-lim│ │ h=disrup.│ │ depth=3      │              │
│ └──────────┘ └──────────┘ └──────────┘ └──────────────┘              │
│ GA Optimiser: evolves {λ1,λ2,λ3} reward weights                      │
├─────────────────────────────────────────────────────────────────────────┤
│ BLUE AGENT (trained first, then FROZEN)                               │
│ Stream 1: LSTM-AE            Stream 2: GAT GNN                       │
│ Enc(65→64→32→16)→z           65 nodes, Pearson≥0.7 → 436 edges       │
│ Dec(16→32→64→65)             GATConv(5→16, heads=8) × 2 layers       │
│ └── lstm_score ─────────────── └── gnn_score ──────┐                  │
│                FUSION: α×lstm + (1-α)×gnn > τ → ANOMALY              │
├──────────────── FREEZE BLUE ───────────────────────────────────────────┤
│ DQN RED AGENT (Multi-Sensor)                                          │
│ State(60) → DQN [256,256] → MultiDiscrete([51,5,52,5])               │
│ Reward: λ1×Σ|δᵢ| − λ2×detected − λ3×(Σ|δᵢ|)² (quadratic stealth)   │
├─────────────────────────────────────────────────────────────────────────┤
│ EVALUATION: ML Baselines | DL Baselines | Ablation | Cross-domain    │
│             All results: mean±std (5 seeds), paired t-test p<0.001   │
├─────────────────────────────────────────────────────────────────────────┤
│ DASHBOARD: React (Vercel) ←WebSocket 10Hz→ FastAPI (HuggingFace)     │
│ Mode 1: Normal | Mode 2: SWaT Attack | Mode 3: DQN Red Agent         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
rakshak-ics/
├── configs/
│   └── default.yaml            # All hyperparameters (LSTM, GAT, DQN, fusion, etc.)
│
├── data/
│   ├── swat/                   # SWaT dataset CSVs (gitignored, NDA-protected)
│   ├── hai/                    # HAI dataset CSVs (gitignored)
│   └── proof/                  # Privacy-safe stats only (dataset_stats.json)
│
├── notebooks/                  # Sequential experiment notebooks
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_ml_baselines.ipynb   # DT, RF, KNN, NB, IsoForest, K-Means, XGBoost
│   ├── 03b_ai_baselines.ipynb  # Random, IDDFS, A*, AlphaBeta
│   ├── 03c_dl_baselines.ipynb  # Anomaly Transformer + USAD
│   ├── 04_lstm_autoencoder.ipynb
│   ├── 05_gat.ipynb            # GAT GNN training
│   ├── 05b_ablation.ipynb      # 4-config ablation study
│   ├── 06_fusion_eval.ipynb
│   ├── 07_rl_red_agent.ipynb
│   ├── 07b_ga_reward_tuning.ipynb
│   └── 08_adversarial_eval.ipynb
│
├── src/                        # Core source modules (flat layout)
│   ├── __init__.py
│   ├── preprocess.py           # Data pipeline: clean → scale → window → graph
│   ├── baselines.py            # 6 ML baselines + XGBoost
│   ├── ai_search.py            # AI search attackers + GA optimiser
│   ├── lstm_ae.py              # LSTM Autoencoder (Blue Stream 1)
│   ├── gnn.py                  # GAT GNN (Blue Stream 2)
│   ├── fusion.py               # Score fusion + threshold tuning + ablation
│   ├── rl_env.py               # Gymnasium env — MultiDiscrete Red Agent
│   ├── rl_train.py             # DQN training script (SB3)
│   ├── evaluate.py             # Unified evaluation + AHG metric
│   └── stat_utils.py           # 5-seed wrapper, paired t-test, Cohen's d
│
├── scripts/
│   └── check_data_privacy.py   # Pre-commit NDA compliance checker
│
├── dashboard/
│   ├── backend/                # FastAPI + WebSocket server
│   │   ├── main.py             # App entry + /ws/stream endpoint
│   │   ├── inference.py        # BlueAgent + RedAgent inference classes
│   │   ├── data_stream.py      # SWaTStream sensor replay + live mode
│   │   ├── Dockerfile          # HuggingFace Spaces deployment
│   │   └── requirements.txt
│   └── frontend/               # React 18 + Vite app
│       ├── src/
│       │   ├── App.jsx
│       │   └── components/     # SensorChart, AnomalyGauge, StatusBanner,
│       │                       # EventFeed, AttributionBar, RedAgentPanel,
│       │                       # ModeSelector
│       ├── package.json
│       └── vite.config.js
│
├── models/                     # Saved weights (gitignored)
├── results/
│   ├── figures/                # Plots and visualizations
│   ├── tables/                 # CSV comparison tables
│   └── logs/                   # Training logs (gitignored)
├── plans/                      # Project plans and guides
├── report/                     # Final year project report
│
├── .gitignore                  # NDA-safe exclusions
├── requirements.txt            # Python 3.10 pinned deps
├── system_design.md            # Full system design document
└── README.md
```

---

## 📊 Datasets

### SWaT A9 (Secure Water Treatment) — Primary

| Property | Value |
|--|--|
| Source | iTrust, SUTD Singapore |
| Version | A9 (Dec 2023) — Clean dataset (no attacks) |
| Features | **65** after preprocessing (87 raw → cleaned) |
| Raw Columns | 31 continuous (.Pv), 32 actuators (.Status), 15 alarms, 2 speed, 6 state |
| Files | dataset1.csv (28,860 rows), dataset2.csv (29,160), dataset3.csv (29,160) |
| Total Rows | 87,180 |
| Access | NDA required — request at [iTrust](https://itrust.sutd.edu.sg) |

#### Preprocessing Pipeline Output

| Split | Windows Shape | Node Features |
|-------|---------------|---------------|
| Train | (60,966, 60, 65) | (60,966, 65, 5) |
| Val | (13,018, 60, 65) | (13,018, 65, 5) |
| Test | (13,019, 60, 65) | (13,019, 65, 5) |

- **Scaler**: MinMaxScaler fit on training data only
- **Graph**: 65 nodes, 436 edges (Pearson ≥ 0.7)
- **Node features**: [mean, std, min, max, range] per sensor per window

### HAI (HIL-based Augmented ICS) — Cross-domain

| Property | Value |
|--|--|
| Source | NICS Lab, Korea |
| Channels | 86 (steam turbine + boiler) |
| Purpose | Test Blue Agent generalisation without retraining |
| Access | [GitHub](https://github.com/icsdataset/hai) |

> ⚠️ **NDA Notice**: SWaT dataset is under NDA with iTrust/SUTD. Raw data must NEVER be committed to git. All results use normalised [0,1] scale only.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 (PyTorch Geometric requirement)
- CUDA 11.8 (optional, for GPU training)
- Node.js 18+ (for React dashboard)
- Git

### ML/DL Environment

```bash
# Clone
git clone https://github.com/<your-username>/rakshak-ics.git
cd rakshak-ics

# Create virtual environment (Python 3.10)
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install core dependencies
pip install -r requirements.txt

# PyG wheels — match your CUDA version
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### Dataset Setup

1. Request SWaT access from [iTrust](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)
2. Place CSV files in `data/swat/` (dataset1.csv, dataset2.csv, dataset3.csv)
3. Download HAI from [GitHub](https://github.com/icsdataset/hai) into `data/hai/`
4. Run preprocessing: `python -m src.preprocess`
5. Verify privacy: `python scripts/check_data_privacy.py`

### Dashboard Setup

```bash
# Backend
cd dashboard/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 7860

# Frontend (separate terminal)
cd dashboard/frontend
npm install
npm run dev
```

---

## 📖 Module Reference

| Module | Purpose | Key Details |
|--|--|--|
| `preprocess.py` | Data cleaning, MinMaxScaling, windowing, graph construction | 887 lines, handles Bad Input, 65 features |
| `baselines.py` | DT, RF, KNN, NB, IsoForest, K-Means, XGBoost | All with `run_with_seeds()` |
| `ai_search.py` | Random, IDDFS, A*, AlphaBeta + GA optimiser | GA: pop=20, gen=50, BLX-α crossover |
| `lstm_ae.py` | LSTM Autoencoder (Blue Stream 1) | Enc(65→64→32→16)→z→Dec, MSE reconstruction |
| `gnn.py` | GAT GNN (Blue Stream 2) | GATConv(5→16, heads=8) × 2, 436 edges |
| `fusion.py` | α-weighted score fusion + ablation | 4 configs: LSTM-only, GAT-only, no-fusion, full |
| `rl_env.py` | Gymnasium env — MultiDiscrete Red Agent | State(60), Action MultiDiscrete([51,5,52,5]) |
| `rl_train.py` | DQN training with SB3 | net_arch=[256,256], 100K steps |
| `evaluate.py` | Unified metrics + AHG computation | F1, AUC-ROC, paired t-test, mean±std |
| `stat_utils.py` | 5-seed reproducibility + statistical testing | set_all_seeds, paired_ttest, Cohen's d |
| `check_data_privacy.py` | Pre-commit NDA compliance checker | Blocks raw CSV/timestamps/sensor values |

---

## 🖥️ Dashboard

**Tech Stack**: React 18 + Recharts + Framer Motion + TailwindCSS (Vercel) ↔ FastAPI + WebSocket at 10Hz (HuggingFace Docker)

| Component | Purpose |
|--|--|
| `SensorChart` | Recharts live multi-sensor time-series |
| `AnomalyGauge` | Framer Motion animated anomaly score dial |
| `StatusBanner` | Normal / Alert / Attack status with colour animation |
| `EventFeed` | Scrolling log of detection events |
| `AttributionBar` | Top-5 contributing sensors bar chart |
| `RedAgentPanel` | DQN actions, perturbation magnitude, evasion status |
| `ModeSelector` | Toggle: Normal → SWaT Attack → DQN Red Agent |

### Demo Modes
1. **Normal** — replay clean SWaT sensor data, Blue Agent shows no anomalies
2. **SWaT Attack** — replay labelled attack scenarios, watch Blue Agent detect
3. **DQN Red Agent** — live Red Agent coordinated perturbations, adversarial evasion

---

## 📅 8-Week Progress Checklist

### Week 1 — Foundation ✅
- [x] Project scaffold, folder structure, requirements.txt
- [x] README with architecture and module docs
- [x] Git repo initialised, branch `dev/aiml-pipeline` created
- [x] `configs/default.yaml` — all hyperparameters
- [x] `scripts/check_data_privacy.py` — pre-commit NDA checker
- [x] `src/stat_utils.py` — 5-seed wrapper + paired t-test
- [x] `.gitignore` strengthened for NDA compliance

### Week 2 — Data Pipeline & EDA ✅
- [x] `src/preprocess.py` — full 887-line production pipeline
- [x] SWaT A9 data loaded, cleaned, normalised (65 features)
- [x] MinMaxScaler fit on training data only
- [x] Sliding windows: (60,966, 60, 65) train shape
- [x] Sensor correlation graph: 65 nodes, 436 edges
- [x] 5-dim node features: [mean, std, min, max, range]
- [x] `data/proof/dataset_stats.json` — privacy-safe stats
- [ ] `01_eda.ipynb` — 6 dark-themed visualizations
- [ ] `02_preprocessing.ipynb` — pipeline validation notebook

### Week 3 — All Baselines
- [ ] `src/baselines.py` — 6 ML models + XGBoost, `run_with_seeds()` results
- [ ] `src/ai_search.py` — 4 attackers + GA (placeholder Blue Agent)
- [ ] `03_ml_baselines.ipynb` — all results as mean±std
- [ ] `03b_ai_baselines.ipynb` — attacker evaluation framework
- [ ] `03c_dl_baselines.ipynb` — Anomaly Transformer + USAD

### Week 4 — LSTM Autoencoder
- [ ] `src/lstm_ae.py` — full implementation
- [ ] `04_lstm_autoencoder.ipynb` — 5-seed training
- [ ] Threshold tuning at 95th percentile
- [ ] Paired t-test vs RF baseline → p<0.001
- [ ] Save: `models/lstm_ae.pt`, `models/scaler.pkl`

### Week 5 — GAT + Fusion + HAI
- [ ] `src/gnn.py` — GATConv implementation
- [ ] `05_gat.ipynb` — GAT training with 5-dim node features
- [ ] `05b_ablation.ipynb` — 4-config ablation study
- [ ] `06_fusion_eval.ipynb` — α-sweep, threshold optimisation
- [ ] HAI cross-domain evaluation (no retraining)
- [ ] Save: `models/gnn.pt`, `models/fusion_params.json`

### Week 6 — Red Agent (DQN) ⚠️ TIMEBOXED
- [ ] `src/rl_env.py` — ICSAttackEnv with MultiDiscrete
- [ ] `src/rl_train.py` — DQN training, 100K steps
- [ ] `07_rl_red_agent.ipynb` — training + reward curves
- [ ] `07b_ga_reward_tuning.ipynb` — GA for λ weights (optional)
- [ ] `08_adversarial_eval.ipynb` — full AHG table
- [ ] **HARD RULE**: If reward flat after 50K steps → fallback to single-sensor DQN

### Week 7 — Dashboard
- [ ] `dashboard/backend/` — FastAPI + WebSocket, all files
- [ ] `dashboard/frontend/` — React components wired to backend
- [ ] Deploy: Vercel (frontend) + HuggingFace Spaces (backend)
- [ ] Test all 3 modes: Normal / SWaT Attack / DQN Red Agent

### Week 8 — Report + Submission
- [ ] Final report with mean±std results, AHG finding
- [ ] 12-slide presentation deck
- [ ] arXiv cs.CR preprint submission
- [ ] GitHub repo → public, README with all links

---

## 📈 Results

> All results reported as **mean±std across 5 seeds** (42, 123, 456, 789, 1024).
> Statistical significance via paired t-test at p<0.001.

### Table A — Detection Performance

| Model | F1 | Precision | Recall | AUC-ROC |
|--|--|--|--|--|
| Decision Tree | — | — | — | — |
| Random Forest | — | — | — | — |
| KNN | — | — | — | — |
| Naive Bayes | — | — | — | — |
| Isolation Forest | — | — | — | — |
| K-Means | — | — | — | — |
| XGBoost | — | — | — | — |
| USAD | — | — | — | — |
| Anomaly Transformer | — | — | — | — |
| **LSTM-AE only** | — | — | — | — |
| **GAT only** | — | — | — | — |
| **Fused (α=opt)** | — | — | — | — |
| **HAI (zero-shot)** | — | — | — | — |

### Table B — Adversarial Robustness (AHG)

| Attacker | F1 (Blue vs attacker) | AHG | Evasion Rate |
|--|--|--|--|
| Random | — | — | — |
| IDDFS | — | — | — |
| A* Search | — | — | — |
| AlphaBeta (d=3) | — | — | — |
| DQN (single-sensor) | — | — | — |
| **DQN (multi-sensor)** | — | — | — |

### Table C — Ablation Study

| Configuration | F1 | Δ from Full |
|--|--|--|
| LSTM-only (α=1.0) | — | — |
| GAT-only (α=0.0) | — | — |
| Both, no fusion | — | — |
| **Full fused (α=opt)** | — | — |

---

## 🎓 AIML Curriculum Mapping

| AIML Unit | Topic | RAKSHAK-ICS Component |
|--|--|--|
| Unit I | Search: BFS, DFS, IDDFS | `ai_search.py` — Random, IDDFS attackers |
| Unit I | Informed Search: A* | `ai_search.py` — A* attacker, h=disruption |
| Unit II | Adversarial Search, AlphaBeta | `ai_search.py` — AlphaBeta attacker (depth=3) |
| Unit II | Genetic Algorithm | `ai_search.py` — GA reward weight tuning |
| Unit II | Hill Climbing, SA | `rl_train.py` — gradient ≈ HC, ε-decay ≈ SA |
| Unit III | Decision Trees | `baselines.py` — DT classifier |
| Unit IV | KNN, Naive Bayes | `baselines.py` — KNN, NB classifiers |
| Unit IV | Ensemble Methods | `baselines.py` — Random Forest, XGBoost |
| Unit V | Clustering | `baselines.py` — K-Means anomaly scoring |
| Unit V | Unsupervised Learning | `baselines.py` — Isolation Forest |
| Deep Learning | LSTM, Autoencoders | `lstm_ae.py` — LSTM Autoencoder |
| Deep Learning | Graph Attention Networks | `gnn.py` — GAT (GATConv, multi-head attention) |
| RL | DQN, Multi-sensor Actions | `rl_env.py`, `rl_train.py` — SB3 DQN MultiDiscrete |

---

## 📚 References

1. Mathur & Tippenhauer. *SWaT: A Water Treatment Testbed for Research and Training on ICS Security.* CySWater, 2016.
2. Veličković et al. *Graph Attention Networks (GAT).* ICLR, 2018.
3. Malhotra et al. *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection.* ICML Workshop, 2016.
4. Mnih et al. *Playing Atari with Deep Reinforcement Learning (DQN).* NeurIPS Workshop, 2013.
5. Shin et al. *HAI 1.0: HIL-based Augmented ICS Security Dataset.* USENIX, 2020.
6. Xu et al. *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy.* ICLR, 2022.
7. Audibert et al. *USAD: UnSupervised Anomaly Detection on Multivariate Time Series.* KDD, 2020.
8. Chen et al. *PCGAT: Process-Conditioned Graph Attention for ICS Anomaly Detection.* 2025.

---

*Built with ❤️ for critical infrastructure security.*
