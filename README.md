# 🛡️ RAKSHAK-ICS v4

**Adversarial Anomaly Detection for Industrial Control Systems**

> B.E. — CSE (AI & ML), RV College of Engineering, Bengaluru

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-ee4c2c.svg)](https://pytorch.org)

---

## 📋 Table of Contents

- [Overview](#-overview)
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

1. **Blue Agent (Defender)**: A fused LSTM Autoencoder + GraphSAGE GNN that learns normal sensor behavior and inter-sensor dependencies to detect anomalies.
2. **Red Agent (Attacker)**: A DQN reinforcement learning agent that learns to craft evasive perturbations against the frozen Blue Agent.
3. **AI Search Baselines**: Classical AI search algorithms (Random, IDDFS, A*, Alpha-Beta) formulated as attacker strategies.
4. **ML Baselines**: 6 classical ML models (DT, RF, KNN, NB, IsoForest, K-Means) for comparison.
5. **Live Dashboard**: React + FastAPI real-time monitoring with WebSocket sensor streaming.

The adversarial loop forces the Blue Agent to become robust against attacks it has never seen, while the Red Agent discovers novel evasion strategies.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      RAKSHAK-ICS v4 — AIML Complete                     │
├─────────────────────────────────────────────────────────────────────────┤
│  DATA LAYER                                                             │
│  SWaT CSV (51 sensors, 500K rows) → Normalise → Sliding Windows (60×51) │
│  HAI  CSV (86 channels) → Same scaler (cross-domain, no refit)          │
├─────────────────────────────────────────────────────────────────────────┤
│  AI SEARCH LAYER                                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐                │
│  │ Random   │ │ IDDFS    │ │ A*Search │ │ AlphaBeta    │                │
│  │ (BFS)    │ │ depth-lim│ │ h=disrup.│ │ depth=3      │                │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘                │
│  GA Optimiser: evolves {λ1,λ2,λ3} reward weights                        │
├─────────────────────────────────────────────────────────────────────────┤
│  BLUE AGENT (trained first, then FROZEN)                                │
│  Stream 1: LSTM-AE              Stream 2: GraphSAGE GNN                 │
│  Enc(51→64→32→16)→z             51 nodes, Pearson≥0.7 edges             │
│  Dec(16→32→64→51)               SAGEConv(2 layers, dim=32)              │
│  └── lstm_score ────────────────── └── gnn_score ──────┐                │
│                    FUSION: α×lstm + (1-α)×gnn > τ → ANOMALY             │
├──────────────── FREEZE BLUE  ───────────────────────────────────────────┤
│  DQN RED AGENT                                                          │
│  State(58) → DQN → Action(255)                                          │
│  Reward: λ1×|δ| − λ2×detected − λ3×|δ| (GA-optimised λ)                 │
├─────────────────────────────────────────────────────────────────────────┤
│  EVALUATION: ML Baselines | AI Baselines | Ablation | Cross-domain      │
├─────────────────────────────────────────────────────────────────────────┤
│  DASHBOARD: React (Vercel) ←WebSocket→ FastAPI (HuggingFace Docker)     │
│  Mode 1: Normal | Mode 2: SWaT Attack | Mode 3: DQN Red Agent           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
rakshak-ics/
├── data/
│   ├── swat/                    # SWaT dataset CSVs (gitignored)
│   ├── hai/                     # HAI dataset CSVs (gitignored)
│   └── proof/                   # Processed tensors, scalers (gitignored)
│
├── notebooks/                   # Sequential experiment notebooks
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_ml_baselines.ipynb    # DT, RF, KNN, NB, IsoForest, K-Means
│   ├── 03b_ai_baselines.ipynb   # Random, IDDFS, A*, AlphaBeta
│   ├── 04_lstm_autoencoder.ipynb
│   ├── 05_gnn.ipynb
│   ├── 06_fusion_eval.ipynb
│   ├── 07_rl_red_agent.ipynb
│   ├── 07b_ga_reward_tuning.ipynb
│   └── 08_adversarial_eval.ipynb
│
├── src/                         # Core source modules (flat layout)
│   ├── __init__.py
│   ├── preprocess.py            # Data pipeline: clean → scale → window → graph
│   ├── baselines.py             # 6 ML baseline models
│   ├── ai_search.py             # AI search attackers + GA optimiser
│   ├── lstm_ae.py               # LSTM Autoencoder (Blue Stream 1)
│   ├── gnn.py                   # GraphSAGE GNN (Blue Stream 2)
│   ├── fusion.py                # Score fusion + threshold tuning
│   ├── rl_env.py                # Gymnasium environment for Red Agent
│   ├── rl_train.py              # DQN training script (SB3)
│   └── evaluate.py              # Unified evaluation framework
│
├── dashboard/
│   ├── backend/                 # FastAPI server
│   │   ├── main.py              # App entry + WebSocket endpoint
│   │   ├── model_loader.py      # Load trained models at startup
│   │   ├── inference.py         # Real-time Blue/Red inference
│   │   ├── data_stream.py       # Sensor data replay + live mode
│   │   ├── requirements.txt     # Backend-only deps
│   │   └── Dockerfile           # HuggingFace Spaces deployment
│   └── frontend/                # React + Vite app
│       ├── src/
│       │   ├── App.jsx
│       │   ├── components/
│       │   │   ├── SensorChart.jsx
│       │   │   ├── AnomalyGauge.jsx
│       │   │   ├── StatusBanner.jsx
│       │   │   ├── EventFeed.jsx
│       │   │   ├── AttributionBar.jsx
│       │   │   ├── RedAgentPanel.jsx
│       │   │   └── ModeSelector.jsx
│       │   └── hooks/
│       │       └── useSensorStream.js
│       ├── package.json
│       └── vite.config.js
│
├── models/                      # Saved weights (gitignored)
├── results/
│   ├── figures/                 # Plots and visualizations
│   ├── tables/                  # CSV comparison tables
│   └── logs/                    # Training logs (gitignored)
├── report/                      # Final year project report
│
├── .gitignore
├── requirements.txt             # Python 3.10 pinned deps
├── LICENSE
└── README.md
```

---

## 📊 Datasets

### SWaT (Secure Water Treatment) — Primary
| Property         | Value                                           |
|------------------|-------------------------------------------------|
| Source           | iTrust, SUTD Singapore                           |
| Sensors          | 51 (flow, level, pressure, actuator states)      |
| Rows             | ~500,000 (7 days normal + 4 days attack)         |
| Attack Types     | 36 scenarios across 6 process stages             |
| Access           | Request at https://itrust.sutd.edu.sg            |

### HAI (HIL-based Augmented ICS) — Cross-domain
| Property         | Value                                           |
|------------------|-------------------------------------------------|
| Source           | NICS Lab, Korea                                  |
| Channels         | 86 (steam turbine + boiler)                      |
| Purpose          | Test Blue Agent generalisation without retraining|
| Access           | https://github.com/icsdataset/hai                |

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
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Install core dependencies
pip install -r requirements.txt

# PyG wheels — match your CUDA version
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

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

### Dataset Setup

1. Request SWaT access from [iTrust](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)
2. Place CSV/XLSX files in `data/swat/`
3. Download HAI from [GitHub](https://github.com/icsdataset/hai) into `data/hai/`
4. Run preprocessing: `python -m src.preprocess`

---

## 📖 Module Reference

| Module            | Purpose                                      | AIML Unit |
|-------------------|----------------------------------------------|-----------|
| `preprocess.py`   | Data cleaning, scaling, windowing, graph adj. | —         |
| `baselines.py`    | DT, RF, KNN, NB, IsoForest, K-Means         | III–V     |
| `ai_search.py`    | Random, IDDFS, A*, AlphaBeta + GA optimiser  | I–II      |
| `lstm_ae.py`      | LSTM Autoencoder (Blue Stream 1)             | DL        |
| `gnn.py`          | GraphSAGE GNN (Blue Stream 2)               | DL        |
| `fusion.py`       | α-weighted score fusion + threshold tuning   | DL        |
| `rl_env.py`       | Gymnasium env for adversarial game           | RL        |
| `rl_train.py`     | DQN training with SB3 + optional GA          | RL/II     |
| `evaluate.py`     | Unified metrics: F1, AUC-ROC, confusion      | —         |

---

## 🖥️ Dashboard

**Tech Stack**: React 18 + Vite (Vercel) ↔ FastAPI + WebSocket (HuggingFace Docker)

| Component          | Purpose                                              |
|--------------------|------------------------------------------------------|
| `SensorChart`      | Recharts live multi-sensor time-series               |
| `AnomalyGauge`     | Framer Motion animated anomaly score dial            |
| `StatusBanner`     | Normal / Alert / Attack status with colour animation |
| `EventFeed`        | Scrolling log of detection events                    |
| `AttributionBar`   | Top-5 contributing sensors bar chart                 |
| `RedAgentPanel`    | DQN actions, perturbation magnitude, evasion status  |
| `ModeSelector`     | Toggle: Normal → SWaT Attack → DQN Red Agent        |

### Demo Modes
1. **Normal** — replay clean SWaT sensor data, Blue Agent should show no anomalies
2. **SWaT Attack** — replay labelled attack scenarios, watch Blue Agent detect
3. **DQN Red Agent** — live Red Agent perturbations, adversarial evasion attempts

---

## 📅 8-Week Progress Checklist

### Week 1 — Foundation
- [x] Project scaffold, folder structure, requirements.txt
- [x] README with architecture and module docs
- [ ] Git repo initialised, first commit pushed
- [ ] SWaT dataset access request submitted
- [ ] Python 3.10 venv created, deps installed

### Week 2 — Data Pipeline & EDA
- [ ] `01_eda.ipynb` — exploratory data analysis on SWaT
- [ ] `02_preprocessing.ipynb` — cleaning, scaling, windowing
- [ ] `src/preprocess.py` — production data pipeline
- [ ] Pearson correlation graph constructed (adjacency matrix)
- [ ] Train/Val/Test split with attack labels

### Week 3 — ML Baselines
- [ ] `03_ml_baselines.ipynb` — all 6 models trained and evaluated
- [ ] `src/baselines.py` — reusable baseline module
- [ ] Results table: F1, Precision, Recall, AUC-ROC per model
- [ ] Confusion matrices saved to `results/figures/`

### Week 4 — Blue Agent (LSTM-AE + GNN)
- [ ] `04_lstm_autoencoder.ipynb` — LSTM-AE training on normal data
- [ ] `05_gnn.ipynb` — GraphSAGE training on sensor graph
- [ ] `src/lstm_ae.py` and `src/gnn.py` — model modules
- [ ] Reconstruction error threshold tuned on validation set
- [ ] Both models saved to `models/blue_agent/`

### Week 5 — Fusion + AI Baselines
- [ ] `06_fusion_eval.ipynb` — α-sweep, threshold optimisation
- [ ] `src/fusion.py` — fusion module with ablation support
- [ ] `03b_ai_baselines.ipynb` — all 4 AI search attackers
- [ ] `src/ai_search.py` — search attacker + GA modules
- [ ] Ablation table: LSTM-only vs GNN-only vs Fused

### Week 6 — Red Agent (DQN)
- [ ] `07_rl_red_agent.ipynb` — DQN training against frozen Blue
- [ ] `src/rl_env.py` — Gymnasium environment
- [ ] `src/rl_train.py` — SB3 DQN training script
- [ ] `07b_ga_reward_tuning.ipynb` — optional GA for λ weights
- [ ] Red Agent policy saved to `models/red_agent/`

### Week 7 — Evaluation + Dashboard
- [ ] `08_adversarial_eval.ipynb` — full comparison table
- [ ] `src/evaluate.py` — unified evaluation framework
- [ ] Cross-domain test: SWaT-trained Blue → HAI dataset
- [ ] Dashboard backend: FastAPI + WebSocket streaming
- [ ] Dashboard frontend: React components + live mode

### Week 8 — Polish + Report
- [ ] Dashboard deployed: Vercel (frontend) + HuggingFace (backend)
- [ ] Final results tables and figures generated
- [ ] LaTeX tables exported for report
- [ ] `report/` — final project report draft
- [ ] README updated with final results
- [ ] Code cleanup, docstrings, type hints

---

## 📈 Results

<!-- TODO: Fill after experiments are complete -->

### ML Baselines vs Blue Agent

| Model              | F1     | Precision | Recall | AUC-ROC |
|--------------------|--------|-----------|--------|---------|
| Decision Tree      | —      | —         | —      | —       |
| Random Forest      | —      | —         | —      | —       |
| KNN                | —      | —         | —      | —       |
| Naive Bayes        | —      | —         | —      | —       |
| Isolation Forest   | —      | —         | —      | —       |
| K-Means            | —      | —         | —      | —       |
| **LSTM-AE only**   | —      | —         | —      | —       |
| **GNN only**       | —      | —         | —      | —       |
| **Fused (α=opt)**  | —      | —         | —      | —       |

### Adversarial Robustness

| Attacker           | Blue Evasion Rate | Disruption Score |
|--------------------|-------------------|------------------|
| Random             | —                 | —                |
| IDDFS              | —                 | —                |
| A* Search          | —                 | —                |
| AlphaBeta (d=3)    | —                 | —                |
| **DQN Red Agent**  | —                 | —                |

### Cross-domain Transfer (SWaT → HAI)

| Metric    | SWaT (in-domain) | HAI (zero-shot) |
|-----------|-------------------|-----------------|
| F1        | —                 | —               |
| AUC-ROC   | —                 | —               |

---

## 🎓 AIML Curriculum Mapping

| AIML Unit        | Topic                          | RAKSHAK-ICS Component                         |
|------------------|--------------------------------|-----------------------------------------------|
| Unit I           | Search: BFS, DFS, IDDFS        | `ai_search.py` — Random, IDDFS attackers      |
| Unit I           | Informed Search: A*            | `ai_search.py` — A* attacker, h=disruption    |
| Unit II          | Adversarial Search, AlphaBeta  | `ai_search.py` — AlphaBeta attacker (depth=3) |
| Unit II          | Genetic Algorithm              | `ai_search.py` — GA reward weight tuning      |
| Unit II          | Hill Climbing, SA              | `rl_train.py` — gradient ≈ HC, ε-decay ≈ SA   |
| Unit III         | Decision Trees                 | `baselines.py` — DT classifier                |
| Unit IV          | KNN, Naive Bayes               | `baselines.py` — KNN, NB classifiers          |
| Unit IV          | Ensemble Methods               | `baselines.py` — Random Forest                |
| Unit V           | Clustering                     | `baselines.py` — K-Means anomaly scoring      |
| Unit V           | Unsupervised Learning          | `baselines.py` — Isolation Forest             |
| Deep Learning    | LSTM, Autoencoders             | `lstm_ae.py` — LSTM Autoencoder               |
| Deep Learning    | Graph Neural Networks          | `gnn.py` — GraphSAGE                          |
| RL               | DQN, Reward Design             | `rl_env.py`, `rl_train.py` — SB3 DQN          |

---

## 📚 References

1. Mathur & Tippenhauer. *SWaT: A Water Treatment Testbed for Research and Training on ICS Security.* CySWater, 2016.
2. Hamilton, Ying & Leskovec. *Inductive Representation Learning on Large Graphs (GraphSAGE).* NeurIPS, 2017.
3. Malhotra et al. *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection.* ICML Workshop, 2016.
4. Mnih et al. *Playing Atari with Deep Reinforcement Learning (DQN).* NeurIPS Workshop, 2013.
5. Shin et al. *HAI 1.0: HIL-based Augmented ICS Security Dataset.* USENIX, 2020.

---

