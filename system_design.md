# RAKSHAK-ICS — System Design & Timeline

> Master blueprint — review and approve before any code is written.

---

## 1. High-Level System Architecture

```mermaid
graph TB
 subgraph DATA["📊 DATA LAYER"]
 SWAT["SWaT Dataset\n51 sensors · 500K rows\n36 attack types"]
 HAI["HAI Dataset\n86 channels\nCross-domain test"]
 PREP["preprocess.py\nClean → Scale → Window(60×51)\nPearson Graph (≥0.7)"]
 SWAT --> PREP
 HAI -.->|"same scaler\nno refit"| PREP
 end

 subgraph BLUE["🛡️ BLUE AGENT (Defender) — Trained First, Then FROZEN"]
 direction LR
 LSTM["Stream 1: LSTM-AE\nEnc(51→64→32→16)→z\nDec(16→32→64→51)\nMSE loss, normal-only"]
 GAT["Stream 2: GAT GNN\n51 nodes, Pearson≥0.7 edges\nGATConv(2-head, 8-dim)\n5-dim node features"]
 FUSION["Fusion Layer\nα × lstm_score\n+ (1-α) × gnn_score\n> τ → ANOMALY"]
 LSTM -->|"lstm_score"| FUSION
 GAT -->|"gnn_score"| FUSION
 end

 subgraph RED["🔴 RED AGENT (Attacker) — Trained Against Frozen Blue"]
 DQN["DQN (SB3)\nMultiDiscrete([51,5,52,5])\n2 sensors simultaneously\n100K steps, Colab T4"]
 REWARD["Reward Function\nλ1×Σ|δ| − λ2×detected\n− λ3×(Σ|δ|)²\nGA-optimised λ values"]
 DQN --> REWARD
 end

 subgraph BASELINES["📐 BASELINES"]
 ML["ML Classical (Units III-V)\nDT · RF · KNN · NB\nIsoForest · K-Means"]
 BOOST["ML Boosting\nXGBoost · LightGBM\nStrongest ML baselines"]
 AI["AI Search (Units I-II)\nRandom · IDDFS\nA* · AlphaBeta(d=3)"]
 DL["DL Baselines\nAnomaly Transformer\nUSAD"]
 end

 subgraph EVAL["📈 EVALUATION"]
 AHG["AHG Metric\nF1(static) − F1(attacker)\n5-seed mean±std\npaired t-test p<0.001"]
 end

 subgraph DASH["🖥️ DASHBOARD"]
 BACK["FastAPI Backend\nWebSocket /ws/stream\nHuggingFace Docker"]
 FRONT["React Frontend\nRecharts · Framer Motion\nVercel Deployment"]
 BACK <-->|"WebSocket\n10Hz"| FRONT
 end

 PREP --> BLUE
 PREP --> BASELINES
 BLUE -->|"FREEZE"| RED
 BLUE --> EVAL
 RED --> EVAL
 BASELINES --> EVAL
 BLUE --> BACK
 RED --> BACK

 style DATA fill:#0f172a,stroke:#38bdf8,color:#e2e8f0
 style BLUE fill:#0f172a,stroke:#34d399,color:#e2e8f0
 style RED fill:#0f172a,stroke:#f87171,color:#e2e8f0
 style BASELINES fill:#0f172a,stroke:#818cf8,color:#e2e8f0
 style EVAL fill:#0f172a,stroke:#fbbf24,color:#e2e8f0
 style DASH fill:#0f172a,stroke:#38bdf8,color:#e2e8f0
```

---

## 2. Data Flow & Training Pipeline

The order matters — each stage feeds the next. Blue must be frozen before Red trains.

```mermaid
flowchart TD
 A["🗂️ Raw SWaT CSV\n51 sensors + labels + timestamps"] --> B["🧹 Clean & Validate\nDrop timestamps, handle NaN\nRemove constant columns"]
 B --> C["📏 Normalize\nMinMaxScaler fit on\nTRAIN-NORMAL only"]
 C --> D["🪟 Sliding Windows\n(N, 60, 51) tensors\nstride=1"]
 C --> E["🔗 Build Sensor Graph\nPearson corr ≥ 0.7 → edges\n51 nodes, ~120 edges"]
 D --> F["✂️ Train / Val / Test Split\n70% / 15% / 15%\nStratified on attack labels"]

 F -->|"Normal windows only"| G["🏋️ Train LSTM-AE\n50 epochs, MSE loss\nAdam lr=1e-3\n5 seeds × 43 min each"]
 F -->|"Normal windows only"| H["🏋️ Train GAT\n30 epochs, MSE recon\nAdam lr=5e-4\n5-dim node features"]
 E --> H

 G -->|"lstm_score"| I["⚖️ Tune Fusion\nα sweep 0.0→1.0\nτ at optimal F1\non validation set"]
 H -->|"gnn_score"| I

 I --> J["❄️ FREEZE Blue Agent\nrequires_grad=False\nSave: lstm_ae.pt + gnn.pt\n+ fusion_params.json"]

 J --> K["🎮 Build RL Environment\nState(60) · Action MultiDiscrete\nFrozen Blue inside env"]
 K --> L["🏋️ Train DQN Red Agent\n100K steps, SB3\nε-decay = SA cooling\n~45 min Colab T4"]

 L --> M["📊 Full Evaluation\nAll attackers vs Blue\nAHG computation\n5-seed stat testing"]

 F -->|"All labelled data"| N["🏋️ Train ML Baselines\n8 models, 5 seeds each\nWindow → 51-dim mean vector\nIncludes XGBoost + LightGBM"]
 F --> O["📖 Clone & Run\nAnomaly Transformer\n+ USAD"]

 N --> M
 O --> M

 style A fill:#1e293b,stroke:#38bdf8,color:#e2e8f0
 style J fill:#1e293b,stroke:#f87171,color:#fca5a5
 style M fill:#1e293b,stroke:#fbbf24,color:#fde68a
```

---

## 3. Adversarial Game — Red vs Blue

This is the core research contribution. The game alternates MAX (Red) and MIN (Blue) moves.

```mermaid
sequenceDiagram
 participant ENV as 🏭 ICS Sensors (51)
 participant RED as 🔴 Red Agent (MAX)
 participant BLUE as 🛡️ Blue Agent (MIN)
 participant SCORE as 📈 AHG Tracker

 Note over ENV,SCORE: Episode = 500 steps

 loop Each Step (×500)
 ENV->>RED: State(60): [sensors | det_flag | deviation | rolling_errors]
 RED->>ENV: Action: perturb sensor_i by δ₁, sensor_j by δ₂
 Note right of RED: MAX: maximise disruption<br/>while minimising detection
 ENV->>BLUE: Perturbed window (60×51)
 BLUE->>ENV: {anomaly_flag, fused_score, top5_sensors}
 Note left of BLUE: MIN: detect anomaly<br/>(frozen weights)
 ENV->>RED: Reward: λ₁×|δ| − λ₂×detected − λ₃×|δ|²
 ENV->>SCORE: Update F1, detection_rate, stealth
 end

 SCORE->>SCORE: AHG = F1(static) − F1(DQN)
 Note over SCORE: Expected AHG ≈ 0.16<br/>p < 0.001 (5 seeds)
```

---

## 4. Evaluation Matrix

Every model is evaluated against every attacker — this table IS the paper's central result.

```mermaid
graph LR
 subgraph DETECTORS["Detection Models (Rows)"]
 D1["DT · RF · KNN · NB"]
 D1b["XGBoost · LightGBM"]
 D2["IsoForest · K-Means"]
 D3["USAD · Anomaly Transformer"]
 D4["LSTM-AE only"]
 D5["GAT only"]
 D6["LSTM+GAT Fused ⭐"]
 end

 subgraph ATTACKERS["Attack Scenarios (Columns)"]
 A1["Static SWaT\n(36 types)"]
 A2["Random\n(Unit I)"]
 A3["IDDFS\n(Unit I)"]
 A4["A* Search\n(Unit II)"]
 A5["AlphaBeta d=3\n(Unit II)"]
 A6["DQN single\n(RL)"]
 A7["DQN multi ⭐\n(RL upgraded)"]
 end

 subgraph METRICS["Metrics Per Cell"]
 M1["F1 mean±std"]
 M2["AHG"]
 M3["p-value"]
 M4["Stealth Score"]
 end

 DETECTORS --> METRICS
 ATTACKERS --> METRICS

 style DETECTORS fill:#0f172a,stroke:#34d399,color:#e2e8f0
 style ATTACKERS fill:#0f172a,stroke:#f87171,color:#e2e8f0
 style METRICS fill:#0f172a,stroke:#fbbf24,color:#e2e8f0
```

### Expected Results Table

> All results as mean±std across 5 seeds. XGBoost/LightGBM are the strongest ML baselines — the gap between them and LSTM+GAT Fused is the key argument for DL.

| Model | Type | Expected F1 | AUC-ROC | AIML Unit |
|---|---|---|---|---|
| Decision Tree | Supervised ML | 0.71±0.012 | 0.83 | Unit III |
| Random Forest | Ensemble ML | 0.71±0.010 | 0.87 | Unit IV |
| KNN (k=5) | Supervised ML | 0.65±0.015 | 0.79 | Unit IV |
| Naive Bayes | Supervised ML | 0.63±0.009 | 0.71 | Unit IV |
| **XGBoost** | **Gradient Boosting** | **0.76±0.011** | **0.89** | **Unit IV** |
| **LightGBM** | **Gradient Boosting** | **0.77±0.010** | **0.90** | **Unit IV** |
| Isolation Forest | Unsupervised | 0.67±0.011 | 0.77 | Unit V |
| K-Means (k=2) | Clustering | 0.57±0.018 | N/A | Unit V |
| USAD | DL Baseline | 0.85±0.013 | 0.91 | DL comparison |
| Anomaly Transformer | DL Baseline | 0.91±0.009 | 0.96 | DL comparison |
| LSTM-AE only | Your DL | 0.90±0.011 | 0.97 | Ablation A |
| GAT only | Your DL | 0.89±0.013 | 0.94 | Ablation B |
| **LSTM+GAT Fused** | **Your contribution** | **0.93±0.009** | **0.98** | **Primary result** |

> *"Even LightGBM — the strongest classical ML baseline at F1=0.77 — falls 16% short of our LSTM+GAT fusion (F1=0.93), demonstrating that handcrafted feature aggregation cannot capture temporal-structural patterns."*

---

## 5. Dashboard Architecture

```mermaid
graph LR
 subgraph VERCEL["Vercel (Free)"]
 REACT["React 18 + Vite"]
 SC["SensorChart\nRecharts LineChart"]
 AG["AnomalyGauge\nFramer Motion SVG"]
 SB["StatusBanner\nNORMAL / ATTACK"]
 EF["EventFeed\nScrolling alerts"]
 AB["AttributionBar\nTop-5 sensors"]
 RP["RedAgentPanel\nDQN actions live"]
 MS["ModeSelector\nNormal|Attack|DQN"]
 REACT --> SC & AG & SB & EF & AB & RP & MS
 end

 subgraph HF["HuggingFace Spaces (Docker, Free)"]
 FAST["FastAPI\nuvicorn :7860"]
 WS["WebSocket\n/ws/stream"]
 INF["inference.py\nBlueAgent + RedAgent"]
 DS["data_stream.py\nReplay + Live DQN"]
 ML["model_loader.py\nlstm_ae.pt + gnn.pt\n+ dqn_red_agent.zip"]
 FAST --> WS --> INF
 INF --> DS
 ML --> INF
 end

 REACT <-->|"wss:// WebSocket\n10Hz frames"| WS

 style VERCEL fill:#0f172a,stroke:#38bdf8,color:#e2e8f0
 style HF fill:#0f172a,stroke:#818cf8,color:#e2e8f0
```

---

## 6. Dataset Privacy & Safety Workflow

```mermaid
flowchart TD
 USER["👤 Rishab provides\nSWaT CSV files"] --> PLACE["Place in data/swat/\n(local only)"]
 PLACE --> GITIGNORE["🚫 .gitignore blocks\n*.csv, *.xlsx, *.npy\n*.pt, *.pkl in data/"]
 PLACE --> PROCESS["preprocess.py\nNormalize to [0,1]\nNever expose raw values"]

 PROCESS --> SAVE["Save processed tensors\ndata/proof/*.pt\n(also gitignored)"]
 PROCESS --> STATS["Save metadata only\ndata/proof/dataset_stats.json\nshape, col names, mean/std"]

 STATS -->|"✅ Safe to commit"| GIT["Git Repository"]
 SAVE -->|"🚫 Never committed"| BLOCKED["Blocked by .gitignore"]

 subgraph HOOK["Pre-commit Safety Check"]
 H1["scripts/check_data_privacy.py"]
 H2["Scan staged files:\n• No CSVs >1MB\n• No raw sensor values >1.0\n• No raw timestamps"]
 H1 --> H2
 end

 GIT --> HOOK
 HOOK -->|"❌ REJECT if\ndata leak found"| GIT
 HOOK -->|"✅ PASS"| PUSH["GitHub Push\n(clean, safe)"]

 GITIGNORE --> GIT

 style USER fill:#1e293b,stroke:#34d399,color:#e2e8f0
 style BLOCKED fill:#1e293b,stroke:#f87171,color:#fca5a5
 style PUSH fill:#1e293b,stroke:#34d399,color:#6ee7b7
 style HOOK fill:#1e293b,stroke:#fbbf24,color:#fde68a
```

---

## 7. File Structure

```
rakshak-ics/
├── configs/
│ └── default.yaml # All hyperparameters
├── data/
│ ├── swat/.gitkeep # 🔒 Raw CSVs here (gitignored)
│ ├── hai/.gitkeep # 🔒 HAI CSVs here (gitignored)
│ └── proof/.gitkeep # 🔒 Processed tensors (gitignored)
│ # dataset_stats.json ✅ (safe)
├── external/ # Cloned baseline repos
│ ├── anomaly-transformer/ # thuml/Anomaly-Transformer
│ └── usad/ # (gitignored or submoduled)
├── notebooks/ # Sequential pipeline
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_ml_baselines.ipynb ← DT, RF, KNN, NB, IsoForest, K-Means, XGBoost, LightGBM
│ ├── 03b_ai_baselines.ipynb ← Random, IDDFS, A*, AlphaBeta
│ ├── 03c_dl_baselines.ipynb ← Anomaly Transformer, USAD
│ ├── 04_lstm_autoencoder.ipynb
│ ├── 05_gat.ipynb ← GAT (not GraphSAGE)
│ ├── 05b_ablation.ipynb
│ ├── 06_fusion_eval.ipynb
│ ├── 07_rl_red_agent.ipynb
│ ├── 07b_ga_reward_tuning.ipynb
│ └── 08_adversarial_eval.ipynb
├── src/
│ ├── __init__.py
│ ├── preprocess.py # Data pipeline
│ ├── baselines.py # 8 ML models (incl. XGBoost, LightGBM)
│ ├── ai_search.py # 4 AI attackers + GA
│ ├── dl_baselines.py # Anomaly Transformer + USAD wrappers
│ ├── lstm_ae.py # LSTM Autoencoder
│ ├── gnn.py # GAT (upgraded from GraphSAGE)
│ ├── fusion.py # α-weighted fusion + ablation
│ ├── rl_env.py # Gymnasium env, MultiDiscrete
│ ├── rl_train.py # DQN training (SB3)
│ ├── evaluate.py # Unified eval + AHG
│ └── stat_utils.py # 5-seed wrapper + t-tests
├── scripts/
│ └── check_data_privacy.py # Pre-commit NDA checker
├── dashboard/
│ ├── backend/ # FastAPI + WebSocket
│ │ ├── main.py
│ │ ├── model_loader.py
│ │ ├── inference.py
│ │ ├── data_stream.py
│ │ ├── requirements.txt
│ │ └── Dockerfile
│ └── frontend/ # React + Vite
│ ├── src/
│ │ ├── App.jsx
│ │ ├── components/ (7 files)
│ │ └── hooks/useSensorStream.js
│ ├── package.json
│ └── vite.config.js
├── models/ # 🔒 Saved weights (gitignored)
├── results/
│ ├── figures/
│ ├── tables/
│ └── logs/ # 🔒 Training logs (gitignored)
├── report/ # Final project report
├── plans/ # Claude guides + blueprints
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 8. Eight-Week Timeline

```mermaid
gantt
 title RAKSHAK-ICS — 8-Week Build Plan
 dateFormat YYYY-MM-DD
 axisFormat %b %d

 section Week 1 — Foundation
 Project restructure project :w1a, 2026-06-02, 2d
 Email iTrust for SWaT :w1b, 2026-06-02, 1d
 Download HAI dataset :w1c, 2026-06-03, 1d
 Setup venv + install deps :w1d, 2026-06-04, 1d
 Privacy scripts + configs :w1e, 2026-06-04, 1d
 Git init + first push :w1f, 2026-06-05, 1d

 section Week 2 — EDA + Pipeline
 preprocess.py implementation :w2a, 2026-06-09, 2d
 01_eda.ipynb (6 visualizations) :w2b, 2026-06-09, 2d
 02_preprocessing.ipynb :w2c, 2026-06-11, 1d
 Correlation graph + node features :w2d, 2026-06-11, 2d
 EDA writeup for report :w2e, 2026-06-13, 1d

 section Week 3 — All Baselines
 8 ML baselines + 5-seed eval :w3a, 2026-06-16, 2d
 4 AI search attackers :w3b, 2026-06-18, 2d
 Anomaly Transformer + USAD :w3c, 2026-06-18, 2d
 Master comparison table :w3d, 2026-06-20, 1d

 section Week 4 — LSTM Autoencoder
 lstm_ae.py implementation :w4a, 2026-06-23, 1d
 5-seed training on Colab T4 :w4b, 2026-06-24, 2d
 Threshold tuning + attribution :w4c, 2026-06-26, 1d
 Update AI baselines with real Blue :w4d, 2026-06-27, 1d

 section Week 5 — GAT + Fusion
 GAT implementation + training :w5a, 2026-06-30, 2d
 Fusion α-sweep + ablation study :w5b, 2026-07-02, 2d
 HAI cross-domain evaluation :w5c, 2026-07-04, 1d

 section Week 6 — Red Agent ⚠️
 ICSAttackEnvV2 MultiDiscrete :crit, w6a, 2026-07-07, 1d
 DQN training 100K steps :crit, w6b, 2026-07-08, 2d
 GA reward tuning (optional) :w6c, 2026-07-10, 1d
 Full adversarial eval + AHG table :crit, w6d, 2026-07-11, 1d

 section Week 7 — Dashboard
 FastAPI backend + WebSocket :w7a, 2026-07-14, 1d
 React frontend (7 components) :w7b, 2026-07-15, 2d
 Deploy Vercel + HuggingFace :w7c, 2026-07-17, 1d
 Record OBS backup demo video :w7d, 2026-07-18, 1d

 section Week 8 — Report
 Final report with all results :w8a, 2026-07-21, 2d
 12-slide presentation :w8b, 2026-07-23, 1d
 arXiv cs.CR preprint submission :w8c, 2026-07-24, 1d
 Viva prep (30 questions) :w8d, 2026-07-25, 1d
```

---

## 9. AIML Curriculum Coverage Map

Every AIML unit explicitly maps to a RAKSHAK-ICS component:

```mermaid
graph LR
 ROOT((RAKSHAK-ICS)) --- U1["Unit I — Agents & Search"]
 ROOT --- U2["Unit II — Adversarial & Local"]
 ROOT --- U3["Unit III — Decision Trees"]
 ROOT --- U4["Unit IV — KNN NB Ensemble"]
 ROOT --- U5["Unit V — Clustering"]
 ROOT --- DL["Deep Learning"]
 ROOT --- RL["Reinforcement Learning"]

 U1 --- U1_1["Blue Agent PEAS<br/>• Goal-based + learning<br/>• Partially observable"]
 U1 --- U1_2["Red Agent PEAS<br/>• Utility-maximising<br/>• Multi-agent environment"]
 U1 --- U1_3["RandomAttacker<br/>• BFS analogue"]
 U1 --- U1_4["IDDFSAttacker<br/>• Depth-limited"]

 U2 --- U2_1["A* Attacker<br/>• h_n = remaining disruption<br/>• Admissible heuristic"]
 U2 --- U2_2["AlphaBeta depth=3<br/>• RED=MAX, BLUE=MIN<br/>• Lab Exercise 2"]
 U2 --- U2_3["Hill Climbing<br/>• DQN gradient = steepest ascent"]
 U2 --- U2_4["Simulated Annealing<br/>• ε-decay = cooling schedule"]
 U2 --- U2_5["Genetic Algorithm<br/>• Evolve λ1 λ2 λ3 weights"]

 U3 --- U3_1["DT Baseline<br/>• Pruning analysis<br/>• Feature importance"]

 U4 --- U4_1["KNN (k=5)"]
 U4 --- U4_2["Naive Bayes<br/>• Independence failure demo"]
 U4 --- U4_3["Random Forest<br/>• 100 estimators"]
 U4 --- U4_4["XGBoost<br/>• Gradient boosting<br/>• Strongest ML baseline"]
 U4 --- U4_5["LightGBM<br/>• Histogram-based<br/>• Validates XGB result"]

 U5 --- U5_1["K-Means (k=2 & k=5)<br/>• Elbow method<br/>• Silhouette score"]
 U5 --- U5_2["Isolation Forest<br/>• Anomaly scoring"]

 DL --- DL_1["LSTM Autoencoder<br/>• Temporal patterns"]
 DL --- DL_2["GAT GNN<br/>• Attention weights<br/>• Sensor dependencies"]
 DL --- DL_3["Anomaly Transformer<br/>• ICLR 2022 baseline"]

 RL --- RL_1["DQN Red Agent<br/>• MultiDiscrete action<br/>• 100K training steps"]
 RL --- RL_2["AHG Metric<br/>• Novel contribution"]

 style ROOT fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#e2e8f0
 style U1 fill:#1e293b,stroke:#818cf8,color:#e2e8f0
 style U2 fill:#1e293b,stroke:#818cf8,color:#e2e8f0
 style U3 fill:#1e293b,stroke:#818cf8,color:#e2e8f0
 style U4 fill:#1e293b,stroke:#818cf8,color:#e2e8f0
 style U5 fill:#1e293b,stroke:#818cf8,color:#e2e8f0
 style DL fill:#1e293b,stroke:#34d399,color:#e2e8f0
 style RL fill:#1e293b,stroke:#f87171,color:#e2e8f0
```

---

## 10. Key Research Contributions (What Makes This Publishable)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | First LSTM+GAT dual-stream fusion for ICS anomaly detection | PCGAT has GAT but no LSTM. STA-GNN has attention but no LSTM-AE fusion. | ✅ Novel |
| 2 | First DQN Red Agent on SWaT water treatment SCADA | No found paper. NREL 2024 is energy DER, not water. | ✅ Novel |
| 3 | First formal AHG metric definition and quantification | Stojanovic 2021 shows degradation exists but never formalises it. | ✅ Novel |
| 4 | First adversarial ICS paper with Indian policy context | No Indian institutional adversarial ICS paper found. CERT-In 2025 mandates timing. | ✅ Novel |
| ~~5~~ | ~~SOTA F1 on SWaT~~ | PCGAT 2025 claims SOTA. Our 0.93 is competitive, not definitively #1. | ❌ Do NOT claim |

---

> [!IMPORTANT]
> **Ready when you are.** All diagrams above are the blueprint. No code has been touched. When you say "start", I'll begin executing Week 1 tasks and will handle your datasets with full NDA safety.
