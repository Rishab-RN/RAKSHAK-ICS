"""
rl_train.py
───────────
DQN Red Agent training script using Stable-Baselines3.

Pipeline:
  1. Load frozen Blue Agent (LSTM-AE + GAT GNN weights locked)
  2. Instantiate ICSAttackEnv from rl_env.py (MultiDiscrete action space)
  3. Configure DQN hyperparameters:
     - learning_rate: 1e-4
     - buffer_size: 100,000
     - batch_size: 128
     - gamma: 0.99
     - exploration_fraction: 0.4
     - exploration_final_eps: 0.05
     - target_update_interval: 1,000
     - total_timesteps: 100,000
     - net_arch: [256, 256]
  4. Train DQN agent (~45 min on Colab T4)
  5. Save trained policy to models/red_agent/

Timebox rule:
  - If reward flat after 50K steps → fallback to single-sensor Discrete(255) DQN

Optional GA reward tuning:
  - Uses GA from ai_search.py to evolve {λ1, λ2, λ3}
  - Each individual = one DQN training run (shortened to 10K steps)
  - Fitness = evasion rate against Blue Agent
"""
