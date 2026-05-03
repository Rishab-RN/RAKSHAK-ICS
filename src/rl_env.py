"""
rl_env.py
─────────
Gymnasium-compatible environment for DQN Red Agent training.

Adversarial game formulation:
  - MAX player: Red Agent (attacker) — maximise evasion + disruption
  - MIN player: Blue Agent (frozen defender) — minimise attacker reward

State space (58-dim):
  [51 sensor values | detection_flag | deviation_magnitude | 5×rolling_errors]

Action space (255 discrete):
  sensor_id (0–50) × perturbation_magnitude (5 levels) = 51 × 5 = 255

Reward function:
  R = λ1 × |δ_disruption| − λ2 × detected_penalty − λ3 × |δ_magnitude|
  Where λ1, λ2, λ3 are optionally GA-optimised (see ai_search.py)

Episode:
  - 200 steps per episode
  - Terminates early if Blue Agent detects 10 consecutive anomalies
  - ε-decay ≈ simulated annealing cooling schedule

RL ↔ AI mapping:
  - DQN ≈ approximate Minimax value function via temporal difference
  - ε-greedy decay ≈ Simulated Annealing cooling schedule
  - Gradient updates ≈ Hill Climbing steps in reward landscape
"""
