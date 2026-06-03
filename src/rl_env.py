"""
rl_env.py
─────────
Gymnasium-compatible environment for DQN Red Agent training.

Adversarial game formulation:
  - MAX player: Red Agent (attacker) — maximise evasion + disruption
  - MIN player: Blue Agent (frozen defender) — minimise attacker reward

State space (60-dim):
  [51 sensors | detection_flag | deviation_magnitude | 5×rolling_errors | 2×last_attacked]

Action space — MultiDiscrete([51, 5, 52, 5]):
  Attack 2 sensors simultaneously:
    - sensor1_id (0–50) × perturbation_mag1 (5 levels)
    - sensor2_id (0–51, 51=no-op) × perturbation_mag2 (5 levels)

Reward function:
  R = λ1 × Σ|δᵢ| − λ2 × detected_penalty − λ3 × (Σ|δᵢ|)²
  Where λ1, λ2, λ3 are optionally GA-optimised (see ai_search.py)
  Quadratic stealth penalty prevents large obvious perturbations.

Episode:
  - 200 steps per episode
  - Terminates early if Blue Agent detects 10 consecutive anomalies
  - ε-decay ≈ simulated annealing cooling schedule

Fallback:
  - ICSAttackEnvV1 with Discrete(255) if MultiDiscrete doesn't converge
"""
