"""
rl_env.py
─────────
Gymnasium-compatible environment for DQN Red Agent training.

Adversarial game formulation:
  - MAX player: Red Agent (attacker) — maximise evasion + disruption
  - MIN player: Blue Agent (frozen defender) — minimise attacker reward
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional, Callable


class ICSAttackEnvV2(gym.Env):
    """
    Gymnasium environment for adversarial training of DQN Red Agent.
    
    State space (60-dim):
      [51 sensors | detection_flag | deviation_magnitude | 5×rolling_errors | 2×last_attacked]
      
    Action space — Discrete(66300):
      - Equivalent to MultiDiscrete([51, 5, 52, 5]) joint actions, decoded internally:
        - sensor1_id (0–50) × perturbation_mag1 (5 levels)
        - sensor2_id (0–51, 51=no-op) × perturbation_mag2 (5 levels)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        blue_agent: Any,
        X_data: np.ndarray,  # (N, 60, 65) preprocessed sliding windows
        lambda1: float = 1.0,
        lambda2: float = 10.0,
        lambda3: float = 0.5,
        episode_length: int = 200,
        consecutive_detection_limit: int = 10,
        perturbation_levels: Optional[list] = None,
    ):
        super().__init__()
        
        self.blue_agent = blue_agent
        self.X_data = X_data
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.episode_length = episode_length
        self.consecutive_detection_limit = consecutive_detection_limit
        
        if perturbation_levels is None:
            self.perturbation_levels = [0.01, 0.03, 0.05, 0.08, 0.12]
        else:
            self.perturbation_levels = perturbation_levels
            
        # Action space: Discrete(66300) mapping to MultiDiscrete([51, 5, 52, 5])
        self.action_space = spaces.Discrete(66300)
        
        # State space: 60-dimensional box
        low = np.array([-np.inf] * 60, dtype=np.float32)
        high = np.array([np.inf] * 60, dtype=np.float32)
        
        # Sensor bounds are [0, 1]
        low[:51] = 0.0
        high[:51] = 1.0
        # Detection flag is binary
        low[51] = 0.0
        high[51] = 1.0
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.current_idx = 0
        self.step_count = 0
        self.consecutive_detections = 0
        
        self.original_readings = None
        self.current_readings = None
        self.rolling_errors = np.zeros(5, dtype=np.float32)
        self.last_attacked = [51.0, 51.0]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Choose a random window context
        if seed is not None:
            np.random.seed(seed)
        self.current_idx = np.random.randint(0, len(self.X_data))
        
        # Dynamically align dataset context in the Blue Agent's sub-agents
        if hasattr(self.blue_agent, "lstm_agent"):
            self.blue_agent.lstm_agent.X_test = self.X_data
        if hasattr(self.blue_agent, "gnn_agent"):
            self.blue_agent.gnn_agent.X_test_windows = self.X_data
        
        # Initialize defender context
        if hasattr(self.blue_agent, "set_context"):
            self.blue_agent.set_context(self.current_idx)
            
        # Get starting readings (the last step of the current window context)
        self.original_readings = self.X_data[self.current_idx, -1, :].copy()
        self.current_readings = self.original_readings.copy()
        
        self.step_count = 0
        self.consecutive_detections = 0
        self.rolling_errors = np.zeros(5, dtype=np.float32)
        self.last_attacked = [51.0, 51.0]
        
        state = self._get_state(detection_flag=0.0, deviation=0.0)
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        
        # Decode action in [0, 66299] to s1, mag1, s2, mag2
        # Size mapping: [51, 5, 52, 5]
        action = int(action)
        mag2 = action % 5
        temp = action // 5
        s2 = temp % 52
        temp = temp // 52
        mag1 = temp % 5
        s1 = temp // 5
        
        # Apply perturbation to sensor 1
        p_val1 = self.perturbation_levels[mag1]
        dir1 = 1.0 if self.original_readings[s1] < 0.5 else -1.0
        self.current_readings[s1] = np.clip(
            self.original_readings[s1] + dir1 * p_val1, 0.0, 1.0
        )
        
        # Apply perturbation to sensor 2 (if not no-op)
        if s2 < 51:
            p_val2 = self.perturbation_levels[mag2]
            dir2 = 1.0 if self.original_readings[s2] < 0.5 else -1.0
            self.current_readings[s2] = np.clip(
                self.original_readings[s2] + dir2 * p_val2, 0.0, 1.0
            )
            self.last_attacked = [float(s1), float(s2)]
        else:
            self.last_attacked = [float(s1), 51.0]
            
        # Query Blue Agent
        detected, score = self.blue_agent(self.current_readings)
        
        # Update rolling errors
        self.rolling_errors = np.roll(self.rolling_errors, -1)
        self.rolling_errors[-1] = score
        
        # Consecutive detections counter
        if detected:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 0
            
        # Compute deviation and reward
        deviation = np.sum(np.abs(self.current_readings - self.original_readings))
        
        # Equation: R = λ1 × Σ|δ| − λ2 × detected − λ3 × (Σ|δ|)²
        reward = (
            self.lambda1 * deviation 
            - self.lambda2 * (1.0 if detected else 0.0)
            - self.lambda3 * (deviation ** 2)
        )
        
        # Termination conditions
        terminated = self.consecutive_detections >= self.consecutive_detection_limit
        truncated = self.step_count >= self.episode_length
        
        state = self._get_state(
            detection_flag=1.0 if detected else 0.0,
            deviation=deviation
        )
        
        info = {
            "idx": self.current_idx,
            "detected": detected,
            "score": score,
            "deviation": deviation,
            "consecutive_detections": self.consecutive_detections,
            "action_decoded": (s1, mag1, s2, mag2)
        }
        
        return state, float(reward), terminated, truncated, info

    def _get_state(self, detection_flag: float, deviation: float) -> np.ndarray:
        state = np.zeros(60, dtype=np.float32)
        # 51 sensors
        state[:51] = self.current_readings[:51]
        # Blue Agent detection flag
        state[51] = detection_flag
        # Current deviation
        state[52] = deviation
        # 5 rolling reconstruction errors
        state[53:58] = self.rolling_errors
        # Last attacked sensors
        state[58:60] = self.last_attacked
        return state
