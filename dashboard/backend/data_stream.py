"""
data_stream.py
──────────────
SWaTStream sensor replay + live mode.
Handles loading test windows and replaying them under different modes.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

import sys
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class SWaTStream:
    def __init__(self):
        proof_dir = ROOT_DIR / "data" / "proof"
        self.X_test_clean = np.load(proof_dir / "X_test.npy")  # (N, 60, 65)
        
        # Inject standard anomalies for Mode 2 (SWaT Attack Mode)
        # Using the same seed 42 and ratio 5% from notebooks
        self.X_test_attack, self.y_test_attack = self._inject_anomalies(
            self.X_test_clean, ratio=0.05, seed=42
        )
        
        self.total_samples = len(self.X_test_clean)
        self.current_idx = 0

    def _inject_anomalies(self, X: np.ndarray, ratio: float = 0.05, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        X_aug = X.copy()
        n = len(X_aug)
        n_anom = int(n * ratio)
        anom_idx = rng.choice(n, size=n_anom, replace=False)
        
        # Perturb: add Gaussian noise 3x std of each feature
        for idx in anom_idx:
            # Perturb a random sensor column
            feature_idx = rng.integers(0, X_aug.shape[2])
            std_val = X_aug[:, :, feature_idx].std()
            if std_val == 0:
                std_val = 0.1
            X_aug[idx, -1, feature_idx] += rng.normal(0, 3.0 * std_val)
            X_aug[idx, -1, feature_idx] = np.clip(X_aug[idx, -1, feature_idx], 0.0, 1.0)
            
        y = np.zeros(n, dtype=int)
        y[anom_idx] = 1
        return X_aug, y

    def get_next_frame(self, mode: int) -> Dict[str, Any]:
        """
        Get the next snapshot window based on the requested mode:
          Mode 1: Normal (Clean SWaT Test Stream)
          Mode 2: SWaT Attack (Synthetic anomalous stream)
          Mode 3: DQN Attack (yields original clean window for Red Agent to perturb)
        """
        idx = self.current_idx
        
        if mode == 1:
            window = self.X_test_clean[idx]
            label = 0
        elif mode == 2:
            window = self.X_test_attack[idx]
            label = int(self.y_test_attack[idx])
        else: # Mode 3 (DQN Attack)
            window = self.X_test_clean[idx]
            label = 0  # To be perturbed by DQN
            
        # Increment index looping back if at end
        self.current_idx = (self.current_idx + 1) % self.total_samples
        
        return {
            "idx": idx,
            "window": window,
            "readings": window[-1].copy(),  # Current timestep readings
            "label": label
        }


# Singleton stream instance
stream = SWaTStream()
