"""
inference.py
────────────
Real-time inference pipeline:
  sensor_window → Blue Agent → {anomaly_flag, score, top5_sensors}
  sensor_window → Red Agent  → {perturbed_values, evasion_flag}
"""

import numpy as np
import torch
from typing import Dict, Any, Tuple

from dashboard.backend.model_loader import cache


def run_defender_inference(readings: np.ndarray, idx: int) -> Dict[str, Any]:
    """
    Run FusionBlueAgent defender on the current sensor readings snapshot.
    
    Returns:
        Dict with anomaly_flag, score, and top-5 contributing sensors (attribution).
    """
    # 1. Update sub-agent window histories
    cache.blue_agent.set_context(idx)
    
    # 2. Evaluate fused detection
    detected, combined_score = cache.blue_agent(readings)
    
    # 3. Compute attribution (top-5 contributing sensors)
    # We can approximate this by taking the absolute reconstruction errors of the LSTM Autoencoder
    lstm_agent = cache.blue_agent.lstm_agent
    window = lstm_agent.X_test[idx].copy()
    window[-1, :] = readings
    
    with torch.no_grad():
        x_tensor = torch.FloatTensor(window).unsqueeze(0).to(lstm_agent.device)
        x_recon = lstm_agent.model(x_tensor)
        # Compute absolute reconstruction error at the last timestep
        recon_error = torch.abs(x_tensor[0, -1, :] - x_recon[0, -1, :]).cpu().numpy()
        
    # Get top-5 sensor indices
    top_indices = np.argsort(recon_error)[::-1][:5]
    top_sensors = []
    
    # Map indices to actual feature names
    for rank, s_idx in enumerate(top_indices):
        sensor_name = cache.feature_names[s_idx] if s_idx < len(cache.feature_names) else f"Sensor_{s_idx}"
        top_sensors.append({
            "name": sensor_name,
            "error": float(recon_error[s_idx]),
            "rank": rank + 1
        })
        
    return {
        "anomaly_flag": bool(detected),
        "score": float(combined_score),
        "top_sensors": top_sensors
    }


def run_attacker_inference(obs: np.ndarray) -> Tuple[int, Tuple[int, int, int, int]]:
    """
    Query the trained DQN Red Agent attacker using the environment observation state.
    
    Returns:
        Tuple of (action_index, decoded_action_tuple).
    """
    # Predict action from DQN model
    action, _ = cache.red_agent.predict(obs, deterministic=True)
    action = int(action)
    
    # Decode action in [0, 66299] to s1, mag1, s2, mag2
    mag2 = action % 5
    temp = action // 5
    s2 = temp % 52
    temp = temp // 52
    mag1 = temp % 5
    s1 = temp // 5
    
    return action, (s1, mag1, s2, mag2)
