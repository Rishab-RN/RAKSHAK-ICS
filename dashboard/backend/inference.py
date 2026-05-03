"""
inference.py
────────────
Real-time inference pipeline:
  sensor_window → Blue Agent → {anomaly_flag, score, top5_sensors}
  sensor_window → Red Agent  → {perturbed_values, evasion_flag}
"""
