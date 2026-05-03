"""
main.py
───────
FastAPI application entry point for RAKSHAK-ICS dashboard backend.

Endpoints:
  GET  /health              — health check
  GET  /api/modes           — available demo modes
  WS   /ws/sensor-stream    — WebSocket live sensor data stream

Demo Modes:
  1. Normal     — replay normal SWaT sensor data
  2. SWaT Attack — replay labelled attack scenarios
  3. DQN Red Agent — live Red Agent perturbations vs Blue Agent

Deployment: HuggingFace Spaces (Docker)
"""
