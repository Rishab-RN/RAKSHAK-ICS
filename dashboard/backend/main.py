"""
main.py
───────
FastAPI web application entry point.
Establishes WebSocket stream endpoint and model loader cache initialization.
"""

import os
import json
import asyncio
import logging
import numpy as np
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


# Ensure parent directory is in path for imports
import sys
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.backend.model_loader import cache
from dashboard.backend.data_stream import stream
from dashboard.backend.inference import run_defender_inference, run_attacker_inference
from src.rl_env import ICSAttackEnvV2

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rakshak.backend")

app = FastAPI(title="RAKSHAK-ICS SCADA Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Load model cache at server startup."""
    logger.info("Initializing RAKSHAK-ICS backend and loading model weights...")
    cache.load_all()
    logger.info("FastAPI backend started successfully.")


@app.get("/health")
def health_check():
    """Health check endpoint for deployment validation."""
    return {
        "status": "ok",
        "models_loaded": cache.blue_agent is not None and cache.red_agent is not None,
        "features_count": len(cache.feature_names) if cache.feature_names else 0
    }


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint streaming live sensor data and model predictions.
    Handles user commands to switch modes and streaming speeds.
    """
    await websocket.accept()
    logger.info("WebSocket connection established.")
    
    # Default parameters
    mode = 1  # 1 = Normal, 2 = SWaT Attack, 3 = DQN Attack
    speed = 1.0  # Frequency in Hz (e.g. 1Hz, 5Hz, 10Hz)
    
    # Instantiate the DQN environment context for this session
    reward_w = cache.config["rl_env"]["reward_weights"]
    env = ICSAttackEnvV2(
        blue_agent=cache.blue_agent,
        X_data=stream.X_test_clean,
        lambda1=reward_w["lambda1"],
        lambda2=reward_w["lambda2"],
        lambda3=reward_w["lambda3"],
        episode_length=cache.config["dqn"]["episode_length"],
        consecutive_detection_limit=cache.config["dqn"]["consecutive_detection_limit"],
    )
    
    # Initialize DQN observation
    obs, _ = env.reset()
    
    # Task to handle incoming messages/commands from the client
    async def receive_commands():
        nonlocal mode, speed
        try:
            while True:
                data = await websocket.receive_text()
                cmd = json.loads(data)
                if "mode" in cmd:
                    new_mode = int(cmd["mode"])
                    if new_mode in [1, 2, 3]:
                        mode = new_mode
                        logger.info(f"Mode changed to: {mode}")
                        if mode == 3:
                            env.reset()
                if "speed" in cmd:
                    new_speed = float(cmd["speed"])
                    if 0.1 <= new_speed <= 10.0:
                        speed = new_speed
                        logger.info(f"Speed changed to: {speed} Hz")
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Error receiving WebSocket commands: {e}")
            
    # Spawn background receiver task
    cmd_task = asyncio.create_task(receive_commands())
    
    try:
        while True:
            # 1. Get next frame from data stream
            frame = stream.get_next_frame(mode)
            idx = frame["idx"]
            readings = frame["readings"]
            
            red_agent_info = None
            
            # 2. If DQN Attack mode, execute DQN policy step to perturb readings
            if mode == 3:
                # Synchronize environment with current window index
                env.current_idx = idx
                env.blue_agent.set_context(idx)
                
                # Fetch observation from current env readings
                obs = env._get_state(
                    detection_flag=1.0 if env.consecutive_detections > 0 else 0.0,
                    deviation=np.sum(np.abs(env.current_readings - env.original_readings))
                )
                
                # Predict action
                action, action_decoded = run_attacker_inference(obs)
                s1, mag1, s2, mag2 = action_decoded
                
                # Step environment
                obs, reward, term, trunc, info = env.step(action)
                
                # Apply perturbed readings to stream frame
                readings = env.current_readings.copy()
                
                red_agent_info = {
                    "sensor1": cache.feature_names[s1] if s1 < len(cache.feature_names) else f"Sensor_{s1}",
                    "mag1": float(env.perturbation_levels[mag1]),
                    "sensor2": cache.feature_names[s2] if s2 < len(cache.feature_names) else f"Sensor_{s2}",
                    "mag2": float(env.perturbation_levels[mag2]) if s2 < 51 else 0.0,
                    "reward": float(reward),
                    "caught": bool(term)
                }
                
                # Reset environment if terminated or truncated
                if term or trunc:
                    env.reset()
                    
            # 3. Execute Blue Agent defender inference on current readings
            inference_res = run_defender_inference(readings, idx)
            
            # Map readings array to dict with sensor names for the frontend
            readings_dict = {}
            for s_idx, f_name in enumerate(cache.feature_names):
                readings_dict[f_name] = float(readings[s_idx])
                
            # 4. Construct payload
            payload = {
                "idx": idx,
                "mode": mode,
                "readings": readings_dict,
                "anomaly_flag": inference_res["anomaly_flag"],
                "score": inference_res["score"],
                "threshold": float(cache.blue_agent.threshold),
                "top_sensors": inference_res["top_sensors"],
                "red_agent": red_agent_info
            }
            
            # Send payload
            await websocket.send_text(json.dumps(payload))
            
            # Sleep based on speed frequency (Hz)
            await asyncio.sleep(1.0 / speed)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        cmd_task.cancel()
        await websocket.close()
