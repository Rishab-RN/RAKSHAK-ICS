"""
rl_train.py
───────────
DQN Red Agent training script using Stable-Baselines3.
"""

import os
import json
import logging
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from src.rl_env import ICSAttackEnvV2
from src.fusion import FusionBlueAgent
from src.lstm_ae import LSTMAutoencoder, LSTMBlueAgent
from src.gnn import SensorGAT, GATBlueAgent

logger = logging.getLogger("rakshak.rl_train")


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_blue_agent(config: Dict[str, Any], device: str = "cpu") -> FusionBlueAgent:
    """Load the pre-trained defender components and assemble the FusionBlueAgent."""
    models_dir = Path(config["paths"]["models_dir"])
    proof_dir = Path(config["data"]["proof_dir"])
    
    # 1. Load LSTM Autoencoder
    logger.info("Loading LSTM Autoencoder...")
    lstm_model = LSTMAutoencoder(input_dim=config["lstm_ae"]["input_dim"])
    lstm_model.load_state_dict(torch.load(models_dir / "lstm_ae.pt", map_location=device))
    lstm_model.eval()
    
    # 2. Load SensorGAT
    logger.info("Loading SensorGAT GNN...")
    gat_model = SensorGAT(
        node_feature_dim=config["gat"]["node_feature_dim"],
        hidden_dim=config["gat"]["hidden_dim"],
        num_heads=config["gat"]["num_heads"],
        dropout=config["gat"]["dropout"],
        edge_dim=config["gat"]["edge_dim"],
    )
    gat_model.load_state_dict(torch.load(models_dir / "gnn.pt", map_location=device))
    gat_model.eval()
    
    # 3. Load graph structures
    edge_index = np.load(proof_dir / "edge_index.npy")
    edge_weights = np.load(proof_dir / "edge_weights.npy")
    X_val_nf = np.load(proof_dir / "node_features_val.npy")
    X_val_windows = np.load(proof_dir / "X_val.npy")
    
    # 4. Load Fusion parameters
    with open(models_dir / "fusion_params.json", "r") as f:
        fusion_params = json.load(f)
    alpha = fusion_params["alpha"]
    threshold = fusion_params["threshold"]
    
    logger.info(f"Loaded Fusion parameters: alpha={alpha:.3f}, threshold={threshold:.4f}")
    
    # 5. Wrap sub-agents
    lstm_agent = LSTMBlueAgent(
        model=lstm_model,
        threshold=threshold,
        X_test=X_val_windows,
        device=device
    )
    
    gnn_agent = GATBlueAgent(
        model=gat_model,
        X_test_node_features=X_val_nf,
        edge_index=edge_index,
        edge_weights=edge_weights,
        threshold=threshold,
        X_test_windows=X_val_windows
    )
    
    # 6. Assemble Fused agent
    fused_agent = FusionBlueAgent(
        lstm_agent=lstm_agent,
        gnn_agent=gnn_agent,
        alpha=alpha,
        threshold=threshold,
        mode="weighted"
    )
    
    return fused_agent


def train_dqn(config_path: str = "configs/default.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    # Load Blue Agent
    blue_agent = build_blue_agent(config, device=device)
    
    # Load Train/Val data
    proof_dir = Path(config["data"]["proof_dir"])
    X_train = np.load(proof_dir / "X_train.npy")
    X_val = np.load(proof_dir / "X_val.npy")
    
    # Create gymnasium environments
    env_config = config["rl_env"]
    reward_w = env_config["reward_weights"]
    
    logger.info("Initializing ICSAttackEnvV2 training environment...")
    train_env = ICSAttackEnvV2(
        blue_agent=blue_agent,
        X_data=X_train,
        lambda1=reward_w["lambda1"],
        lambda2=reward_w["lambda2"],
        lambda3=reward_w["lambda3"],
        episode_length=config["dqn"]["episode_length"],
        consecutive_detection_limit=config["dqn"]["consecutive_detection_limit"],
    )
    
    logger.info("Initializing ICSAttackEnvV2 evaluation environment...")
    eval_env = ICSAttackEnvV2(
        blue_agent=blue_agent,
        X_data=X_val,
        lambda1=reward_w["lambda1"],
        lambda2=reward_w["lambda2"],
        lambda3=reward_w["lambda3"],
        episode_length=config["dqn"]["episode_length"],
        consecutive_detection_limit=config["dqn"]["consecutive_detection_limit"],
    )
    
    dqn_config = config["dqn"]
    
    # Define DQN policy network architecture
    policy_kwargs = dict(
        net_arch=dqn_config["net_arch"]
    )
    
    # Instantiate DQN Agent
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=float(dqn_config["learning_rate"]),
        buffer_size=int(dqn_config["buffer_size"]),
        batch_size=int(dqn_config["batch_size"]),
        gamma=float(dqn_config["gamma"]),
        exploration_fraction=float(dqn_config["exploration_fraction"]),
        exploration_final_eps=float(dqn_config["exploration_final_eps"]),
        target_update_interval=int(dqn_config["target_update_interval"]),
        tensorboard_log=os.path.join(config["paths"]["logs_dir"], "tb_dqn"),
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
    )
    
    # Configure Callbacks
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir / "red_agent"),
        log_path=config["paths"]["logs_dir"],
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=str(models_dir / "red_agent" / "checkpoints"),
        name_prefix="dqn_attack_env"
    )
    
    # Start training
    total_timesteps = int(dqn_config["total_timesteps"])
    logger.info(f"Starting DQN training for {total_timesteps} steps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback]
    )
    
    # Save final model
    final_model_path = models_dir / "red_agent" / "dqn_red_agent_final"
    model.save(final_model_path)
    logger.info(f"Saved final DQN model to: {final_model_path}")


if __name__ == "__main__":
    train_dqn()
