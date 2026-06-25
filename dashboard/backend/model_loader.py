"""
model_loader.py
───────────────
Loads trained Blue Agent and Red Agent model weights at server startup.
Provides cached model instances for inference.py to use.
"""

import os
import json
import logging
import pickle
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from stable_baselines3 import DQN

# Ensure parent directory is in path for imports
import sys
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.lstm_ae import LSTMAutoencoder, LSTMBlueAgent
from src.gnn import SensorGAT, GATBlueAgent
from src.fusion import FusionBlueAgent

logger = logging.getLogger("rakshak.model_loader")


class ModelCache:
    def __init__(self):
        self.config: Optional[Dict[str, Any]] = None
        self.scaler: Optional[Any] = None
        self.blue_agent: Optional[FusionBlueAgent] = None
        self.red_agent: Optional[DQN] = None
        
        self.edge_index: Optional[np.ndarray] = None
        self.edge_weights: Optional[np.ndarray] = None
        self.feature_names: Optional[list] = None

    def load_all(self) -> None:
        """Load all configs, models, scalers, and graph parameters."""
        config_path = ROOT_DIR / "configs" / "default.yaml"
        logger.info(f"Loading config from: {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        models_dir = ROOT_DIR / Path(self.config["paths"]["models_dir"])
        proof_dir = ROOT_DIR / Path(self.config["data"]["proof_dir"])
        
        # Load scaler
        scaler_path = models_dir / "scaler.pkl"
        if not scaler_path.exists():
            scaler_path = proof_dir / "scaler.pkl"
        logger.info(f"Loading scaler from: {scaler_path}")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        # Load dataset stats / metadata
        stats_path = proof_dir / "dataset_stats.json"
        logger.info(f"Loading feature metadata from: {stats_path}")
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.feature_names = stats.get("feature_names", [])
        
        # Load GAT graph variables
        logger.info("Loading GAT graph topology...")
        self.edge_index = np.load(proof_dir / "edge_index.npy")
        self.edge_weights = np.load(proof_dir / "edge_weights.npy")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device for model loading: {device}")
        
        # 1. Load LSTM Autoencoder
        logger.info("Loading LSTM Autoencoder...")
        lstm_model = LSTMAutoencoder(input_dim=self.config["lstm_ae"]["input_dim"])
        lstm_model.load_state_dict(torch.load(models_dir / "lstm_ae.pt", map_location=device))
        lstm_model.eval()
        
        # 2. Load SensorGAT
        logger.info("Loading SensorGAT GNN...")
        gat_model = SensorGAT(
            node_feature_dim=self.config["gat"]["node_feature_dim"],
            hidden_dim=self.config["gat"]["hidden_dim"],
            num_heads=self.config["gat"]["num_heads"],
            dropout=self.config["gat"]["dropout"],
            edge_dim=self.config["gat"]["edge_dim"],
        )
        gat_model.load_state_dict(torch.load(models_dir / "gnn.pt", map_location=device))
        gat_model.eval()
        
        # 3. Load Fusion parameters
        with open(models_dir / "fusion_params.json", "r") as f:
            fusion_params = json.load(f)
        alpha = fusion_params["alpha"]
        threshold = fusion_params["threshold"]
        
        # Load test windows for sequence contexts
        X_test_windows = np.load(proof_dir / "X_test.npy")
        X_test_nf = np.load(proof_dir / "node_features_test.npy")
        
        # 4. Wrap sub-agents
        lstm_agent = LSTMBlueAgent(
            model=lstm_model,
            threshold=threshold,
            X_test=X_test_windows,
            device=device
        )
        
        gnn_agent = GATBlueAgent(
            model=gat_model,
            X_test_node_features=X_test_nf,
            edge_index=self.edge_index,
            edge_weights=self.edge_weights,
            threshold=threshold,
            X_test_windows=X_test_windows
        )
        
        # 5. Assemble FusionBlueAgent
        self.blue_agent = FusionBlueAgent(
            lstm_agent=lstm_agent,
            gnn_agent=gnn_agent,
            alpha=alpha,
            threshold=threshold,
            mode="weighted"
        )
        
        # 6. Load DQN Red Agent
        red_agent_path = models_dir / "red_agent" / "best_model.zip"
        if not red_agent_path.exists():
            red_agent_path = models_dir / "red_agent" / "dqn_red_agent_final.zip"
            
        logger.info(f"Loading DQN Red Agent from: {red_agent_path}")
        # Note: We load it without passing env at start; env will be set up during WebSocket sessions
        self.red_agent = DQN.load(red_agent_path, device=device)
        logger.info("All model artifacts successfully cached!")


# Singleton cache instance
cache = ModelCache()
