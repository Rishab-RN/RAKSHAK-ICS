"""
lstm_ae.py
──────────
LSTM Autoencoder for time-series anomaly detection (Blue Agent — Stream 1).

Architecture:
  Encoder: Linear(65→64) → LSTM(64→32, 2 layers, dropout=0.1) → Linear(32→16) → z
  Decoder: Linear(16→32) → LSTM(32→64, 2 layers, dropout=0.1) → Linear(64→65)

Training:
  - Normal-only data (SWaT A9 clean dataset)
  - MSE reconstruction loss
  - Anomaly score = per-sample MSE (higher = more anomalous)
  - Threshold τ set at 95th percentile of validation reconstruction error
  - Early stopping: patience=10, ReduceLROnPlateau(factor=0.5, patience=5)
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("rakshak.lstm_ae")


# ══════════════════════════════════════════════════════════════════════
#  Model Architecture
# ══════════════════════════════════════════════════════════════════════

class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for multi-variate time-series anomaly detection.

    Architecture:
      Encoder: Linear(65→64) -> LSTM(64→32, 2 layers, dropout=0.1) -> Linear(32→16) -> z
      Decoder: Linear(16→32) -> Repeat(seq_len) -> LSTM(32→64, 2 layers, dropout=0.1) -> Linear(64→65)
    """

    def __init__(
        self,
        input_dim: int = 65,
        sequence_length: int = 60,
        encoder_hidden: int = 64,
        latent_dim: int = 32,
        bottleneck_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.encoder_hidden = encoder_hidden
        self.latent_dim = latent_dim
        self.bottleneck_dim = bottleneck_dim

        # Encoder
        self.enc_proj = nn.Linear(input_dim, encoder_hidden)
        self.enc_lstm = nn.LSTM(
            input_size=encoder_hidden,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.enc_bottleneck = nn.Linear(latent_dim, bottleneck_dim)

        # Decoder
        self.dec_bottleneck = nn.Linear(bottleneck_dim, latent_dim)
        self.dec_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=encoder_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dec_proj = nn.Linear(encoder_hidden, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Reconstructed tensor of shape (batch_size, seq_len, input_dim).
        """
        batch_size, seq_len, _ = x.size()

        # Encoder
        x_proj = self.enc_proj(x)  # (batch, seq_len, encoder_hidden)
        out_enc, _ = self.enc_lstm(x_proj)  # (batch, seq_len, latent_dim)
        z_enc = out_enc[:, -1, :]  # (batch, latent_dim) - take last timestep
        z = self.enc_bottleneck(z_enc)  # (batch, bottleneck_dim)

        # Decoder
        z_dec = self.dec_bottleneck(z)  # (batch, latent_dim)
        z_dec_seq = z_dec.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, latent_dim)
        out_dec, _ = self.dec_lstm(z_dec_seq)  # (batch, seq_len, encoder_hidden)
        x_recon = self.dec_proj(out_dec)  # (batch, seq_len, input_dim)

        # Sigmoid output constraints reconstruction range to [0, 1]
        x_recon = torch.sigmoid(x_recon)

        return x_recon


# ══════════════════════════════════════════════════════════════════════
#  Training Harness
# ══════════════════════════════════════════════════════════════════════

def fit_lstm_ae(
    model: nn.Module,
    train_data: np.ndarray,
    val_data: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    patience: int = 10,
    device: str = "cpu",
    scheduler_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[float]]:
    """Train the LSTM Autoencoder model using normal-only data.

    Args:
        model: PyTorch model.
        train_data: Numpy array of shape (N, seq_len, features) with normal data.
        val_data: Numpy array of shape (N_val, seq_len, features) with normal data.
        epochs: Max number of training epochs.
        batch_size: DataLoader batch size.
        learning_rate: Initial learning rate.
        patience: Epochs patience for early stopping.
        device: Device to train on.
        scheduler_config: Config for learning rate scheduler.

    Returns:
        Dict containing lists of 'train_loss' and 'val_loss' histories.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(train_data))
    val_dataset = TensorDataset(torch.FloatTensor(val_data))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Learning rate scheduler
    scheduler = None
    if scheduler_config and scheduler_config.get("type") == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 5),
        )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    t0 = time.time()
    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            recon_x = model(batch_x)
            loss = criterion(recon_x, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        # ---- Val ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                recon_x = model(batch_x)
                loss = criterion(recon_x, batch_x)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch+1:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

        if scheduler is not None:
            scheduler.step(val_loss)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}!")
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    logger.info(f"Training finished in {time.time() - t0:.2f}s. Best Val Loss: {best_val_loss:.6f}")
    return history


# ══════════════════════════════════════════════════════════════════════
#  Scoring & Inference
# ══════════════════════════════════════════════════════════════════════

def compute_reconstruction_scores(
    model: nn.Module,
    data: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reconstruction errors (scores) and per-sensor errors for each sample.

    Args:
        model: Trained PyTorch LSTM-AE model.
        data: Input data array of shape (N, seq_len, features).
        batch_size: DataLoader batch size.
        device: Device to evaluate on.

    Returns:
        scores: 1D array of shape (N,) containing overall sample MSE.
        sensor_errors: 2D array of shape (N, features) containing per-sensor MSE.
    """
    model.eval()
    model.to(device)

    dataset = TensorDataset(torch.FloatTensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_scores = []
    all_sensor_errors = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            recon_x = model(batch_x)

            # Compute squared errors: (batch, seq_len, features)
            se = (batch_x - recon_x) ** 2

            # Overall sample score (mean over seq_len and features)
            scores = torch.mean(se, dim=(1, 2)).cpu().numpy()
            all_scores.append(scores)

            # Per-sensor error (mean over seq_len, keeping features dimension)
            sensor_err = torch.mean(se, dim=1).cpu().numpy()
            all_sensor_errors.append(sensor_err)

    return (
        np.concatenate(all_scores),
        np.concatenate(all_sensor_errors)
    )


# ══════════════════════════════════════════════════════════════════════
#  Blue Agent Attacker Wrapper
# ══════════════════════════════════════════════════════════════════════

class LSTMBlueAgent:
    """Stateful Blue Agent wrapper around LSTM-AE for attacker search evaluation.

    Satisfies the expected function signature of attackers:
      blue_agent_fn(readings) -> (detected, score)
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float,
        X_test: Optional[np.ndarray] = None,
        device: str = "cpu",
    ):
        """Initialize LSTMBlueAgent.

        Args:
            model: Trained PyTorch LSTM-AE model.
            threshold: Anomaly score detection threshold.
            X_test: Optional full test set tensor (N, seq_len, features).
            device: Device to run evaluation on.
        """
        self.model = model
        self.threshold = threshold
        self.X_test = X_test
        self.device = device
        self.current_idx: Optional[int] = None
        self.model.eval()
        self.model.to(self.device)

    def set_context(self, idx: int) -> None:
        """Set the active test sample index for sequence context construction."""
        self.current_idx = idx

    def __call__(self, readings: np.ndarray) -> Tuple[bool, float]:
        """Evaluate a single timestep of readings (possibly perturbed).

        Forms a 60-step window by replacing the last step of the test window
        at current_idx with the perturbed readings, computes reconstruction MSE.

        Args:
            readings: 65-dimensional numpy array of sensor readings.

        Returns:
            Tuple of (detected, score).
        """
        if self.X_test is None or self.current_idx is None:
            # Fallback to simple dev-from-mean if no window context is set
            score = float(np.mean(np.abs(readings - 0.5)))
            return (score > self.threshold, score)

        # Form the window: take the history from X_test[self.current_idx]
        window = self.X_test[self.current_idx].copy()
        # Replace the last timestep with the perturbed readings
        window[-1, :] = readings

        # Convert to tensor and run model
        with torch.no_grad():
            x_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)  # (1, seq_len, 65)
            x_recon = self.model(x_tensor)
            # Compute MSE score
            mse = torch.mean((x_tensor - x_recon) ** 2).item()

        return (mse > self.threshold, mse)
