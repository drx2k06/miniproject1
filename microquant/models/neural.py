"""
PyTorch Deep Neural Network for Microstructure Alpha Prediction.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from microquant.core.logging import get_logger

logger = get_logger("microquant.neural")


class MicroQuantMLP(nn.Module):
    """
    Multi-Layer Perceptron architecture for high-frequency microstructure classification.
    Incorporates Batch Normalization, LeakyReLU, and Dropout regularization.
    """

    def __init__(self, input_dim: int, hidden_dim1: int = 128, hidden_dim2: int = 64, hidden_dim3: int = 32, dropout_rate: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate * 0.8),

            nn.Linear(hidden_dim2, hidden_dim3),
            nn.LeakyReLU(0.1),

            nn.Linear(hidden_dim3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PyTorchMLPTrainer:
    """Trains and evaluates the PyTorch MicroQuantMLP model."""

    def __init__(self, input_dim: int, epochs: int = 35, batch_size: int = 32, lr: float = 0.001, seed: int = 42):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = MicroQuantMLP(input_dim)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Fits neural network with AdamW and validation tracking."""
        logger.info(f"Training PyTorch MLP on {len(X_train)} samples for {self.epochs} epochs...")
        self.model.train()

        tensor_x = torch.tensor(X_train, dtype=torch.float32)
        tensor_y = torch.tensor(y_train, dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for bx, by in train_loader:
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(bx)

            avg_loss = epoch_loss / len(X_train)

            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(torch.tensor(X_val, dtype=torch.float32))
                    val_loss = criterion(val_logits, torch.tensor(y_val, dtype=torch.float32)).item()
                scheduler.step(val_loss)
                self.model.train()

        self.model.eval()
        logger.info("PyTorch MLP training complete.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities [P(0), P(1)]."""
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32)
            logits = self.model(tensor_x)
            p1 = torch.sigmoid(logits).cpu().numpy()
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary classification label."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
