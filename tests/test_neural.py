"""
Unit Tests for PyTorch Deep Neural Network.
"""

import pytest
import numpy as np
import torch
from microquant.models.neural import MicroQuantMLP, PyTorchMLPTrainer


def test_microquant_mlp_architecture():
    input_dim = 18
    model = MicroQuantMLP(input_dim=input_dim)
    dummy_input = torch.randn(8, input_dim)
    output = model(dummy_input)

    assert output.shape == (8,)


def test_pytorch_mlp_trainer():
    np.random.seed(42)
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 2, 50)

    trainer = PyTorchMLPTrainer(input_dim=10, epochs=5, batch_size=16)
    trainer.fit(X, y)

    probs = trainer.predict_proba(X)
    assert probs.shape == (50, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)

    preds = trainer.predict(X)
    assert len(preds) == 50
    assert set(np.unique(preds)).issubset({0, 1})
