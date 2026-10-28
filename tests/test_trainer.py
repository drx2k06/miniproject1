"""
Unit Tests for Model Suite Trainer.
"""

import pytest
import numpy as np
from pathlib import Path
from microquant.pipeline.features import MicrostructureFeatureExtractor
from microquant.models.dataset import AlphaDatasetBuilder
from microquant.models.trainer import ModelSuiteTrainer


def test_model_suite_trainer(sample_clean_trades, tmp_path):
    fe = MicrostructureFeatureExtractor()
    df_feat = fe.extract(sample_clean_trades, dropna=True)

    builder = AlphaDatasetBuilder(horizon=5, train_ratio=0.7)
    _, splits = builder.build_tick_dataset(df_feat)

    trainer = ModelSuiteTrainer(model_dir=tmp_path, random_state=42)
    suite = trainer.train_and_evaluate(splits)

    metrics = suite["metrics"]
    assert "baseline_logistic" in metrics
    assert "random_forest" in metrics
    assert "gradient_boosting" in metrics
    assert "pytorch_mlp" in metrics

    for m_name, m_dict in metrics.items():
        assert "accuracy" in m_dict
        assert "f1_score" in m_dict
        assert "roc_auc" in m_dict
        assert 0.0 <= m_dict["accuracy"] <= 1.0

    # Ensure serialized artifacts exist in tmp_path
    assert (tmp_path / "baseline_logistic.joblib").exists()
    assert (tmp_path / "random_forest.joblib").exists()
    assert (tmp_path / "gradient_boosting.joblib").exists()
    assert (tmp_path / "pytorch_mlp.pt").exists()
    assert (tmp_path / "model_metrics.json").exists()
