"""
Alpha Model Training, Evaluation, and Serialization Engine.
Evaluates multiple machine learning algorithms with out-of-sample metrics.
"""

from typing import Dict, Any, List
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
)
import joblib
import torch

from microquant.core.logging import get_logger
from microquant.models.dataset import DatasetSplits
from microquant.models.neural import PyTorchMLPTrainer

logger = get_logger("microquant.trainer")


class ModelSuiteTrainer:
    """
    Trains, validates, and serializes a suite of competitive quantitative models.
    """

    def __init__(self, model_dir: Path, random_state: int = 42):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

    def train_and_evaluate(self, splits: DatasetSplits) -> Dict[str, Any]:
        """
        Trains Baseline, Random Forest, Gradient Boosting, and PyTorch MLP models,
        evaluates them on out-of-sample test splits, and persists models to disk.
        """
        logger.info("Starting Multi-Model Training and Evaluation Suite...")
        X_train, y_train = splits.X_train, splits.y_train
        X_test, y_test = splits.X_test, splits.y_test
        feature_names = splits.feature_names

        results: Dict[str, Any] = {}
        trained_models: Dict[str, Any] = {}

        # 1. Baseline: Logistic Regression
        logger.info("Training 1/4: Baseline Regularized Logistic Regression...")
        log_reg = LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state)
        log_reg.fit(X_train, y_train)
        trained_models["baseline_logistic"] = log_reg

        # 2. Random Forest Classifier
        logger.info("Training 2/4: Random Forest Classifier...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=4,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        trained_models["random_forest"] = rf

        # 3. Gradient Boosting Classifier
        logger.info("Training 3/4: Gradient Boosting Classifier...")
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=self.random_state,
        )
        gb.fit(X_train, y_train)
        trained_models["gradient_boosting"] = gb

        # 4. PyTorch Deep MLP
        logger.info("Training 4/4: PyTorch Deep Multi-Layer Perceptron (MLP)...")
        mlp = PyTorchMLPTrainer(
            input_dim=len(feature_names),
            epochs=35,
            batch_size=32,
            lr=0.001,
            seed=self.random_state,
        )
        mlp.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        trained_models["pytorch_mlp"] = mlp

        # Evaluate all models
        for name, model in trained_models.items():
            logger.info(f"Evaluating {name} on out-of-sample test set ({len(X_test)} samples)...")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            try:
                auc = float(roc_auc_score(y_test, y_prob))
            except Exception:
                auc = 0.5
            loss = float(log_loss(y_test, y_prob))
            brier = float(brier_score_loss(y_test, y_prob))
            cm = confusion_matrix(y_test, y_pred).tolist()

            results[name] = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(auc, 4),
                "log_loss": round(loss, 4),
                "brier_score": round(brier, 4),
                "confusion_matrix": cm,
            }
            logger.info(f"[{name}] Acc={acc:.4f} | F1={f1:.4f} | ROC-AUC={auc:.4f} | LogLoss={loss:.4f}")

        # Extract feature importances
        feature_importance_dict = {}
        if hasattr(rf, "feature_importances_"):
            rf_importances = dict(zip(feature_names, [round(float(v), 5) for v in rf.feature_importances_]))
            feature_importance_dict["random_forest"] = sorted(
                rf_importances.items(), key=lambda x: x[1], reverse=True
            )

        if hasattr(gb, "feature_importances_"):
            gb_importances = dict(zip(feature_names, [round(float(v), 5) for v in gb.feature_importances_]))
            feature_importance_dict["gradient_boosting"] = sorted(
                gb_importances.items(), key=lambda x: x[1], reverse=True
            )

        # Save artifacts
        logger.info(f"Serializing trained models and metadata to {self.model_dir}...")
        joblib.dump(trained_models["baseline_logistic"], self.model_dir / "baseline_logistic.joblib")
        joblib.dump(trained_models["random_forest"], self.model_dir / "random_forest.joblib")
        joblib.dump(trained_models["gradient_boosting"], self.model_dir / "gradient_boosting.joblib")
        torch.save(trained_models["pytorch_mlp"].model.state_dict(), self.model_dir / "pytorch_mlp.pt")
        joblib.dump(splits.scaler, self.model_dir / "scaler.joblib")

        with open(self.model_dir / "model_metrics.json", "w") as f:
            json.dump(results, f, indent=2)

        with open(self.model_dir / "feature_importances.json", "w") as f:
            json.dump(feature_importance_dict, f, indent=2)

        with open(self.model_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f, indent=2)

        logger.info("Model artifacts successfully saved.")
        return {
            "metrics": results,
            "feature_importances": feature_importance_dict,
            "models": trained_models,
        }
