"""
Integration Tests for FastAPI Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from microquant.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_stats_endpoint(client):
    res = client.get("/api/v1/stats")
    assert res.status_code == 200
    data = res.json()
    assert "clean_records" in data
    assert "models_trained" in data


def test_predict_tick_endpoint(client):
    payload = {
        "trade_id": "100999",
        "timestamp": "2026-10-01 09:20:00.000",
        "symbol": "BTCUSDT",
        "side": "buy",
        "price": 62100.0,
        "quantity": 1.5,
        "bid_price_1": 62099.5,
        "bid_qty_1": 2.5,
        "ask_price_1": 62100.5,
        "ask_qty_1": 1.2,
        "bid_price_2": 62099.0,
        "bid_qty_2": 3.0,
        "ask_price_2": 62101.0,
        "ask_qty_2": 2.0,
        "bid_price_3": 62098.5,
        "bid_qty_3": 4.0,
        "ask_price_3": 62101.5,
        "ask_qty_3": 3.0,
    }
    res = client.post("/api/v1/predict/tick", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_direction"] in ["UP", "DOWN"]
    assert 0.0 <= data["probability_up"] <= 1.0
    assert "latency_ms" in data


def test_model_metrics_endpoint(client):
    res = client.get("/api/v1/models/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "metrics" in data
    assert "feature_importances" in data


def test_backtest_endpoint(client):
    payload = {
        "maker_fee_bps": 1.0,
        "taker_fee_bps": 4.0,
        "slippage_bps": 1.5,
        "initial_capital": 100000.0,
        "long_threshold": 0.52,
        "short_threshold": 0.48,
        "model_name": "gradient_boosting",
    }
    res = client.post("/api/v1/backtest/run", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "total_return_pct" in data
    assert "sharpe_ratio" in data
    assert "equity_curve" in data
    assert len(data["equity_curve"]) > 0
