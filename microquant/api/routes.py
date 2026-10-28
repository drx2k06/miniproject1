"""
FastAPI Route Definitions for MicroQuant Platform.
"""

import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from fastapi import APIRouter, HTTPException, status

from microquant.core.config import settings
from microquant.core.logging import get_logger
from microquant.pipeline.cleaner import OrderBookCleaner
from microquant.pipeline.features import MicrostructureFeatureExtractor
from microquant.pipeline.resampler import CandleResampler
from microquant.models.dataset import AlphaDatasetBuilder
from microquant.models.trainer import ModelSuiteTrainer
from microquant.models.backtester import MicrostructureBacktester
from microquant.api.schemas import (
    HealthResponse,
    OrderBookTickRequest,
    OrderBookPredictionResponse,
    PipelineRunResponse,
    BacktestRequest,
    BacktestResponse,
)

logger = get_logger("microquant.routes")
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health():
    """System health and operational status check."""
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
    )


@router.get("/stats")
def get_stats():
    """Returns dataset records, timestamps, and model registry statuses."""
    clean_exists = settings.CLEAN_DATA_PATH.exists()
    features_exists = settings.FEATURES_DATA_PATH.exists()
    metrics_file = settings.MODEL_DIR / "model_metrics.json"

    clean_count = 0
    time_span = "N/A"
    if clean_exists:
        df_c = pd.read_csv(settings.CLEAN_DATA_PATH, usecols=["timestamp"])
        clean_count = len(df_c)
        if clean_count > 0:
            time_span = f"{df_c['timestamp'].iloc[0]} to {df_c['timestamp'].iloc[-1]}"

    metrics_data = {}
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)

    return {
        "raw_data_path": str(settings.RAW_DATA_PATH),
        "clean_records": clean_count,
        "time_span": time_span,
        "models_trained": list(metrics_data.keys()),
        "features_computed": features_exists,
    }


@router.post("/pipeline/run", response_model=PipelineRunResponse)
def run_pipeline():
    """Triggers the full end-to-end data cleaning, feature extraction, resampling, and model training."""
    try:
        t0 = time.time()
        logger.info("Executing API-triggered pipeline run...")

        # 1. Clean
        cleaner = OrderBookCleaner()
        df_raw = pd.read_csv(settings.RAW_DATA_PATH, dtype=str)
        df_clean, report = cleaner.clean(df_raw)
        settings.ensure_directories()
        df_clean.to_csv(settings.CLEAN_DATA_PATH, index=False)

        # 2. Features
        fe = MicrostructureFeatureExtractor()
        df_feat = fe.extract(df_clean)
        df_feat.to_csv(settings.FEATURES_DATA_PATH, index=False)

        # 3. Resample
        resampler_1m = CandleResampler(time_frame="1min")
        c1m = resampler_1m.resample(df_clean)
        c1m.to_csv(settings.CANDLES_1MIN_PATH, index=False)

        resampler_5s = CandleResampler(time_frame="5s")
        c5s = resampler_5s.resample(df_clean)
        c5s.to_csv(settings.CANDLES_5S_PATH, index=False)

        # 4. Dataset
        builder = AlphaDatasetBuilder(horizon=settings.TICK_HORIZON, train_ratio=settings.TRAIN_TEST_SPLIT_RATIO)
        df_ml, splits = builder.build_tick_dataset(df_feat)
        df_ml.to_csv(settings.DATASET_ML_PATH, index=False)

        # 5. Trainer
        trainer = ModelSuiteTrainer(model_dir=settings.MODEL_DIR, random_state=settings.RANDOM_STATE)
        suite_res = trainer.train_and_evaluate(splits)

        elapsed = round(time.time() - t0, 2)
        return PipelineRunResponse(
            status="success",
            message=f"Pipeline completed successfully in {elapsed}s",
            cleaning_report=report.to_dict(),
            feature_count=len(df_feat),
            candles_1min_count=len(c1m),
            candles_5s_count=len(c5s),
            model_metrics=suite_res["metrics"],
        )
    except Exception as e:
        logger.exception("Pipeline run failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/predict/tick", response_model=OrderBookPredictionResponse)
def predict_tick(tick: OrderBookTickRequest):
    """Generates real-time microsecond alpha directional prediction for an incoming orderbook tick."""
    t0 = time.perf_counter()
    model_path = settings.MODEL_DIR / "gradient_boosting.joblib"
    scaler_path = settings.MODEL_DIR / "scaler.joblib"
    feature_names_path = settings.MODEL_DIR / "feature_names.json"

    if not model_path.exists() or not scaler_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Models not yet trained. Run /pipeline/run first.",
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)

    # Calculate real-time snapshot features
    spread = tick.ask_price_1 - tick.bid_price_1
    mid_price = (tick.ask_price_1 + tick.bid_price_1) / 2.0
    spread_bps = (spread / mid_price) * 10000.0
    order_imbalance_1 = (tick.bid_qty_1 - tick.ask_qty_1) / (tick.bid_qty_1 + tick.ask_qty_1 + 1e-9)

    bid2_qty = tick.bid_qty_2 if tick.bid_qty_2 is not None else 0.0
    bid3_qty = tick.bid_qty_3 if tick.bid_qty_3 is not None else 0.0
    ask2_qty = tick.ask_qty_2 if tick.ask_qty_2 is not None else 0.0
    ask3_qty = tick.ask_qty_3 if tick.ask_qty_3 is not None else 0.0

    total_bid = tick.bid_qty_1 + bid2_qty + bid3_qty
    total_ask = tick.ask_qty_1 + ask2_qty + ask3_qty
    total_depth = total_bid + total_ask
    depth_imbalance_3 = (total_bid - total_ask) / (total_depth + 1e-9)

    micro_price = (tick.bid_price_1 * tick.ask_qty_1 + tick.ask_price_1 * tick.bid_qty_1) / (tick.bid_qty_1 + tick.ask_qty_1 + 1e-9)
    micro_price_bias = (micro_price - mid_price) / (spread + 1e-9)

    w_bid = 0.5 * tick.bid_qty_1 + 0.3 * bid2_qty + 0.2 * bid3_qty
    w_ask = 0.5 * tick.ask_qty_1 + 0.3 * ask2_qty + 0.2 * ask3_qty
    weighted_imbalance = (w_bid - w_ask) / (w_bid + w_ask + 1e-9)

    bid_p3 = tick.bid_price_3 if tick.bid_price_3 is not None else tick.bid_price_1
    ask_p3 = tick.ask_price_3 if tick.ask_price_3 is not None else tick.ask_price_1
    bid_slope = (tick.bid_price_1 - bid_p3) / (total_bid + 1e-9)
    ask_slope = (ask_p3 - tick.ask_price_1) / (total_ask + 1e-9)
    slope_ratio = bid_slope / (ask_slope + 1e-9)

    signed_qty = tick.quantity if tick.side.lower() == "buy" else -tick.quantity
    effective_spread = 2.0 * abs(tick.price - mid_price)

    # Feature mapping dictionary
    feat_map = {
        "spread_bps": spread_bps,
        "order_imbalance_1": order_imbalance_1,
        "depth_imbalance_3": depth_imbalance_3,
        "weighted_imbalance": weighted_imbalance,
        "micro_price_bias": micro_price_bias,
        "bid_slope": bid_slope,
        "ask_slope": ask_slope,
        "slope_ratio": slope_ratio,
        "signed_quantity": signed_qty,
        "price_to_vwap_bps": 0.0,
        "effective_spread": effective_spread,
        "return_1": 0.0,
        "log_return_1": 0.0,
        "volatility_10": 0.0001,
        "volatility_20": 0.0001,
        "ma_ratio": 0.0,
        "rsi_14": 50.0,
        "bb_pct_b": 0.5,
    }

    x_vec = np.array([[feat_map.get(fn, 0.0) for fn in feature_names]])
    x_scaled = scaler.transform(x_vec)

    prob_up = float(model.predict_proba(x_scaled)[0, 1])
    predicted_dir = "UP" if prob_up >= 0.5 else "DOWN"
    confidence = abs(prob_up - 0.5) * 2.0 * 100.0
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return OrderBookPredictionResponse(
        symbol=tick.symbol,
        model_name="gradient_boosting",
        predicted_direction=predicted_dir,
        probability_up=round(prob_up, 4),
        confidence_pct=round(confidence, 2),
        spread_bps=round(spread_bps, 4),
        order_imbalance_1=round(order_imbalance_1, 4),
        micro_price=round(micro_price, 2),
        latency_ms=round(latency_ms, 2),
    )


@router.get("/models/metrics")
def get_model_metrics():
    """Retrieves saved test set evaluation metrics and feature rankings for all models."""
    metrics_path = settings.MODEL_DIR / "model_metrics.json"
    importances_path = settings.MODEL_DIR / "feature_importances.json"

    if not metrics_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No model metrics found.")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    importances = {}
    if importances_path.exists():
        with open(importances_path, "r") as f:
            importances = json.load(f)

    return {
        "metrics": metrics,
        "feature_importances": importances,
    }


@router.post("/backtest/run", response_model=BacktestResponse)
def run_backtest_endpoint(req: BacktestRequest):
    """Executes quantitative strategy simulation on the test partition with custom fee/slippage parameters."""
    dataset_path = settings.DATASET_ML_PATH
    model_path = settings.MODEL_DIR / f"{req.model_name}.joblib"
    scaler_path = settings.MODEL_DIR / "scaler.joblib"
    feature_names_path = settings.MODEL_DIR / "feature_names.json"

    if not dataset_path.exists() or not model_path.exists():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Models or dataset not initialized.")

    df_ml = pd.read_csv(dataset_path)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)

    # Prepare out-of-sample test split
    split_idx = int(len(df_ml) * settings.TRAIN_TEST_SPLIT_RATIO)
    test_df = df_ml.iloc[split_idx:].copy().reset_index(drop=True)

    X_test = scaler.transform(test_df[feature_names].values)
    probs = model.predict_proba(X_test)[:, 1]

    bt = MicrostructureBacktester(
        maker_fee_bps=req.maker_fee_bps,
        taker_fee_bps=req.taker_fee_bps,
        slippage_bps=req.slippage_bps,
        initial_capital=req.initial_capital,
        long_threshold=req.long_threshold,
        short_threshold=req.short_threshold,
    )

    metrics, hist = bt.run_backtest(
        probabilities=probs,
        prices=test_df["price"].values if "price" in test_df.columns else test_df["mid_price"].values,
        spreads=test_df["spread"].values if "spread" in test_df.columns else None,
        timestamps=test_df["timestamp"] if "timestamp" in test_df.columns else None,
    )

    # Downsample equity curve to max 100 points for response payload
    step = max(1, len(hist) // 100)
    sampled = hist.iloc[::step].copy()
    equity_curve_points = [
        {
            "timestamp": str(r["timestamp"]),
            "equity": round(float(r["equity"]), 2),
            "benchmark_equity": round(float(r["benchmark_equity"]), 2),
            "drawdown": round(float(r["drawdown"]) * 100.0, 2),
        }
        for _, r in sampled.iterrows()
    ]

    return BacktestResponse(
        model_name=req.model_name,
        initial_capital=metrics.initial_capital,
        final_equity=metrics.final_equity,
        total_return_pct=metrics.total_return_pct,
        benchmark_return_pct=metrics.benchmark_return_pct,
        alpha_pct=metrics.alpha_pct,
        sharpe_ratio=metrics.sharpe_ratio,
        sortino_ratio=metrics.sortino_ratio,
        max_drawdown_pct=metrics.max_drawdown_pct,
        total_trades=metrics.total_trades,
        win_rate_pct=metrics.win_rate_pct,
        profit_factor=metrics.profit_factor,
        total_fees_paid=metrics.total_fees_paid,
        total_slippage_paid=metrics.total_slippage_paid,
        equity_curve=equity_curve_points,
    )
