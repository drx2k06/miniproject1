"""
Aetheris-MicroQuant Command-Line Interface (CLI).
"""

import sys
import subprocess
from pathlib import Path
import click
import pandas as pd
import joblib

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from microquant.core.config import settings
from microquant.core.logging import get_logger
from microquant.pipeline.cleaner import OrderBookCleaner
from microquant.pipeline.features import MicrostructureFeatureExtractor
from microquant.pipeline.resampler import CandleResampler
from microquant.models.dataset import AlphaDatasetBuilder
from microquant.models.trainer import ModelSuiteTrainer
from microquant.models.backtester import MicrostructureBacktester

logger = get_logger("microquant.cli")


@click.group()
def main():
    """Aetheris-MicroQuant: Quantitative Order Book Alpha Platform."""
    pass


@main.command()
@click.option("--raw-path", type=click.Path(exists=True), default=None, help="Path to raw dirty CSV.")
@click.option("--out-path", type=click.Path(), default=None, help="Output path for cleaned CSV.")
def clean(raw_path, out_path):
    """Cleans dirty market microstructure order book and trade data."""
    raw_p = Path(raw_path) if raw_path else settings.RAW_DATA_PATH
    out_p = Path(out_path) if out_path else settings.CLEAN_DATA_PATH

    click.echo(f"[*] Cleaning raw orderbook dataset: {raw_p}")
    df_raw = pd.read_csv(raw_p, dtype=str)
    cleaner = OrderBookCleaner()
    df_clean, report = cleaner.clean(df_raw)

    settings.ensure_directories()
    df_clean.to_csv(out_p, index=False)
    click.echo(f"[+] Cleaned data saved to: {out_p}")
    click.echo(f"    Initial: {report.initial_rows} -> Clean: {report.final_rows} (Retained: {report.final_rows/report.initial_rows*100:.1f}%)")


@main.command()
@click.option("--in-path", type=click.Path(exists=True), default=None)
@click.option("--out-path", type=click.Path(), default=None)
def features(in_path, out_path):
    """Extracts depth-weighted orderbook signals, flow toxicity, and alphas."""
    in_p = Path(in_path) if in_path else settings.CLEAN_DATA_PATH
    out_p = Path(out_path) if out_path else settings.FEATURES_DATA_PATH

    click.echo(f"[*] Extracting microstructure features from: {in_p}")
    df_clean = pd.read_csv(in_p)
    fe = MicrostructureFeatureExtractor()
    df_feat = fe.extract(df_clean)

    settings.ensure_directories()
    df_feat.to_csv(out_p, index=False)
    click.echo(f"[+] Features saved to: {out_p} ({len(df_feat)} rows, {df_feat.shape[1]} columns)")


@main.command()
@click.option("--in-path", type=click.Path(exists=True), default=None)
@click.option("--tf", default="1min", help="Candle resolution (e.g. 1min, 5s, 15s).")
def resample(in_path, tf):
    """Resamples tick trades into time-series OHLCV candles with volume flow."""
    in_p = Path(in_path) if in_path else settings.CLEAN_DATA_PATH
    out_p = settings.DATA_DIR / "processed" / f"candles_{tf}.csv"

    click.echo(f"[*] Resampling {in_p} into '{tf}' candles...")
    df_clean = pd.read_csv(in_p)
    resampler = CandleResampler(time_frame=tf)
    candles = resampler.resample(df_clean)

    settings.ensure_directories()
    candles.to_csv(out_p, index=False)
    click.echo(f"[+] Resampled candles saved to: {out_p} ({len(candles)} candles)")


@main.command()
@click.option("--horizon", default=10, help="Forward prediction horizon in ticks.")
def train(horizon):
    """Trains Baseline, Random Forest, Gradient Boosting, and PyTorch MLP models."""
    click.echo(f"[*] Training Alpha Model Suite (Horizon = {horizon} ticks)...")
    df_feat = pd.read_csv(settings.FEATURES_DATA_PATH)

    builder = AlphaDatasetBuilder(horizon=horizon, train_ratio=settings.TRAIN_TEST_SPLIT_RATIO)
    df_ml, splits = builder.build_tick_dataset(df_feat)
    settings.ensure_directories()
    df_ml.to_csv(settings.DATASET_ML_PATH, index=False)

    trainer = ModelSuiteTrainer(model_dir=settings.MODEL_DIR, random_state=settings.RANDOM_STATE)
    res = trainer.train_and_evaluate(splits)

    click.echo("\n[+] Out-of-Sample Performance Summary:")
    for name, m in res["metrics"].items():
        click.echo(f"  * {name:20}: Accuracy={m['accuracy']*100:.2f}% | F1={m['f1_score']:.4f} | ROC-AUC={m['roc_auc']:.4f}")


@main.command()
@click.option("--taker-fee", default=4.0, help="Taker fee in basis points.")
@click.option("--slippage", default=1.5, help="Execution slippage in basis points.")
def backtest(taker_fee, slippage):
    """Executes quantitative strategy simulation on test partition."""
    click.echo(f"[*] Executing Microstructure Strategy Simulation (Fee={taker_fee} bps, Slip={slippage} bps)...")
    df_ml = pd.read_csv(settings.DATASET_ML_PATH)
    split_idx = int(len(df_ml) * settings.TRAIN_TEST_SPLIT_RATIO)
    test_df = df_ml.iloc[split_idx:].reset_index(drop=True)

    model = joblib.load(settings.MODEL_DIR / "gradient_boosting.joblib")
    scaler = joblib.load(settings.MODEL_DIR / "scaler.joblib")
    import json
    with open(settings.MODEL_DIR / "feature_names.json") as f:
        f_names = json.load(f)

    X_test = scaler.transform(test_df[f_names].values)
    probs = model.predict_proba(X_test)[:, 1]

    bt = MicrostructureBacktester(taker_fee_bps=taker_fee, slippage_bps=slippage)
    m, _ = bt.run_backtest(
        probabilities=probs,
        prices=test_df["price"].values,
        timestamps=test_df["timestamp"],
    )

    click.echo(f"\n[+] Strategy Results:")
    click.echo(f"  * Net Return      : {m.total_return_pct:.2f}%")
    click.echo(f"  * Benchmark Return: {m.benchmark_return_pct:.2f}%")
    click.echo(f"  * Strategy Alpha  : {m.alpha_pct:.2f}%")
    click.echo(f"  * Max Drawdown    : {m.max_drawdown_pct:.2f}%")
    click.echo(f"  * Total Trades    : {m.total_trades}")
    click.echo(f"  * Win Rate        : {m.win_rate_pct:.2f}%")


@main.command(name="run-all")
def run_all():
    """Executes the complete end-to-end pipeline in one automated pass."""
    click.echo("[*] Launching End-to-End Aetheris-MicroQuant Processing Pipeline...")
    ctx = click.get_current_context()
    ctx.invoke(clean)
    ctx.invoke(features)
    ctx.invoke(resample, tf="1min")
    ctx.invoke(resample, tf="5s")
    ctx.invoke(train, horizon=settings.TICK_HORIZON)
    ctx.invoke(backtest, taker_fee=4.0, slippage=1.5)
    click.echo("\n[SUCCESS] Complete Pipeline Execution Finished Successfully!")


@main.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000)
def serve(host, port):
    """Starts the FastAPI REST API Server."""
    import uvicorn
    click.echo(f"[*] Starting MicroQuant API server on http://{host}:{port} ...")
    uvicorn.run("microquant.api.app:app", host=host, port=port, reload=False)


@main.command()
@click.option("--port", default=8501)
def dashboard(port):
    """Starts the Streamlit Visual Analytics Dashboard."""
    dashboard_file = Path(__file__).parent / "ui" / "app.py"
    click.echo(f"[*] Starting Streamlit Dashboard on port {port}...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_file), "--server.port", str(port)])


if __name__ == "__main__":
    main()

