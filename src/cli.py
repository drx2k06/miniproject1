"""
MicroQuant High-Frequency Market Microstructure Command-Line Interface.

Author: Daniel Richard Xavier
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MicroQuant.CLI")

from src.data_loader import DataLoader
from src.cleaner import DataCleaner
from src.features import MicrostructureFeatureExtractor
from src.resampler import OHLCVResampler
from src.analysis import StatisticalAnalyzer
from src.visualization import VisualizationSuite


def run_pipeline(
    raw_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    processed_dir: Optional[str] = None,
):
    """Executes the full end-to-end data audit, cleaning, feature, and analysis workflow."""
    logger.info("=" * 70)
    logger.info("STARTING MICROQUANT END-TO-END MARKET MICROSTRUCTURE PIPELINE")
    logger.info("Author: Daniel Richard Xavier")
    logger.info("=" * 70)

    project_root = Path(__file__).resolve().parent.parent

    raw_file = Path(raw_path) if raw_path else project_root / "data" / "raw" / "trades_orderbook_STUDENT_dirty.csv"
    proc_path = Path(processed_dir) if processed_dir else project_root / "data" / "processed"
    out_path = Path(output_dir) if output_dir else project_root / "outputs"
    fig_path = out_path / "figures"
    rep_path = out_path / "reports"

    proc_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    # 1. Loading & Validation
    loader = DataLoader(raw_file)
    df_raw = loader.load_raw_data()
    val_report = loader.validate_schema(df_raw)
    logger.info(f"Schema Validation Passed: {val_report.is_valid}. Total records: {val_report.total_rows}")

    # 2. Cleaning & Sanitization
    cleaner = DataCleaner(max_mid_gap_pct=0.02)
    df_clean, audit = cleaner.clean(df_raw)
    clean_csv_path = proc_path / "trades_clean.csv"
    loader.save_dataset(df_clean, clean_csv_path)
    logger.info(f"Cleaned records saved to {clean_csv_path} (Retained: {audit.retention_rate_pct}%)")

    # 3. Microstructure Feature Engineering
    extractor = MicrostructureFeatureExtractor(rsi_period=14, vol_window=20)
    df_features = extractor.extract(df_clean, dropna=True)
    feat_csv_path = proc_path / "features_tick.csv"
    loader.save_dataset(df_features, feat_csv_path)
    logger.info(f"Features extracted: {df_features.shape[1]} columns saved to {feat_csv_path}")

    # 4. Multi-Timeframe OHLCV Resampling
    resampler = OHLCVResampler()
    df_candles_1m = resampler.resample(df_clean, timeframe="1min")
    candles_1m_path = proc_path / "candles_1min.csv"
    loader.save_dataset(df_candles_1m, candles_1m_path)

    df_candles_5s = resampler.resample(df_clean, timeframe="5s")
    candles_5s_path = proc_path / "candles_5s.csv"
    loader.save_dataset(df_candles_5s, candles_5s_path)
    logger.info("Multi-resolution candles generated (1min and 5s).")

    # 5. Econometric & Statistical Tests
    analyzer = StatisticalAnalyzer()
    logger.info("Executing econometric stationarity and microstructure impact tests...")

    adf_price = analyzer.adf_test(df_features["price"], "Price Level")
    adf_ret = analyzer.adf_test(df_features["return_1"], "Arithmetic Return")
    adf_spread = analyzer.adf_test(df_features["spread_bps"], "Spread (bps)")
    adf_obi = analyzer.adf_test(df_features["order_imbalance_1"], "Order Imbalance (L1)")

    kyle_res = analyzer.kyle_lambda_impact(df_features)
    micro_res = analyzer.micro_price_lead_lag(df_features)
    vol_autocorr = analyzer.volatility_clustering_autocorr(df_features)
    dist_ret = analyzer.distribution_analysis(df_features["return_1"], "Returns")
    dist_qty = analyzer.distribution_analysis(df_clean["quantity"], "Trade Size")

    stats_summary = {
        "adf_tests": [
            adf_price.to_dict(),
            adf_ret.to_dict(),
            adf_spread.to_dict(),
            adf_obi.to_dict(),
        ],
        "kyle_lambda": kyle_res.to_dict(),
        "micro_price_lead_lag": micro_res.to_dict(),
        "volatility_clustering": vol_autocorr,
        "distribution_metrics": {
            "returns": dist_ret.to_dict(),
            "trade_quantity": dist_qty.to_dict(),
        },
    }

    def np_encoder(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (pd.Timestamp, np.datetime64)):
            return str(obj)
        return str(obj)

    stats_rep_file = rep_path / "statistical_summary.json"
    with open(stats_rep_file, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2, default=np_encoder)

    audit_rep_file = rep_path / "data_audit.json"
    with open(audit_rep_file, "w", encoding="utf-8") as f:
        json.dump({
            "validation_report": val_report.to_dict(),
            "cleaning_audit": audit.to_dict(),
        }, f, indent=2, default=np_encoder)

    logger.info(f"Statistical and audit reports persisted to {rep_path}.")

    # 6. Visualization Suite Generation
    viz = VisualizationSuite(fig_path)
    logger.info("Rendering publication-quality visualization suite...")

    viz.plot_data_quality_audit(audit.to_dict(), df_raw)
    viz.plot_candlestick_and_volume(df_candles_1m)
    viz.plot_order_book_depth_ladder(df_clean)
    viz.plot_spread_and_volatility(df_features)
    viz.plot_micro_price_lead_lag(df_features)
    viz.plot_kyle_lambda(df_features)
    viz.plot_feature_correlation_matrix(df_features)
    viz.plot_trade_distribution_qq(df_clean)

    logger.info("All 8 publication-grade figures successfully generated.")

    # 7. ML Target Generation (Zero-Leakage Forward Horizon on Tick Features)
    df_ml = df_features.copy()
    future_mid = df_ml["mid_price"].shift(-10)
    df_ml["target"] = (future_mid > df_ml["mid_price"]).astype(float)
    df_ml.loc[future_mid.isna(), "target"] = float("nan")
    df_ml = df_ml.dropna().reset_index(drop=True)
    ml_path = proc_path / "dataset_ml_ticks.csv"
    loader.save_dataset(df_ml, ml_path)
    logger.info(f"ML dataset with forward horizon saved to {ml_path}.")

    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION SUCCESSFULLY COMPLETED!")
    logger.info(f"Processed datasets: {proc_path}")
    logger.info(f"Visual figures:     {fig_path}")
    logger.info(f"Audits & Reports:   {rep_path}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="MicroQuant CLI Pipeline Runner")
    parser.add_argument("--run-all", action="store_true", help="Execute complete analysis pipeline")
    parser.add_argument("--raw-path", type=str, default="data/raw/trades_orderbook_STUDENT_dirty.csv", help="Path to raw CSV")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for reports/figures")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Directory for processed CSVs")

    args = parser.parse_args()
    run_pipeline(raw_path=args.raw_path, output_dir=args.output_dir, processed_dir=args.processed_dir)


if __name__ == "__main__":
    main()
