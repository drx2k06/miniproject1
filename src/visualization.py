"""
Publication-Grade Market Microstructure Visualization Suite.

Author: Daniel Richard Xavier
"""

from pathlib import Path
from typing import Optional, Union, List
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)


class VisualizationSuite:
    """Produces standardized, publication-quality graphics for market microstructure research."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._apply_style()

    @staticmethod
    def _apply_style():
        """Configures clean, modern aesthetic with high contrast and accessibility."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.8,
            "grid.color": "#ebebeb",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "figure.autolayout": True,
            "figure.dpi": 300,
        })

    def plot_data_quality_audit(
        self,
        audit_dict: dict,
        df_raw: pd.DataFrame,
        filename: str = "01_data_quality_audit.png"
    ) -> Path:
        """Visualizes data cleaning stages and raw anomaly composition."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # 1. Missingness per column
        null_s = df_raw.isnull().sum()
        null_s = null_s[null_s > 0].sort_values(ascending=True)
        if len(null_s) > 0:
            colors = sns.color_palette("mako", len(null_s))
            ax1.barh(null_s.index, null_s.values, color=colors, edgecolor="none", height=0.6)
            ax1.set_title("Missing Value Counts in Raw Dataset", fontsize=12, fontweight="bold", pad=10)
            ax1.set_xlabel("Missing Records Count")
            for i, v in enumerate(null_s.values):
                ax1.text(v + 0.5, i, f"{v} ({v/len(df_raw)*100:.1f}%)", va="center", fontsize=9)
        else:
            ax1.text(0.5, 0.5, "No Missing Values", ha="center", va="center")

        # 2. Pipeline Filtering Waterfall / Breakdown
        categories = [
            "Initial Rows",
            "Duplicate Rows",
            "Null Quotes",
            "Non-Positive",
            "Crossed Books",
            "100x Scale Fix",
            "Final Cleaned"
        ]
        values = [
            audit_dict.get("initial_records", len(df_raw)),
            -audit_dict.get("duplicate_rows_removed", 35),
            -audit_dict.get("null_quotes_dropped", 15),
            -audit_dict.get("non_positive_values_dropped", 12),
            -audit_dict.get("crossed_books_dropped", 10),
            audit_dict.get("scale_anomalies_corrected", 8),
            audit_dict.get("final_clean_records", 1970)
        ]

        bar_colors = ["#2b5c8f" if v > 0 and i in [0, 6] else "#d9534f" if v < 0 else "#f0ad4e" for i, v in enumerate(values)]
        bars = ax2.bar(categories, [abs(v) for v in values], color=bar_colors, width=0.55)
        ax2.set_title("Data Cleaning Filter Cascade & Remediation", fontsize=12, fontweight="bold", pad=10)
        ax2.set_ylabel("Record Volume")
        ax2.tick_params(axis="x", rotation=30)
        for bar, v in zip(bars, values):
            label = f"{v:+d}" if v < 0 else f"{v}"
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15, label, ha="center", fontsize=8.5, fontweight="bold")

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {out_path}")
        return out_path

    def plot_candlestick_and_volume(
        self,
        df_candles: pd.DataFrame,
        filename: str = "02_candlestick_and_volume.png"
    ) -> Path:
        """Renders OHLC candlesticks with volume breakdown and moving averages."""
        fig, (ax_price, ax_vol) = plt.subplots(
            2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        df = df_candles.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        dates = df["timestamp"]
        # Candlestick plotting
        for i, row in df.iterrows():
            t = row["timestamp"]
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            color = "#2ca02c" if c >= o else "#d62728"
            
            # High-Low Wick
            ax_price.plot([t, t], [l, h], color=color, linewidth=1.2, alpha=0.8)
            # Open-Close Body (using small interval width)
            delta = pd.Timedelta(seconds=20)
            ax_price.fill_between([t - delta, t + delta], min(o, c), max(o, c), color=color, alpha=0.9)

        # Moving Averages
        if len(df) >= 5:
            df["ma_5"] = df["close"].rolling(5).mean()
            ax_price.plot(df["timestamp"], df["ma_5"], label="SMA 5", color="#1f77b4", linewidth=1.5)
        if len(df) >= 10:
            df["ma_10"] = df["close"].rolling(10).mean()
            ax_price.plot(df["timestamp"], df["ma_10"], label="SMA 10", color="#ff7f0e", linewidth=1.5, linestyle="--")

        ax_price.set_title("BTC/USDT 1-Minute OHLCV Candlestick Dynamics", fontsize=14, fontweight="bold", pad=12)
        ax_price.set_ylabel("Price (USDT)", fontsize=11)
        ax_price.legend(loc="upper left", frameon=True)

        # Volume bars (Buy volume green, Sell volume red)
        if "buy_volume" in df.columns and "sell_volume" in df.columns:
            ax_vol.bar(dates, df["buy_volume"], width=pd.Timedelta(seconds=35), color="#2ca02c", alpha=0.7, label="Buy Vol")
            ax_vol.bar(dates, df["sell_volume"], bottom=df["buy_volume"], width=pd.Timedelta(seconds=35), color="#d62728", alpha=0.7, label="Sell Vol")
            ax_vol.legend(loc="upper right", frameon=True)
        else:
            ax_vol.bar(dates, df["volume"], width=pd.Timedelta(seconds=35), color="#4a7bb0", alpha=0.7)

        ax_vol.set_ylabel("Volume (BTC)", fontsize=11)
        ax_vol.set_xlabel("Time (UTC)", fontsize=11)
        ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {out_path}")
        return out_path

    def plot_order_book_depth_ladder(
        self,
        df_clean: pd.DataFrame,
        filename: str = "03_order_book_depth_ladder.png"
    ) -> Path:
        """Visualizes Level-1 to Level-3 cumulative depth ladder for a snapshot."""
        sample_row = df_clean.iloc[len(df_clean) // 2]
        
        bids = [
            (sample_row["bid_price_1"], sample_row["bid_qty_1"]),
            (sample_row["bid_price_2"], sample_row["bid_qty_2"]),
            (sample_row["bid_price_3"], sample_row["bid_qty_3"]),
        ]
        asks = [
            (sample_row["ask_price_1"], sample_row["ask_qty_1"]),
            (sample_row["ask_price_2"], sample_row["ask_qty_2"]),
            (sample_row["ask_price_3"], sample_row["ask_qty_3"]),
        ]
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])

        bid_p = [b[0] for b in bids]
        bid_q = np.cumsum([b[1] for b in bids])

        ask_p = [a[0] for a in asks]
        ask_q = np.cumsum([a[1] for a in asks])

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.step(bid_p, bid_q, where="post", label="Cumulative Bid Depth", color="#2ca02c", linewidth=2.5)
        ax.fill_between(bid_p, bid_q, step="post", alpha=0.25, color="#2ca02c")

        ax.step(ask_p, ask_q, where="pre", label="Cumulative Ask Depth", color="#d62728", linewidth=2.5)
        ax.fill_between(ask_p, ask_q, step="pre", alpha=0.25, color="#d62728")

        mid = (sample_row["bid_price_1"] + sample_row["ask_price_1"]) / 2.0
        ax.axvline(mid, color="#333333", linestyle="--", linewidth=1.2, label=f"Mid Price: ${mid:,.2f}")

        ax.set_title(f"Limit Order Book Depth Ladder Snapshot ({sample_row['timestamp']})", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Order Price (USDT)", fontsize=11)
        ax.set_ylabel("Cumulative Depth (BTC)", fontsize=11)
        ax.legend(frameon=True)

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {out_path}")
        return out_path

    def plot_spread_and_volatility(
        self,
        df_feat: pd.DataFrame,
        filename: str = "04_spread_and_volatility.png"
    ) -> Path:
        """Plots dual-axis bid-ask spread and rolling volatility time series."""
        fig, ax_spread = plt.subplots(figsize=(14, 5.5))
        ax_vol = ax_spread.twinx()

        dates = pd.to_datetime(df_feat["timestamp"])
        spread_bps = df_feat["spread_bps"]
        vol_20 = df_feat["volatility_20"] * 10000.0  # converted to bps

        line1 = ax_spread.plot(dates, spread_bps, color="#1f77b4", linewidth=1.2, alpha=0.8, label="Bid-Ask Spread (bps)")
        line2 = ax_vol.plot(dates, vol_20, color="#d9534f", linewidth=1.5, alpha=0.9, label="20-Tick Rolling Volatility (bps)")

        ax_spread.set_title("Market Liquidity & Volatility Clustering Dynamics", fontsize=13, fontweight="bold", pad=12)
        ax_spread.set_ylabel("Spread (Basis Points)", color="#1f77b4", fontsize=11)
        ax_vol.set_ylabel("Realized Volatility (bps)", color="#d9534f", fontsize=11)
        ax_spread.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax_spread.set_xlabel("Time (UTC)", fontsize=11)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_spread.legend(lines, labels, loc="upper right", frameon=True)

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {out_path}")
        return out_path

    def plot_micro_price_lead_lag(
        self,
        df_feat: pd.DataFrame,
        filename: str = "05_micro_price_lead_lag.png"
    ) -> Path:
        """Renders Stoikov Micro-Price lead-lag relationship and predictive scatter."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Subplot 1: Micro-Price vs Mid-Price over a 100-tick window
        sub = df_feat.iloc[100:200].copy()
        t = pd.to_datetime(sub["timestamp"])
        ax1.plot(t, sub["mid_price"], label="Mid Price", color="#333333", linewidth=1.5)
        ax1.plot(t, sub["micro_price"], label="Stoikov Micro-Price", color="#e67e22", linewidth=1.5, linestyle="--")
        ax1.set_title("Micro-Price Divergence from Mid-Quote (100-Tick Window)", fontsize=12, fontweight="bold", pad=10)
        ax1.set_ylabel("Price (USDT)")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax1.legend(frameon=True)

        # Subplot 2: Predictive regression scatter
        bias = (df_feat["micro_price"] - df_feat["mid_price"]).iloc[:-1]
        next_ret = df_feat["mid_price"].diff().iloc[1:]

        sns.regplot(
            x=bias, y=next_ret, ax=ax2,
            scatter_kws={"alpha": 0.3, "color": "#2980b9", "s": 15},
            line_kws={"color": "#c0392b", "linewidth": 2}
        )
        ax2.set_title("Predictive Power: Micro-Price Bias vs Next Mid Revision", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel(r"Micro-Price Bias ($P_{micro} - Mid$)")
        ax2.set_ylabel(r"Next Mid-Quote Change ($\Delta Mid_{t+1}$)")

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {out_path}")
        return out_path

    def plot_kyle_lambda(
        self,
        df_feat: pd.DataFrame,
        filename: str = "06_kyle_lambda_impact.png"
    ) -> Path:
        """Visualizes Kyle's Lambda price impact regression."""
        fig, ax = plt.subplots(figsize=(10, 6))

        df_sub = df_feat.dropna(subset=["signed_quantity"]).copy()
        delta_p = df_sub["price"].diff().iloc[1:]
        signed_q = df_sub["signed_quantity"].iloc[1:]

        sns.regplot(
            x=signed_q, y=delta_p, ax=ax,
            scatter_kws={"alpha": 0.35, "color": "#16a085", "s": 18},
            line_kws={"color": "#e74c3c", "linewidth": 2.2, "label": "Fitted Kyle's Lambda (Price Impact)"}
        )

        ax.axhline(0, color="#999999", linestyle=":", linewidth=1)
        ax.axvline(0, color="#999999", linestyle=":", linewidth=1)
        ax.set_title("Empirical Price Impact: Trade Flow vs Immediate Price Delta", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Signed Aggressor Volume (BTC) [+Buy / -Sell]", fontsize=11)
        ax.set_ylabel(r"Price Change $\Delta P_t$ (USDT)", fontsize=11)
        ax.legend(frameon=True, loc="upper left")

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {out_path}")
        return out_path

    def plot_feature_correlation_matrix(
        self,
        df_feat: pd.DataFrame,
        filename: str = "07_feature_correlation_matrix.png"
    ) -> Path:
        """Produces annotated correlation heatmap across key microstructure signals."""
        cols = [
            "spread_bps", "order_imbalance_1", "depth_imbalance_3", "micro_price_bias",
            "signed_quantity", "return_1", "volatility_20", "rsi_14", "bb_pct_b"
        ]
        available = [c for c in cols if c in df_feat.columns]
        corr = df_feat[available].corr()

        fig, ax = plt.subplots(figsize=(9, 7.5))
        sns.heatmap(
            corr, ax=ax, cmap="vlag", vmin=-1.0, vmax=1.0,
            annot=True, fmt=".2f", square=True,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
            linewidths=0.5
        )
        ax.set_title("Market Microstructure Signal Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
        ax.tick_params(axis="x", rotation=40)

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {out_path}")
        return out_path

    def plot_trade_distribution_qq(
        self,
        df_clean: pd.DataFrame,
        filename: str = "08_trade_size_distribution.png"
    ) -> Path:
        """Renders trade size distribution with log-normal overlay and Q-Q heavy tail plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        q = df_clean["quantity"].dropna().to_numpy()
        log_q = np.log(q[q > 0])

        # 1. Histogram of log trade size
        sns.histplot(log_q, kde=True, ax=ax1, color="#2c3e50", bins=30, stat="density")
        ax1.set_title("Log-Transformed Trade Size Density", fontsize=12, fontweight="bold", pad=10)
        ax1.set_xlabel(r"$\ln(\text{Quantity})$")
        ax1.set_ylabel("Probability Density")

        # 2. Q-Q plot against theoretical normal
        stats.probplot(log_q, dist="norm", plot=ax2)
        ax2.get_lines()[0].set_markerfacecolor("#3498db")
        ax2.get_lines()[0].set_markeredgecolor("none")
        ax2.get_lines()[0].set_alpha(0.6)
        ax2.get_lines()[1].set_color("#e74c3c")
        ax2.get_lines()[1].set_linewidth(2)
        ax2.set_title("Q-Q Plot: Log Trade Size vs Normal Distribution", fontsize=12, fontweight="bold", pad=10)

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {out_path}")
        return out_path
