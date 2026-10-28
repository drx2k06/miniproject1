"""
Aetheris-MicroQuant: Quantitative Order Book Microstructure & Alpha Dashboard.
Developer: Daniel Richard Xavier
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from microquant.core.config import settings
from microquant.models.backtester import MicrostructureBacktester

# Page configuration
st.set_page_config(
    page_title="MicroQuant | Order Book Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Tech Quant Dark Theme Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .metric-card {
        background: linear-gradient(135deg, #161f30 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-val {
        font-size: 24px;
        font-weight: 700;
        font-family: 'Courier New', Courier, monospace;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .bid-text { color: #22c55e; font-weight: bold; }
    .ask-text { color: #ef4444; font-weight: bold; }
    .header-badge {
        display: inline-block;
        padding: 4px 10px;
        background-color: #0369a1;
        color: white;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
        letter-spacing: 0.08em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_datasets():
    clean_df = pd.read_csv(settings.CLEAN_DATA_PATH) if settings.CLEAN_DATA_PATH.exists() else None
    feat_df = pd.read_csv(settings.FEATURES_DATA_PATH) if settings.FEATURES_DATA_PATH.exists() else None
    c1m = pd.read_csv(settings.CANDLES_1MIN_PATH) if settings.CANDLES_1MIN_PATH.exists() else None
    c5s = pd.read_csv(settings.CANDLES_5S_PATH) if settings.CANDLES_5S_PATH.exists() else None
    ml_df = pd.read_csv(settings.DATASET_ML_PATH) if settings.DATASET_ML_PATH.exists() else None

    metrics = {}
    importances = {}
    m_path = settings.MODEL_DIR / "model_metrics.json"
    i_path = settings.MODEL_DIR / "feature_importances.json"
    if m_path.exists():
        with open(m_path) as f:
            metrics = json.load(f)
    if i_path.exists():
        with open(i_path) as f:
            importances = json.load(f)

    return clean_df, feat_df, c1m, c5s, ml_df, metrics, importances


clean_df, feat_df, c1m, c5s, ml_df, metrics, importances = load_datasets()

# Header
st.markdown('<span class="header-badge">ENTERPRISE MICROSTRUCTURE INTELLIGENCE</span>', unsafe_allow_html=True)
st.title("⚡ Aetheris-MicroQuant")
st.caption("Quantitative High-Frequency Order Book Microstructure & Alpha Engine | Developed by **Daniel Richard Xavier**")

if clean_df is None or feat_df is None:
    st.warning("⚠️ Data has not been cleaned or generated yet. Run `microquant run-all` or trigger via API to initialize datasets.")
    st.stop()

# Sidebar
st.sidebar.image("https://img.shields.io/badge/MicroQuant-v1.0.0-0284c7?style=for-the-badge", use_container_width=True)
st.sidebar.markdown("### 🎛️ Control Panel")
selected_symbol = st.sidebar.selectbox("Active Asset", ["BTCUSDT"])
tick_slider = st.sidebar.slider("Historical Ticks to Inspect", 100, len(feat_df), min(500, len(feat_df)))

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚙️ Engine Summary")
st.sidebar.text(f"Total Ticks: {len(clean_df)}")
st.sidebar.text(f"Timespan: 27 mins")
st.sidebar.text(f"Models Ready: {len(metrics)}")
st.sidebar.markdown("---")
st.sidebar.info("Designed & built by **Daniel Richard Xavier**\nOriginal reference: Keerthana S M K")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Microstructure Depth",
    "🕯️ Candles & Volume Flow",
    "🧠 Alpha Models & Metrics",
    "📈 Strategy Backtest",
    "⚡ What-If Simulator",
])

# ----------------- TAB 1: Microstructure Depth -----------------
with tab1:
    st.subheader("Level-3 Order Book Depth & Microstructure Signals")
    latest = clean_df.iloc[-1]
    latest_feat = feat_df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-lbl">Mid Price</div>
            <div class="metric-val">${latest_feat['mid_price']:.2f}</div>
            <small style="color:#94a3b8;">Micro-Price: ${latest_feat['micro_price']:.2f}</small></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-lbl">Bid-Ask Spread</div>
            <div class="metric-val">{latest_feat['spread_bps']:.3f} bps</div>
            <small style="color:#94a3b8;">${latest_feat['spread']:.2f} absolute</small></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        imbalance_color = "#22c55e" if latest_feat["order_imbalance_1"] > 0 else "#ef4444"
        st.markdown(
            f"""<div class="metric-card"><div class="metric-lbl">Order Imbalance (L1)</div>
            <div class="metric-val" style="color:{imbalance_color};">{latest_feat['order_imbalance_1']:+.3f}</div>
            <small style="color:#94a3b8;">Depth Imb (L3): {latest_feat['depth_imbalance_3']:+.3f}</small></div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-lbl">Wilder's RSI (14)</div>
            <div class="metric-val">{latest_feat['rsi_14']:.1f}</div>
            <small style="color:#94a3b8;">Vol(20): {latest_feat['volatility_20']*10000:.1f} bps</small></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("#### Level-3 Order Book Ladder (Latest Snapshot)")
    ladder_col1, ladder_col2 = st.columns(2)
    with ladder_col1:
        st.markdown("##### 🟢 Bids (Buy Orders)")
        bids_data = {
            "Level": ["Bid 1 (Best)", "Bid 2", "Bid 3"],
            "Price ($)": [f"{latest['bid_price_1']:.2f}", f"{latest['bid_price_2']:.2f}", f"{latest['bid_price_3']:.2f}"],
            "Quantity (BTC)": [f"{latest['bid_qty_1']:.4f}", f"{latest['bid_qty_2']:.4f}", f"{latest['bid_qty_3']:.4f}"],
        }
        st.dataframe(pd.DataFrame(bids_data), use_container_width=True)

    with ladder_col2:
        st.markdown("##### 🔴 Asks (Sell Orders)")
        asks_data = {
            "Level": ["Ask 1 (Best)", "Ask 2", "Ask 3"],
            "Price ($)": [f"{latest['ask_price_1']:.2f}", f"{latest['ask_price_2']:.2f}", f"{latest['ask_price_3']:.2f}"],
            "Quantity (BTC)": [f"{latest['ask_qty_1']:.4f}", f"{latest['ask_qty_2']:.4f}", f"{latest['ask_qty_3']:.4f}"],
        }
        st.dataframe(pd.DataFrame(asks_data), use_container_width=True)

    st.markdown("#### Time-Series Microstructure Signals")
    plot_sub = feat_df.tail(tick_slider)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(plot_sub.index, plot_sub["mid_price"], label="Mid Price", color="#38bdf8", lw=1.2)
    ax1.plot(plot_sub.index, plot_sub["vwap"], label="VWAP", color="#f59e0b", ls="--", lw=1.0)
    ax1.set_ylabel("Price ($)", color="#e2e8f0")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.2)

    ax2.plot(plot_sub.index, plot_sub["order_imbalance_1"], label="Order Imbalance (L1)", color="#a855f7", lw=1.0)
    ax2.axhline(0, color="gray", ls=":", alpha=0.5)
    ax2.set_ylabel("Imbalance [-1, 1]", color="#e2e8f0")
    ax2.set_xlabel("Tick Sequence", color="#e2e8f0")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.2)
    fig.patch.set_facecolor("#0f172a")
    ax1.set_facecolor("#1e293b")
    ax2.set_facecolor("#1e293b")
    ax1.tick_params(colors="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    st.pyplot(fig)

# ----------------- TAB 2: Candles & Volume Flow -----------------
with tab2:
    st.subheader("Multi-Resolution Candlestick Aggregation & Order Flow")
    candle_tf = st.radio("Resolution", ["1-Minute Bars", "5-Second Micro-Bars"], horizontal=True)
    df_c = c1m if candle_tf == "1-Minute Bars" else c5s

    st.dataframe(df_c.tail(15), use_container_width=True)

    fig_c, (ax_c1, ax_c2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [2.5, 1]}, sharex=True)
    ax_c1.plot(df_c.index, df_c["close"], label="Close Price", color="#38bdf8", lw=1.5)
    ax_c1.plot(df_c.index, df_c["vwap"], label="Candle VWAP", color="#eab308", ls="--", lw=1.2)
    ax_c1.set_ylabel("Price ($)")
    ax_c1.legend()
    ax_c1.grid(alpha=0.2)

    ax_c2.bar(df_c.index, df_c["buy_volume"], label="Buy Vol", color="#22c55e", alpha=0.8)
    ax_c2.bar(df_c.index, -df_c["sell_volume"], label="Sell Vol", color="#ef4444", alpha=0.8)
    ax_c2.set_ylabel("Signed Vol (BTC)")
    ax_c2.legend()
    ax_c2.grid(alpha=0.2)

    fig_c.patch.set_facecolor("#0f172a")
    ax_c1.set_facecolor("#1e293b")
    ax_c2.set_facecolor("#1e293b")
    ax_c1.tick_params(colors="#94a3b8")
    ax_c2.tick_params(colors="#94a3b8")
    st.pyplot(fig_c)

# ----------------- TAB 3: Alpha Models & Metrics -----------------
with tab3:
    st.subheader("Quantitative Alpha Model Benchmarks")
    if metrics:
        metric_rows = []
        for m_name, m_data in metrics.items():
            metric_rows.append({
                "Model Architecture": m_name.replace("_", " ").title(),
                "Accuracy": f"{m_data['accuracy']*100:.2f}%",
                "Precision": f"{m_data['precision']:.4f}",
                "Recall": f"{m_data['recall']:.4f}",
                "F1 Score": f"{m_data['f1_score']:.4f}",
                "ROC-AUC": f"{m_data['roc_auc']:.4f}",
                "Log Loss": f"{m_data['log_loss']:.4f}",
                "Brier Score": f"{m_data['brier_score']:.4f}",
            })
        st.table(pd.DataFrame(metric_rows))

        if "gradient_boosting" in importances:
            st.markdown("#### Top Predictive Microstructure Features (Gradient Boosting)")
            gb_feats = pd.DataFrame(importances["gradient_boosting"], columns=["Feature", "Importance"]).head(10)
            fig_imp, ax_imp = plt.subplots(figsize=(10, 4))
            ax_imp.barh(gb_feats["Feature"][::-1], gb_feats["Importance"][::-1], color="#38bdf8")
            ax_imp.set_xlabel("Relative Predictive Importance")
            fig_imp.patch.set_facecolor("#0f172a")
            ax_imp.set_facecolor("#1e293b")
            ax_imp.tick_params(colors="#94a3b8")
            st.pyplot(fig_imp)
    else:
        st.info("Train models to view benchmark comparison.")

# ----------------- TAB 4: Strategy Backtest -----------------
with tab4:
    st.subheader("Quantitative High-Frequency Execution Simulator")
    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        fee_in = st.slider("Taker Fee (bps)", 0.0, 10.0, 4.0, 0.5)
    with bc2:
        slip_in = st.slider("Slippage (bps)", 0.0, 5.0, 1.5, 0.5)
    with bc3:
        thresh_long = st.slider("Long Threshold P(up)", 0.50, 0.60, 0.52, 0.01)
    with bc4:
        thresh_short = st.slider("Short Threshold P(up)", 0.40, 0.50, 0.48, 0.01)

    if ml_df is not None:
        split_idx = int(len(ml_df) * settings.TRAIN_TEST_SPLIT_RATIO)
        test_df = ml_df.iloc[split_idx:].reset_index(drop=True)

        scaler = joblib.load(settings.MODEL_DIR / "scaler.joblib")
        gb_model = joblib.load(settings.MODEL_DIR / "gradient_boosting.joblib")
        with open(settings.MODEL_DIR / "feature_names.json") as f:
            f_names = json.load(f)

        X_t = scaler.transform(test_df[f_names].values)
        probs = gb_model.predict_proba(X_t)[:, 1]

        bt = MicrostructureBacktester(
            taker_fee_bps=fee_in,
            slippage_bps=slip_in,
            long_threshold=thresh_long,
            short_threshold=thresh_short,
        )
        res_m, res_hist = bt.run_backtest(
            probabilities=probs,
            prices=test_df["price"].values,
            timestamps=test_df["timestamp"],
        )

        st.markdown(
            f"""
            <div style="display:flex; gap:12px; margin-bottom:15px;">
                <div class="metric-card" style="flex:1;"><div class="metric-lbl">Total Net Return</div><div class="metric-val">{res_m.total_return_pct:.2f}%</div></div>
                <div class="metric-card" style="flex:1;"><div class="metric-lbl">Benchmark Return</div><div class="metric-val">{res_m.benchmark_return_pct:.2f}%</div></div>
                <div class="metric-card" style="flex:1;"><div class="metric-lbl">Strategy Alpha</div><div class="metric-val">{res_m.alpha_pct:.2f}%</div></div>
                <div class="metric-card" style="flex:1;"><div class="metric-lbl">Max Drawdown</div><div class="metric-val">{res_m.max_drawdown_pct:.2f}%</div></div>
                <div class="metric-card" style="flex:1;"><div class="metric-lbl">Total Trades</div><div class="metric-val">{res_m.total_trades}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_eq, ax_eq = plt.subplots(figsize=(12, 4))
        ax_eq.plot(res_hist.index, res_hist["equity"], label="Strategy Equity", color="#22c55e", lw=1.5)
        ax_eq.plot(res_hist.index, res_hist["benchmark_equity"], label="Buy & Hold Benchmark", color="#94a3b8", ls="--", lw=1.0)
        ax_eq.set_ylabel("Equity ($)")
        ax_eq.legend()
        ax_eq.grid(alpha=0.2)
        fig_eq.patch.set_facecolor("#0f172a")
        ax_eq.set_facecolor("#1e293b")
        ax_eq.tick_params(colors="#94a3b8")
        st.pyplot(fig_eq)

# ----------------- TAB 5: What-If Simulator -----------------
with tab5:
    st.subheader("⚡ Real-Time Microsecond What-If Inference Engine")
    st.write("Adjust synthetic order book quotes to simulate incoming market states and inspect model alpha output:")

    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        sim_price = st.number_input("Trade Execution Price ($)", value=62100.0, step=0.5)
        sim_bid1_p = st.number_input("Best Bid Price ($)", value=62099.5, step=0.5)
        sim_ask1_p = st.number_input("Best Ask Price ($)", value=62100.5, step=0.5)
    with wc2:
        sim_bid1_q = st.number_input("Bid Level-1 Quantity (BTC)", value=2.50, step=0.1)
        sim_ask1_q = st.number_input("Ask Level-1 Quantity (BTC)", value=1.00, step=0.1)
        sim_side = st.selectbox("Trade Direction", ["buy", "sell"])
    with wc3:
        sim_total_depth = st.slider("Total L3 Book Depth (BTC)", 5.0, 50.0, 15.0)
        sim_volatility = st.slider("Microstructure Volatility (bps)", 1.0, 50.0, 5.0)

    # Compute features for what-if
    sim_spread = sim_ask1_p - sim_bid1_p
    sim_mid = (sim_ask1_p + sim_bid1_p) / 2.0
    sim_spread_bps = (sim_spread / sim_mid) * 10000.0
    sim_imb1 = (sim_bid1_q - sim_ask1_q) / (sim_bid1_q + sim_ask1_q + 1e-9)

    scaler = joblib.load(settings.MODEL_DIR / "scaler.joblib")
    gb_model = joblib.load(settings.MODEL_DIR / "gradient_boosting.joblib")
    with open(settings.MODEL_DIR / "feature_names.json") as f:
        f_names = json.load(f)

    sim_dict = {
        "spread_bps": sim_spread_bps,
        "order_imbalance_1": sim_imb1,
        "depth_imbalance_3": sim_imb1 * 0.8,
        "weighted_imbalance": sim_imb1 * 0.9,
        "micro_price_bias": sim_imb1 * 0.3,
        "bid_slope": 0.05,
        "ask_slope": 0.05,
        "slope_ratio": 1.0,
        "signed_quantity": 1.0 if sim_side == "buy" else -1.0,
        "price_to_vwap_bps": 0.0,
        "effective_spread": sim_spread,
        "return_1": 0.0,
        "log_return_1": 0.0,
        "volatility_10": sim_volatility / 10000.0,
        "volatility_20": sim_volatility / 10000.0,
        "ma_ratio": 0.0,
        "rsi_14": 52.0,
        "bb_pct_b": 0.55,
    }
    vec = np.array([[sim_dict.get(fn, 0.0) for fn in f_names]])
    vec_scaled = scaler.transform(vec)
    prob_up = float(gb_model.predict_proba(vec_scaled)[0, 1])

    direction = "📈 UPWARD ALPHA" if prob_up >= 0.5 else "📉 DOWNWARD ALPHA"
    dir_color = "#22c55e" if prob_up >= 0.5 else "#ef4444"

    st.markdown(
        f"""
        <div class="metric-card" style="text-align:center; padding:24px;">
            <div style="font-size:16px; color:#94a3b8;">MODEL DIRECTIONAL VERDICT</div>
            <div style="font-size:32px; font-weight:800; color:{dir_color}; margin:8px 0;">{direction}</div>
            <div style="font-size:20px; color:#e2e8f0;">Probability P(Price Up): <b>{prob_up*100:.2f}%</b> (Confidence: {abs(prob_up-0.5)*200:.1f}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
