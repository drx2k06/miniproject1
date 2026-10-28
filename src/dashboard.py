"""
Interactive Streamlit Dashboard for Market Microstructure EDA.

Author: Daniel Richard Xavier
Usage:
    streamlit run src/dashboard.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="MicroQuant | High-Frequency LOB & Trade Dynamics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    project_root = Path(__file__).resolve().parent.parent
    clean_path = project_root / "data" / "processed" / "trades_clean.csv"
    feat_path = project_root / "data" / "processed" / "features_tick.csv"
    candles_path = project_root / "data" / "processed" / "candles_1min.csv"
    audit_path = project_root / "outputs" / "reports" / "data_audit.json"
    stats_path = project_root / "outputs" / "reports" / "statistical_summary.json"

    df_clean = pd.read_csv(clean_path, parse_dates=["timestamp"]) if clean_path.exists() else pd.DataFrame()
    df_feat = pd.read_csv(feat_path, parse_dates=["timestamp"]) if feat_path.exists() else pd.DataFrame()
    df_candles = pd.read_csv(candles_path, parse_dates=["timestamp"]) if candles_path.exists() else pd.DataFrame()

    audit = {}
    if audit_path.exists():
        with open(audit_path, "r", encoding="utf-8") as f:
            audit = json.load(f)

    stats_summary = {}
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats_summary = json.load(f)

    return df_clean, df_feat, df_candles, audit, stats_summary


df_clean, df_feat, df_candles, audit, stats_summary = load_data()

# Sidebar
st.sidebar.title("📈 MicroQuant Engine")
st.sidebar.markdown("**Lead Data Scientist & Engineer:**  \n**Daniel Richard Xavier**")
st.sidebar.markdown("---")
st.sidebar.info(
    "Transforming raw, noisy cryptocurrency limit order book ticks into "
    "statistically validated high-frequency microstructure insights."
)

if not df_clean.empty:
    st.sidebar.markdown(f"**Dataset Window:**  \n`{df_clean['timestamp'].min().strftime('%H:%M:%S')}` to `{df_clean['timestamp'].max().strftime('%H:%M:%S')} UTC`")
    st.sidebar.markdown(f"**Cleaned Records:** `{len(df_clean):,}`")
    st.sidebar.markdown(f"**Total Traded Volume:** `{df_clean['quantity'].sum():,.2f} BTC`")

# Header
st.title("MicroQuant: High-Frequency Limit Order Book & Trade Dynamics")
st.caption("Deep Exploratory Data Analysis & Econometric Microstructure Forensics | Developed by Daniel Richard Xavier")

# Top KPI Row
if not df_clean.empty:
    c1, c2, c3, c4, c5 = st.columns(5)
    latest_p = df_clean['price'].iloc[-1]
    p_change = (latest_p / df_clean['price'].iloc[0] - 1) * 100
    mean_spread_bps = df_feat['spread_bps'].mean() if not df_feat.empty else 0.0

    c1.metric("Asset Pair", "BTC/USDT", "Binance / LOB Feed")
    c2.metric("Latest Price", f"${latest_p:,.2f}", f"{p_change:+.3f}%")
    c3.metric("Clean Records", f"{len(df_clean):,}", f"Retention: {audit.get('cleaning_audit', {}).get('retention_rate_pct', 92.5):.1f}%")
    c4.metric("Avg Spread (bps)", f"{mean_spread_bps:.2f} bps", "Level-1 Liquidity")
    c5.metric("Total Volume", f"{df_clean['quantity'].sum():,.2f} BTC", f"{len(df_candles)} 1m Bars")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 1. Data Quality Audit",
    "📚 2. Order Book Depth Ladder",
    "🕯️ 3. Candlesticks & Volume Tape",
    "📐 4. Econometric Signals & Kyle's Lambda",
    "📊 5. Trade Flow & Distribution"
])

# ----------------- TAB 1: DATA QUALITY AUDIT -----------------
with tab1:
    st.subheader("Data Cleaning Waterfall & Anomaly Forensics")
    st.markdown(
        "The raw dataset (`trades_orderbook_STUDENT_dirty.csv`) was intentionally seeded with 8 distinct corruption types: "
        "currency string formatting (`$`, `,`), multi-format timestamps, casing inconsistencies, duplicate records, "
        "crossed books (`bid > ask`), non-positive values, and 100x pricing multiplier errors."
    )

    clean_audit = audit.get("cleaning_audit", {})
    val_report = audit.get("validation_report", {})

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("### Cleaning Funnel Metrics")
        audit_df = pd.DataFrame([
            {"Stage / Anomaly": "Initial Raw Records", "Count": clean_audit.get("initial_records", 2035)},
            {"Stage / Anomaly": "Duplicate Rows Removed", "Count": -clean_audit.get("duplicate_rows_removed", 35)},
            {"Stage / Anomaly": "Null Essential Quotes Dropped", "Count": -clean_audit.get("null_quotes_dropped", 15)},
            {"Stage / Anomaly": "Non-Positive Prices/Qty Dropped", "Count": -clean_audit.get("non_positive_values_dropped", 12)},
            {"Stage / Anomaly": "Crossed Books Dropped (Bid >= Ask)", "Count": -clean_audit.get("crossed_books_dropped", 10)},
            {"Stage / Anomaly": "100x Pricing Multipliers Corrected", "Count": clean_audit.get("scale_anomalies_corrected", 8)},
            {"Stage / Anomaly": "Missing Sides Imputed (Quote Rule)", "Count": clean_audit.get("side_imputations_quote_rule", 15)},
            {"Stage / Anomaly": "Final Validated Clean Records", "Count": clean_audit.get("final_clean_records", 1883)},
        ])
        st.dataframe(audit_df, use_container_width=True)

    with col_b:
        fig_waterfall = go.Figure(go.Waterfall(
            name="Records", orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=["Initial", "Duplicates", "Nulls", "Non-Positive", "Crossed Books", "Cleaned"],
            textposition="outside",
            y=[
                clean_audit.get("initial_records", 2035),
                -clean_audit.get("duplicate_rows_removed", 35),
                -clean_audit.get("null_quotes_dropped", 15),
                -clean_audit.get("non_positive_values_dropped", 12),
                -clean_audit.get("crossed_books_dropped", 10),
                clean_audit.get("final_clean_records", 1883)
            ],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig_waterfall.update_layout(title="Record Retention Cascade", height=380, template="plotly_white")
        st.plotly_chart(fig_waterfall, use_container_width=True)


# ----------------- TAB 2: ORDER BOOK DEPTH LADDER -----------------
with tab2:
    st.subheader("Limit Order Book (LOB) Multi-Level Depth Visualizer")
    if not df_clean.empty:
        idx = st.slider("Select Tick Index for Snapshot Inspection", 0, len(df_clean)-1, len(df_clean)//2)
        row = df_clean.iloc[idx]

        st.caption(f"Inspecting Trade ID `{row['trade_id']}` at `{row['timestamp']}` | Executed Price: **${row['price']:,.2f}** ({row['side'].upper()})")

        bids = pd.DataFrame({
            "Level": ["L1", "L2", "L3"],
            "Price": [row["bid_price_1"], row["bid_price_2"], row["bid_price_3"]],
            "Quantity": [row["bid_qty_1"], row["bid_qty_2"], row["bid_qty_3"]]
        }).sort_values("Price", ascending=False)
        bids["CumDepth"] = bids["Quantity"].cumsum()

        asks = pd.DataFrame({
            "Level": ["L1", "L2", "L3"],
            "Price": [row["ask_price_1"], row["ask_price_2"], row["ask_price_3"]],
            "Quantity": [row["ask_qty_1"], row["ask_qty_2"], row["ask_qty_3"]]
        }).sort_values("Price", ascending=True)
        asks["CumDepth"] = asks["Quantity"].cumsum()

        fig_depth = go.Figure()
        fig_depth.add_trace(go.Scatter(
            x=bids["Price"], y=bids["CumDepth"], mode="lines+markers",
            name="Bid Cumulative Depth", line_shape="hv", fill="tozeroy",
            line=dict(color="#2ca02c", width=3)
        ))
        fig_depth.add_trace(go.Scatter(
            x=asks["Price"], y=asks["CumDepth"], mode="lines+markers",
            name="Ask Cumulative Depth", line_shape="vh", fill="tozeroy",
            line=dict(color="#d62728", width=3)
        ))

        mid = (row["bid_price_1"] + row["ask_price_1"]) / 2.0
        fig_depth.add_vline(x=mid, line_dash="dash", line_color="black", annotation_text=f"Mid: ${mid:,.2f}")
        fig_depth.update_layout(
            title=f"L1-L3 Depth Ladder Snapshot (Mid: ${mid:,.2f})",
            xaxis_title="Order Price (USDT)", yaxis_title="Cumulative Volume (BTC)",
            height=420, template="plotly_white"
        )
        st.plotly_chart(fig_depth, use_container_width=True)


# ----------------- TAB 3: CANDLESTICKS & VOLUME -----------------
with tab3:
    st.subheader("Multi-Resolution Candlestick Dynamics")
    if not df_candles.empty:
        fig_candle = go.Figure(data=[
            go.Candlestick(
                x=df_candles["timestamp"],
                open=df_candles["open"], high=df_candles["high"],
                low=df_candles["low"], close=df_candles["close"],
                name="BTC/USDT 1m"
            )
        ])
        fig_candle.update_layout(
            title="BTC/USDT 1-Minute OHLCV Series",
            yaxis_title="Price (USDT)", xaxis_rangeslider_visible=False,
            height=450, template="plotly_white"
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        fig_vol = px.bar(
            df_candles, x="timestamp", y=["buy_volume", "sell_volume"],
            title="1-Minute Buyer vs Seller Initiated Volume",
            color_discrete_map={"buy_volume": "#2ca02c", "sell_volume": "#d62728"},
            labels={"value": "Volume (BTC)", "variable": "Aggressor Flow"}
        )
        fig_vol.update_layout(height=260, template="plotly_white")
        st.plotly_chart(fig_vol, use_container_width=True)


# ----------------- TAB 4: ECONOMETRICS & KYLE'S LAMBDA -----------------
with tab4:
    st.subheader("Econometric Tests & Market Impact Analysis")

    kyle = stats_summary.get("kyle_lambda", {})
    micro = stats_summary.get("micro_price_lead_lag", {})
    adf_tests = stats_summary.get("adf_tests", [])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Kyle's Lambda (Contemporaneous Price Impact)")
        st.markdown(f"**Regression Model:** $\Delta P_t = \lambda \cdot SignedQuantity_t + \epsilon_t$")
        st.metric("Estimated Lambda", f"${kyle.get('slope', 0):.4f} / BTC", f"t = {kyle.get('t_statistic', 0):.2f}")
        st.write(kyle.get("interpretation", "Price impact measurement."))

    with c2:
        st.markdown("### Stoikov Micro-Price Lead-Lag")
        st.markdown(r"**Hypothesis:** $\Delta Mid_{t+1} = \alpha + \beta \cdot (P_{micro, t} - Mid_t) + \epsilon$")
        st.metric("Micro-Price Beta", f"{micro.get('slope', 0):.3f}", f"t = {micro.get('t_statistic', 0):.2f}")
        st.write(micro.get("interpretation", "Micro-price leading indicator test."))

    st.markdown("### Stationarity Diagnostics (Augmented Dickey-Fuller Test)")
    if adf_tests:
        adf_df = pd.DataFrame(adf_tests)
        st.dataframe(adf_df[["variable", "adf_statistic", "p_value", "is_stationary"]], use_container_width=True)


# ----------------- TAB 5: TRADE FLOW & DISTRIBUTION -----------------
with tab5:
    st.subheader("Trade Size Distribution & Heavy Tail Diagnostics")
    if not df_clean.empty:
        c_flow1, c_flow2 = st.columns(2)
        with c_flow1:
            side_counts = df_clean["side"].value_counts().reset_index()
            fig_pie = px.pie(side_counts, values="count", names="side", title="Trade Count by Aggressor Side", color="side", color_discrete_map={"buy": "#2ca02c", "sell": "#d62728"})
            st.plotly_chart(fig_pie, use_container_width=True)

        with c_flow2:
            fig_hist = px.histogram(df_clean, x="quantity", nbins=50, title="Trade Size Distribution (Leptokurtic Heavy Tail)", color_discrete_sequence=["#2980b9"])
            fig_hist.update_layout(xaxis_title="Trade Quantity (BTC)", yaxis_title="Count", height=380, template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("### Live Trade Tape (Sample)")
        st.dataframe(df_clean[["trade_id", "timestamp", "side", "price", "quantity", "bid_price_1", "ask_price_1"]].head(25), use_container_width=True)

st.markdown("---")
st.markdown("<center><small>Developed by <b>Daniel Richard Xavier</b> | MicroQuant Market Microstructure EDA Mini-Project</small></center>", unsafe_allow_html=True)
