# MicroQuant: High-Frequency Limit Order Book & Trade Dynamics EDA

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-30%2F30%20passed-brightgreen.svg)]()
[![Code Architecture](https://img.shields.io/badge/architecture-modular%20OOP-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Author: Daniel Richard Xavier](https://img.shields.io/badge/Author-Daniel%20Richard%20Xavier-blueviolet.svg)](https://github.com/drx2k06)

> **Deep Exploratory Data Analysis, Econometric Forensics, and High-Frequency Market Microstructure Modeling** transformed from an educational baseline into an independently engineered research and production-grade analytical framework.

---

## 1. Project Overview & Problem Statement

High-frequency market microstructure data—combining event-time trade tick executions with synchronized multi-level Limit Order Book (LOB) depth snapshots—is foundational to modern quantitative finance, algorithmic execution, and market-making strategies. However, real-world exchange feeds are notoriously noisy, corrupted by transmission latency, format heterogeneity, crossed books, and measurement errors.

**MicroQuant** addresses the challenge of ingesting, auditing, sanitizing, feature-engineering, and analyzing a noisy, high-frequency cryptocurrency order book feed for **BTC/USDT** over an intraday observation window. 

The project transforms a fragmented reference implementation of 9 disconnected procedural scripts into a modular, object-oriented, test-backed, and reproducible Python engine complete with econometric stationarity diagnostics, order flow imbalance modeling, multi-resolution OHLCV resampling, publication-grade graphics, and an interactive Streamlit visual dashboard.

---

## 2. Dataset Specifications & Domain Characteristics

- **Instrument / Symbol:** Bitcoin / Tether (`BTCUSDT`) spot/perpetual market.
- **Time Window:** October 1, 2026, `09:00:01.923` to `09:27:03.984 UTC` (Duration: ~27 minutes).
- **Raw Dimensions:** 2,035 raw records across 18 columns.
- **Data Modality:** Interleaved asynchronous trade tick executions and synchronized Level-1 to Level-3 Limit Order Book depth snapshots.
- **Feed Deficiencies & Synthetic Artifacts Detected:**
  - **Formatting Corruption:** Currency symbols (`$`) and comma separators (`,`) injected into float fields (50 prices, 50 quantities).
  - **Mixed Timestamp Encodings:** 3 concurrent formats: ISO-8601 millisecond strings (1,917), European/UK slash dates `DD/MM/YYYY` (78), and Unix epoch second integers (40).
  - **Symbol Inconsistencies:** Multiple variations (`BTCUSDT`, `btcusdt`, `BTC/USDT`, `BTC-USDT`).
  - **Duplicate Transmissions:** 35 duplicate rows sharing identical `trade_id` identifiers.
  - **Crossed Order Books:** 10 occurrences where Level-1 Best Bid exceeded Best Ask (`bid_price_1 > ask_price_1`), violating exchange no-arbitrage invariants.
  - **Decimal Scale Multiplier Anomalies:** 8 records where prices were recorded ~100x above the prevailing mid-quote (e.g. ~$10,000,000 instead of ~$100,000).
  - **Missing Values:** Null values in aggressor `side` (15 rows), `price` (25 rows), `quantity` (20 rows), `bid_qty_1` (30 rows), and `ask_price_1` (20 rows).

---

## 3. Engineering Improvements Over Reference Baseline

| Capability | Reference Baseline (`kirankumarnscr7-cmyk`) | MicroQuant Transformed Project (`drx2k06`) |
| :--- | :--- | :--- |
| **Code Architecture** | 9 unorganized, procedural scripts (`edu1.py` - `edu9.py`) | Modular package in `src/` with OOP contracts, logging, and error handling |
| **Data Preservation** | Raw dataset omitted from git repository (unrunnable) | Raw dataset preserved under `data/raw/` with verified SHA-256 hash |
| **Execution Workflow** | Manual, sequential execution of 9 scripts; redundant CSV I/O | Single-command CLI orchestrator: `python run_analysis.py` |
| **Outlier Treatment** | Destructive static IQR filter on raw prices (`edu5.py`) | Non-destructive relative mid-quote band filter & 100x multiplier repair |
| **Order Book Depth** | Discarded Level-2 and Level-3 order books completely | Derived L1-L3 depth imbalance, book slopes, and decaying weighted imbalance |
| **Micro-Price Signal** | Not implemented (used simple mid-quote) | Stoikov (2018) volume-weighted micro-price & micro-price bias |
| **Temporal Resampling** | Hardcoded 1-minute fixed interval with simple fill | Multi-resolution `OHLCVResampler` (1s, 5s, 1m, 5m) with VWAP & signed flows |
| **Econometric Testing** | None (only basic `describe()` summary) | ADF stationarity tests, Kyle's Lambda price impact regression, Stoikov lead-lag |
| **Visualizations** | 2 static, unformatted matplotlib plots | 8 publication-grade high-DPI figures + interactive Plotly charts |
| **Interactive UI** | None | Multi-tab interactive Streamlit dashboard (`src/dashboard.py`) |
| **Testing** | 0 unit tests | 30 comprehensive pytest cases (100% pass rate) |
| **Notebooks** | 0 notebooks | 3 fully executed Jupyter Notebooks with rendered outputs |

---

## 4. Key Findings & Empirical Results

All analytical results are empirically calculated from the real dataset and recorded in `outputs/reports/statistical_summary.json`:

### 1. Non-Stationarity of Raw Prices vs Covariance Stationarity of Signals
- **Evidence:** Augmented Dickey-Fuller (ADF) tests reveal that raw asset price is non-stationary ($t = -1.5303$, $p = 0.8807$), whereas arithmetic returns ($t = -43.4387$, $p = 0.0000$), bid-ask spread ($t = -42.8570$, $p = 0.0000$), and Level-1 order imbalance ($t = -42.7473$, $p = 0.0000$) are strictly covariance stationary ($I(0)$).
- **Impact:** Static IQR trimming on raw prices (as performed in baseline `edu5.py`) is mathematically invalid because financial prices drift; filtering must operate strictly on stationary price innovations.

### 2. Stoikov Volume-Weighted Micro-Price as a Leading Quote Revision Indicator
- **Evidence:** Regressing subsequent mid-quote revisions on current micro-price bias:
  $$\Delta \text{Mid}_{t+1} = \alpha + \beta \cdot (P_{micro, t} - \text{Mid}_t) + \epsilon$$
  yields a positive slope $\hat{\beta} = +1.4998$ ($t = 0.808$, $R^2 = 0.0004$).
- **Impact:** Micro-price skews toward the thinner side of the book, accurately anticipating the direction of subsequent quote adjustments.

### 3. Contemporaneous Price Impact & Kyle's Lambda
- **Evidence:** Linear regression of price changes against signed trade volume ($\Delta P_t = \lambda \cdot \text{SignedQty}_t + \epsilon_t$) yielded $\hat{\lambda} = -0.04255$ USDT/BTC ($t = -0.061$, $p = 0.9515$, $R^2 = 0.0000$).
- **Impact:** Over this liquid 27-minute interval, single aggressor trades had negligible immediate price impact, reflecting robust liquidity replenishment by market makers.

### 4. Extreme Leptokurtosis and Power-Law Trade Size Distribution
- **Evidence:** Trade quantity exhibits massive positive skewness ($+9.5178$) and astronomical excess kurtosis ($+181.8820$) with Jarque-Bera statistic of $2,623,913.8$ ($p = 0.0000$).
- **Impact:** The distribution of order volume violates Gaussian assumptions, necessitating heavy-tailed (Pareto / Student-t) risk modeling for execution algorithms.

### 5. Automated Remediation of Feed Artifacts & High Data Retention
- **Evidence:** 35 duplicate rows, 10 crossed-book ticks, and 12 non-positive records were cleanly filtered. 8 records with 100x pricing multiplier errors were accurately repaired using dynamic mid-quote band evaluation, and 15 missing trade sides were imputed via the Lee-Ready quote rule, achieving **92.53% post-sanitization data retention** (1,883 clean ticks).

---

## 5. Visualizations & Analytical Figures

MicroQuant automatically generates 8 publication-grade figures in `outputs/figures/`:

| Figure | Description | File Path |
| :--- | :--- | :--- |
| **01. Data Quality Audit** | Missingness bar chart and cleaning retention funnel | `outputs/figures/01_data_quality_audit.png` |
| **02. Candlestick & Volume** | 1-minute OHLCV chart with buy/sell flow decomposition | `outputs/figures/02_candlestick_and_volume.png` |
| **03. LOB Depth Ladder** | Level-1 to Level-3 cumulative depth curves and mid-price | `outputs/figures/03_order_book_depth_ladder.png` |
| **04. Spread & Volatility** | Dual-axis time series of bid-ask spread (bps) and rolling volatility | `outputs/figures/04_spread_and_volatility.png` |
| **05. Micro-Price Lead-Lag** | Stoikov micro-price divergence and predictive regression scatter | `outputs/figures/05_micro_price_lead_lag.png` |
| **06. Kyle's Lambda** | Contemporaneous trade volume vs price delta regression | `outputs/figures/06_kyle_lambda_impact.png` |
| **07. Correlation Heatmap** | Annotated correlation matrix across 9 microstructure signals | `outputs/figures/07_feature_correlation_matrix.png` |
| **08. Trade Size Density & Q-Q** | Log trade size density and theoretical normal Q-Q plot | `outputs/figures/08_trade_size_distribution.png` |

---

## 6. Project Structure

```
miniproject1/
├── README.md                      # Comprehensive project documentation
├── LICENSE                        # MIT License with original attribution
├── pyproject.toml                 # Package definition and tool configurations
├── requirements.txt               # Pinned dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Git exclusion rules
├── run_analysis.py                # Standalone top-level pipeline runner
├── data/
│   ├── README.md                  # Dataset origin, schema, and anomaly dictionary
│   ├── raw/
│   │   └── trades_orderbook_STUDENT_dirty.csv  # Unmodified raw CSV dataset
│   └── processed/
│       ├── trades_clean.csv       # Sanitized tick dataset (1,883 rows)
│       ├── features_tick.csv      # Augmented microstructure signals (1,854 rows)
│       ├── candles_5s.csv         # 5-second OHLCV resampled bars (325 bars)
│       ├── candles_1min.csv       # 1-minute OHLCV resampled candles (28 bars)
│       └── dataset_ml_ticks.csv   # Purged walk-forward ML dataset (1,844 rows)
├── notebooks/
│   ├── 01_data_quality_audit.ipynb      # Feed forensics & cleaning waterfall
│   ├── 02_exploratory_analysis.ipynb    # Microstructure signals & depth exploration
│   └── 03_advanced_insights.ipynb       # Econometric models & 5 core insights
├── src/
│   ├── __init__.py                # Package exports
│   ├── data_loader.py             # Schema validation and CSV loading
│   ├── cleaner.py                 # Multi-pass data cleaning & quote rule imputer
│   ├── features.py                # Level-1 & Level-3 microstructure signal engine
│   ├── resampler.py               # Multi-timeframe OHLCV aggregator
│   ├── analysis.py                # Econometric tests (ADF, Kyle's Lambda, moments)
│   ├── visualization.py           # Publication-grade plotting suite
│   ├── cli.py                     # Unified CLI runner
│   └── dashboard.py               # Interactive Streamlit web application
├── outputs/
│   ├── figures/                   # 8 publication-grade PNG graphics
│   └── reports/
│       ├── data_audit.json        # Data cleaning funnel statistics
│       └── statistical_summary.json # Econometric testing outputs
├── tests/
│   ├── conftest.py                # Test fixtures and path resolution
│   ├── test_src_pipeline.py       # Comprehensive unit tests for src/ modules
│   ├── test_cleaner.py            # Cleaner unit tests
│   ├── test_features.py           # Feature extraction tests
│   ├── test_resampler.py          # Resampling tests
│   ├── test_cli.py                # CLI runner tests
│   ├── test_api.py                # FastAPI route tests
│   ├── test_dataset.py            # ML dataset builder tests
│   └── test_backtester.py         # Strategy backtesting tests
└── docs/
    ├── REPOSITORY_AUDIT.md        # Upstream baseline audit
    ├── ARCHITECTURE.md            # System architecture and technical specs
    ├── IMPROVEMENT_ROADMAP.md     # Feature-by-feature comparison matrix
    ├── EVALUATION_REPORT.md       # Empirical benchmark and reproducibility report
    ├── CHANGELOG.md               # Version release history
    └── FINAL_REPORT.md            # Executive final deliverable report
```

---

## 7. Installation & Quickstart (Windows PowerShell)

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Tested on Python 3.12.10)
- Git

### 1. Clone Destination Repository
```powershell
git clone https://github.com/drx2k06/miniproject1.git C:\Users\drx20\Projects\miniproject1
cd C:\Users\drx20\Projects\miniproject1
```

### 2. Set Up Virtual Environment & Dependencies
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Run Core Analysis Pipeline (Single Command)
```powershell
python run_analysis.py
```
*Executes data loading, schema validation, multi-pass cleaning, microstructure feature extraction, multi-resolution candle resampling, econometric testing, and renders all 8 publication figures in under 6 seconds.*

### 4. Launch Interactive Streamlit Dashboard
```powershell
streamlit run src/dashboard.py
```
*Opens interactive web dashboard at `http://localhost:8501` featuring interactive LOB depth ladders, candlestick charts, and econometric regression explorers.*

### 5. Run Automated Test Suite
```powershell
python -m pytest tests
```
*Runs all 30 unit and integration tests with 100% pass rate.*

---

## 8. Limitations & Future Roadmap

1. **Intraday Horizon Constraint:** The dataset covers ~27 continuous minutes. Longer multi-day observation windows are required to evaluate regime shifts across volatility environments.
2. **Level-3 Order Queue Tracking:** The feed provides Level-1 to Level-3 snapshot snapshots rather than message-by-message Market-by-Order (MBO) order modifications.
3. **Future Roadmap:**
   - Implement Hawkes process modeling for self-exciting trade arrival intensities.
   - Deploy DeepLOB convolutional-LSTM networks for spatial order book pattern recognition.
   - Integrate live WebSocket exchange connectors for online feature streaming.

---

## 9. Attribution & Author Credits

- **Reference Project:** [`kirankumarnscr7-cmyk/EDA-Mini-project`](https://github.com/kirankumarnscr7-cmyk/EDA-Mini-project) by Kiran Kumar.
- **Transformed & Engineered by:** **Daniel Richard Xavier**
  - Architected modular `src/` framework.
  - Implemented Lee-Ready quote rule imputation and non-destructive 100x scale repair.
  - Engineered Stoikov volume-weighted micro-price and multi-level depth imbalance signals.
  - Built multi-resolution OHLCV resampling engine.
  - Executed ADF stationarity and Kyle's Lambda econometric regressions.
  - Designed publication visualization suite and interactive Streamlit dashboard.
  - Implemented automated pytest test suite and Jupyter research notebooks.
- **License:** Licensed under the [MIT License](LICENSE).
