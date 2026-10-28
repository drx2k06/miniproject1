# Comprehensive Repository Audit & Analytical Forensics

**Project:** High-Frequency Limit Order Book & Trade Dynamics (EDA Mini-Project)  
**Reference Repository:** [`kirankumarnscr7-cmyk/EDA-Mini-project`](https://github.com/kirankumarnscr7-cmyk/EDA-Mini-project)  
**Destination Repository:** [`drx2k06/miniproject1`](https://github.com/drx2k06/miniproject1)  
**Lead Auditor & Engineer:** Daniel Richard Xavier  
**Date of Audit:** October 28, 2026  

---

## 1. Executive Summary

This audit evaluates the codebase, data artifacts, methodology, and analytical scope of the reference repository `kirankumarnscr7-cmyk/EDA-Mini-project`. The project investigates high-frequency trade ticks and Level-2/Level-3 limit order book (LOB) snapshots for cryptocurrency asset **BTC/USDT** over an intraday observation window. 

While the reference implementation demonstrates an understanding of foundational data processing steps—such as string sanitization, temporal parsing, basic feature construction, candle resampling, and walk-forward train-test splitting—it suffers from severe architectural, statistical, and operational deficits. Most critically, the original repository **failed to commit the dataset itself**, relied on 9 monolithic, disconnected scripts with repetitive disk I/O, used naive outlier removal that corrupts valid financial price series, lacked automated testing, and produced rudimentary visualizations without statistical rigor or reproducibility guarantees.

---

## 2. Dataset Discovery & Domain Identification

### 2.1 Domain & Asset Class
- **Asset / Instrument:** Bitcoin / Tether (`BTCUSDT`) spot or perpetual futures.
- **Data Modality:** High-frequency market microstructure feed combining asynchronous trade execution ticks with synchronized Level-1 to Level-3 Limit Order Book (LOB) depth snapshots.
- **Temporal Horizon:** October 1, 2026, 09:00:01.923 UTC to 09:27:03.984 UTC (duration: ~27 minutes, 2 seconds).
- **Record Count:** 2,035 raw records across 18 columns.

### 2.2 Pedagogical Nature & Intentional Corruption
The dataset (`trades_orderbook_STUDENT_dirty.csv`) is an educational benchmark intentionally seeded with realistic data-entry, parsing, and exchange feed anomalies:
1. **String Formatting Artifacts:** Currency signs (`$`) and comma separators (`,`) injected into float columns (`price`, `quantity`, book prices).
2. **Multi-Format Timestamps:** 1,917 ISO-8601 millisecond strings, 78 UK/European formatted dates (`DD/MM/YYYY HH:MM:SS`), and 40 raw Unix epoch second integers.
3. **Symbol Representation Drift:** Inconsistent casing and delimiters (`BTCUSDT`, `btcusdt`, `BTC/USDT`, `BTC-USDT`).
4. **Duplicate Transmissions:** 35 duplicate rows sharing identical `trade_id` values.
5. **Missing Values:** Null values in aggressor `side` (15 rows), `price` (25 rows), `quantity` (20 rows), `bid_qty_1` (30 rows), and `ask_price_1` (20 rows).
6. **Crossed Order Books:** 10 occurrences where Level-1 Best Bid exceeded Best Ask (`bid_price_1 > ask_price_1`), violating no-arbitrage exchange matching engine invariants.
7. **Decimal Scale Anomalies:** 8 rows where `price` was recorded ~100x higher than prevailing mid-quote (e.g., ~$10,000,000 instead of ~$100,000).
8. **Invalid Non-Positive Observations:** 6 records with `price <= 0` and 6 with `quantity <= 0`.

---

## 3. Detailed Technical Audit of Reference Implementation

The reference repository consists exclusively of 9 Python scripts (`edu1.py` through `edu9.py`). Each script was audited line-by-line:

| Script | Stated Objective | Implementation Summary | Deficiencies & Critical Weaknesses |
| :--- | :--- | :--- | :--- |
| `edu1.py` | Initial data loading & shape | Loads CSV with `dtype=str`, prints `df.shape`, `head()`, `info()` | Hardcoded filename; no command-line arguments; no exception handling; loads full dataset as raw strings without validation. |
| `edu2.py` | Summary statistics | Strips `$` and `,` on `price` and `quantity`; prints summary statistics and value counts | Ignores book price columns (`bid_price_1..3`, `ask_price_1..3`); modifies subset of columns; prints unformatted stdout. |
| `edu3.py` | Problem identification | Detects missing values, duplicates, timestamp regex pattern matching, negative prices, crossed books | Re-executes conversion logic from `edu2.py`; uses naive regex digit replacement (`\d -> 9`); no programmatic logging or structured JSON audit report. |
| `edu4.py` | Data cleaning & export | Removes duplicates; parses timestamps; standardizes symbols; cleans numbers; imputes missing `side` via quote rule; drops remaining nulls; removes crossed books; corrects 100x scale error; saves `trades_clean.csv` | **Redundant disk I/O**: loads dirty file from scratch; hardcoded 2% gap filter (`gap <= 0.02`) without documenting reasons; drops rows with missing L1 quotes rather than interpolating; loses 65 records without comprehensive audit trail. |
| `edu5.py` | Outlier removal via IQR | Computes Q1, Q3, and IQR on `trades_clean.csv` price column; removes observations outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` | **Major Financial Modeling Flaw**: Static IQR trimming on raw asset prices deletes valid market trends. Prices are non-stationary; trimming by static distribution cuts off legitimate intraday price discoveries. |
| `edu6.py` | Feature engineering | Computes `spread`, `mid_price`, `order_imbalance`, `return`, `volatility_20`, `ma_10`, `ma_30`, `rsi_14`; saves `features_tick.csv` | Re-reads CSV from disk; completely ignores Level-2 and Level-3 order book depth columns (`bid_price_2/3`, `ask_price_2/3`); does not compute true volume-weighted micro-price or multi-level book slope; lacks lookahead protection verification. |
| `edu7.py` | OHLCV resampling | Resamples trades to 1-minute frequency; aggregates OHLCV + trade counts; forward-fills empty minutes; saves `candles_1min.csv` | Re-reads CSV from disk; hardcoded 1-minute fixed interval; does not support multi-timeframe aggregation (e.g., 1s, 5s, 5m); ignores volume-weighted average price (VWAP) in bars. |
| `edu8.py` | Visualization | Generates 4-panel subplots (close price line, volume histogram, spread histogram, side bar chart) and a correlation heatmap | Low visual hierarchy; unformatted axis ticks; saves figures to root directory (`step8_charts.png`, `step8_heatmap.png`); does not display interactive depth charts, candlestick charts, or order book dynamics. |
| `edu9.py` | Train/Test Split & Scaling | Computes 5-minute forward return target (`shift(-5)`); performs chronological 70/30 split; applies z-score scaling using training parameters only; saves `train_scaled.csv` and `test_scaled.csv` | Fixed 5-minute horizon; fixed 70/30 split; no cross-validation; no model training, baseline evaluation, or feature importance diagnostics. |

---

## 4. Strengths vs. Weaknesses Matrix

### 4.1 Identified Strengths
- **Sound Microstructure Heuristics:** Implemented the quote rule (Lee-Ready heuristic) in `edu4.py` to impute missing trade direction (`price >= ask -> buy`, `price <= bid -> sell`).
- **Domain-Specific Sanity Checks:** Recognized crossed books (`bid > ask`) as an anomaly requiring filtering.
- **Time-Series Integrity in Splitting:** Correctly avoided random shuffling for time-series data; applied chronological 70/30 split in `edu9.py` and fitted scalers strictly on training data to prevent information leakage.

### 4.2 Critical Weaknesses & Risks
1. **Missing Data File in Upstream Repository:** Upstream repo cannot run out of the box because `trades_orderbook_STUDENT_dirty.csv` was omitted from git commits.
2. **Procedural Script Fragmentation:** 9 individual scripts requiring manual sequential execution. Intermediate dataframes are repeatedly serialized and deserialized to CSV without compression or caching.
3. **Flawed Statistical Treatment of Prices (Static IQR):** In `edu5.py`, static IQR applied to price levels discards legitimate price rallies or dips. In financial time series, outlier detection must operate on price innovations (returns, spreads, or deviations from rolling mid-price), not raw absolute price levels.
4. **Underutilization of Available Features:** The dataset provides Level-1, Level-2, and Level-3 order book quotes and volumes, yet the reference project completely discarded Level-2 and Level-3, wasting valuable order book depth information.
5. **No Modular Architecture / Library:** No functions or classes; no reusable `src/` modules; impossible to import into downstream analytical or trading environments.
6. **No Test Suite or Type Hints:** Zero unit tests, zero integration tests, zero schema validation contracts.
7. **Basic, Low-Information Plots:** Primitive matplotlib charts with overlapping dates, default color palettes, and no interactive or candlestick visualizations.
8. **No Documentation or Attribution:** Missing README, documentation, requirements.txt, license, and change history.

---

## 5. Modernization & Transformation Strategy

To elevate this reference project into a production-grade, reproducible research asset, the following architectural and analytical pillars are established:

1. **Modular Architecture (`src/`):**
   - Reusable `DataLoader` with schema validation and error reporting.
   - Robust `DataCleaner` with explicit audit trails and non-destructive outlier handling.
   - High-performance `MicrostructureFeatureExtractor` computing multi-level depth imbalance, Stoikov micro-price, book slope, VWAP, and rolling volatility.
   - Multi-resolution `OHLCVResampler` supporting arbitrary timeframes (1s, 5s, 1min, 5min).
   - `StatisticalAnalyzer` executing rigorous econometric tests (ADF stationarity, Kyle's lambda price impact, order flow imbalance regressions, ARCH effect).
   - Unified `VisualizationSuite` producing publication-ready static and interactive graphics.
2. **Reproducible Pipeline & CLI:**
   - Single command entry point (`python -m src.cli --run-all` or `python run_analysis.py`).
   - Pinned `pyproject.toml` and `requirements.txt`.
3. **Interactive Visual Dashboard & API:**
   - Streamlit application displaying interactive LOB depth ladders, candlestick charts with volume bars, and feature correlation explorers.
   - FastAPI REST API for streaming trade evaluation and micro-price inference.
4. **Interactive Jupyter Research Notebooks:**
   - `01_data_quality_audit.ipynb`: Visual and tabular exploration of dirty data artifacts.
   - `02_exploratory_analysis.ipynb`: Comprehensive univariate, bivariate, and order book dynamics exploration.
   - `03_advanced_insights.ipynb`: Deep econometric analysis of price impact, micro-price lead-lag, and predictive modeling.
5. **Full Test Suite & Quality Verification:**
   - Comprehensive unit and integration tests covering all modules with edge cases and zero test failures.
