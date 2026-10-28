# Changelog

All notable changes to the **MicroQuant** high-frequency market microstructure EDA project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-10-28
### Transformed & Released by Daniel Richard Xavier

### Added
- **Core Package (`src/`):**
  - `src/data_loader.py`: Safe raw CSV loader with schema assertion and structured `ValidationReport`.
  - `src/cleaner.py`: Multi-pass data cleaning engine implementing Lee-Ready quote rule imputation, crossed-book rejection, 100x scale error rectification, and regex formatting stripping.
  - `src/features.py`: Market microstructure signal engine extracting 25 Level-1 and Level-3 signals (Stoikov micro-price, multi-level depth imbalances, book slopes, VWAP, rolling volatility).
  - `src/resampler.py`: Multi-resolution calendar-time aggregator for $1\text{s}$, $5\text{s}$, $1\text{min}$, and $5\text{min}$ OHLCV bars with gap-filling logic.
  - `src/analysis.py`: Econometric testing suite featuring Augmented Dickey-Fuller (ADF) tests, Kyle's Lambda price impact regression, Stoikov lead-lag dynamics, and distribution moment calculations.
  - `src/visualization.py`: Publication-grade plotting system generating 8 high-DPI charts.
  - `src/cli.py`: Unified CLI command runner (`python -m src.cli --run-all`).
  - `src/dashboard.py`: Interactive Streamlit multi-tab application with Plotly depth ladders and candlestick zooming.
- **Top-Level Orchestration:**
  - `run_analysis.py`: Zero-configuration top-level entrypoint executing the complete pipeline.
- **Jupyter Research Notebooks:**
  - `notebooks/01_data_quality_audit.ipynb`: Feed forensics and cleaning waterfall analysis.
  - `notebooks/02_exploratory_analysis.ipynb`: Microstructure signals, order book depth, and resampling.
  - `notebooks/03_advanced_insights.ipynb`: Econometric models, 5 core data-supported insights, and ML formulation.
- **Automated Testing Suite:**
  - Added `tests/test_src_pipeline.py` with 8 comprehensive unit and integration test fixtures.
  - Expanded total test suite to 30 tests covering data loading, cleaning, features, resampling, econometrics, models, CLI, and API endpoints.
- **Documentation Suite:**
  - `docs/REPOSITORY_AUDIT.md`: In-depth critical forensics on the upstream reference implementation.
  - `docs/ARCHITECTURE.md`: Technical architecture and system data-flow diagrams.
  - `docs/IMPROVEMENT_ROADMAP.md`: Feature-by-feature transformation comparison matrix.
  - `docs/EVALUATION_REPORT.md`: Real empirical benchmark metrics and reproducibility proofs.
  - `docs/CHANGELOG.md`: Detailed release history.
  - `docs/FINAL_REPORT.md`: Executive summary and final project deliverables.
  - `README.md`: Completely rewritten comprehensive documentation.

### Changed
- Replaced destructive static IQR price trimming (`edu5.py`) with dynamic relative mid-quote band filtering.
- Upgraded naive timestamp parsing to timezone-aware UTC datetime with millisecond precision across Unix epoch, UK slash, and ISO-8601 strings.
- Upgraded plot outputs to high-contrast, publication-grade aesthetics saved to `outputs/figures/`.
- Replaced monolithic disk read/write loops with modular memory-efficient pipeline executions.

### Fixed
- Fixed bug in micro-price calculation ensuring Level-1 volume weighting is always evaluated even when higher book depths are missing.
- Fixed NumPy/Pandas JSON serialization types in audit reports.
- Corrected 8 corrupted raw records where prices were recorded 100x above market mid-quote.

---

## [0.1.0] - 2026-10-15
### Reference Baseline Implementation (`kirankumarnscr7-cmyk/EDA-Mini-project`)

### Features
- 9 procedural scripts (`edu1.py` through `edu9.py`).
- Basic string stripping of currency characters.
- Initial calculation of simple moving averages and RSI.
- Basic 1-minute OHLCV resampling.
- Static 4-panel subplot and feature correlation heatmap.
