"""
MicroQuant Configuration and Settings Management.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration settings with environment override support."""

    # Project Information
    APP_NAME: str = "Aetheris-MicroQuant"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Directory Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_PATH: Path = DATA_DIR / "raw" / "trades_orderbook_STUDENT_dirty.csv"
    CLEAN_DATA_PATH: Path = DATA_DIR / "processed" / "trades_clean.csv"
    FEATURES_DATA_PATH: Path = DATA_DIR / "processed" / "features_tick.csv"
    CANDLES_1MIN_PATH: Path = DATA_DIR / "processed" / "candles_1min.csv"
    CANDLES_5S_PATH: Path = DATA_DIR / "processed" / "candles_5s.csv"
    DATASET_ML_PATH: Path = DATA_DIR / "processed" / "dataset_ml_ticks.csv"
    MODEL_DIR: Path = DATA_DIR / "models"

    # API Server Settings
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    API_RELOAD: bool = False

    # Streamlit UI Settings
    STREAMLIT_PORT: int = 8501

    # Machine Learning & Alpha Settings
    TICK_HORIZON: int = 10
    CANDLE_HORIZON: int = 5
    TRAIN_TEST_SPLIT_RATIO: float = 0.75
    RANDOM_STATE: int = 42
    PYTORCH_EPOCHS: int = 40
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001

    # Backtesting Parameters
    MAKER_FEE_BPS: float = 1.0     # 1 basis point (0.01%)
    TAKER_FEE_BPS: float = 4.0     # 4 basis points (0.04%)
    SLIPPAGE_BPS: float = 1.5      # 1.5 basis points slippage
    INITIAL_CAPITAL: float = 100000.0  # $100,000 USD initial capital

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_directories(self) -> None:
        """Ensures that required data and model storage directories exist."""
        (self.DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
