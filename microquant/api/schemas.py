"""
Pydantic Schemas for MicroQuant REST API.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str


class OrderBookTickRequest(BaseModel):
    trade_id: Optional[str] = "109999"
    timestamp: Optional[str] = "2026-10-01 09:25:00.000"
    symbol: str = "BTCUSDT"
    side: str = "buy"
    price: float = Field(..., gt=0, description="Trade execution price")
    quantity: float = Field(..., gt=0, description="Trade quantity")
    bid_price_1: float = Field(..., gt=0)
    bid_qty_1: float = Field(..., gt=0)
    ask_price_1: float = Field(..., gt=0)
    ask_qty_1: float = Field(..., gt=0)
    bid_price_2: Optional[float] = None
    bid_qty_2: Optional[float] = None
    ask_price_2: Optional[float] = None
    ask_qty_2: Optional[float] = None
    bid_price_3: Optional[float] = None
    bid_qty_3: Optional[float] = None
    ask_price_3: Optional[float] = None
    ask_qty_3: Optional[float] = None


class OrderBookPredictionResponse(BaseModel):
    symbol: str
    model_name: str
    predicted_direction: str  # "UP" or "DOWN"
    probability_up: float
    confidence_pct: float
    spread_bps: float
    order_imbalance_1: float
    micro_price: float
    latency_ms: float


class PipelineRunResponse(BaseModel):
    status: str
    message: str
    cleaning_report: Dict[str, Any]
    feature_count: int
    candles_1min_count: int
    candles_5s_count: int
    model_metrics: Dict[str, Any]


class BacktestRequest(BaseModel):
    maker_fee_bps: float = 1.0
    taker_fee_bps: float = 4.0
    slippage_bps: float = 1.5
    initial_capital: float = 100000.0
    long_threshold: float = 0.52
    short_threshold: float = 0.48
    model_name: str = "gradient_boosting"


class BacktestResponse(BaseModel):
    model_name: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    total_fees_paid: float
    total_slippage_paid: float
    equity_curve: List[Dict[str, Any]]
