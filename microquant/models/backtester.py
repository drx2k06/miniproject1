"""
Quantitative Microstructure Strategy Backtester and Execution Simulator.
Evaluates model-driven trading strategies under realistic fees, slippage, and spread friction.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from microquant.core.logging import get_logger

logger = get_logger("microquant.backtester")


@dataclass
class BacktestMetrics:
    """Quantitative performance and risk metrics."""
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MicrostructureBacktester:
    """
    Simulates high-frequency algorithmic trade execution with cost models.
    """

    def __init__(
        self,
        maker_fee_bps: float = 1.0,
        taker_fee_bps: float = 4.0,
        slippage_bps: float = 1.5,
        initial_capital: float = 100000.0,
        long_threshold: float = 0.52,
        short_threshold: float = 0.48,
    ):
        self.maker_fee_rate = maker_fee_bps / 10000.0
        self.taker_fee_rate = taker_fee_bps / 10000.0
        self.slippage_rate = slippage_bps / 10000.0
        self.initial_capital = initial_capital
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def run_backtest(
        self,
        probabilities: np.ndarray,
        prices: np.ndarray,
        spreads: Optional[np.ndarray] = None,
        timestamps: Optional[pd.Series] = None,
    ) -> Tuple[BacktestMetrics, pd.DataFrame]:
        """
        Executes strategy simulation given model upward probabilities and execution prices.

        Args:
            probabilities: Array of predicted P(up) for each test tick.
            prices: Array of execution prices (trade price or mid-price).
            spreads: Array of bid-ask spreads for tick execution cost.
            timestamps: Series or array of test timestamps.

        Returns:
            Tuple of (BacktestMetrics, equity_curve_df).
        """
        n = len(prices)
        logger.info(f"Running high-frequency backtest across {n} ticks...")

        # Derive desired target positions: +1 (Long), -1 (Short), 0 (Neutral)
        positions = np.zeros(n)
        positions[probabilities >= self.long_threshold] = 1.0
        positions[probabilities <= self.short_threshold] = -1.0

        # One tick execution lag: order placed at t executes at t+1
        pos_held = np.roll(positions, 1)
        pos_held[0] = 0.0

        # Price percentage returns
        price_returns = np.zeros(n)
        price_returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]

        # Gross strategy return = position * price return
        gross_returns = pos_held * price_returns

        # Friction costs: fees and slippage incurred when position changes
        position_changes = np.abs(pos_held - np.roll(pos_held, 1))
        position_changes[0] = np.abs(pos_held[0])

        half_spread_rate = (spreads / (2.0 * prices)) if spreads is not None else np.full(n, 0.0001)
        per_trade_cost_rate = self.taker_fee_rate + self.slippage_rate + half_spread_rate

        costs = position_changes * per_trade_cost_rate
        net_returns = gross_returns - costs

        # Cumulative compounding equity curve
        equity_curve = self.initial_capital * np.cumprod(1.0 + net_returns)
        benchmark_curve = self.initial_capital * (prices / prices[0])

        total_return_pct = (equity_curve[-1] / self.initial_capital - 1.0) * 100.0
        benchmark_return_pct = (benchmark_curve[-1] / self.initial_capital - 1.0) * 100.0
        alpha_pct = total_return_pct - benchmark_return_pct

        # Maximum Drawdown calculation
        running_peak = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_peak) / running_peak
        max_drawdown_pct = float(np.min(drawdowns)) * 100.0

        # Risk-adjusted metrics
        mean_ret = np.mean(net_returns)
        std_ret = np.std(net_returns)
        annualization_factor = np.sqrt(252 * 6.5 * 60 * 2)  # High-frequency tick scaling proxy

        sharpe = float((mean_ret / (std_ret + 1e-9)) * annualization_factor) if std_ret > 0 else 0.0

        downside_returns = net_returns[net_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-9
        sortino = float((mean_ret / (downside_std + 1e-9)) * annualization_factor)

        # Trade analytics
        trade_mask = position_changes > 0
        total_trades = int(np.sum(trade_mask))

        # Individual trade PnL segments
        gains = net_returns[net_returns > 0]
        losses = net_returns[net_returns < 0]
        win_rate = float(len(gains) / max(1, len(gains) + len(losses)) * 100.0)
        profit_factor = float(np.sum(gains) / max(1e-9, np.abs(np.sum(losses))))

        total_fees = float(np.sum(position_changes * self.taker_fee_rate * equity_curve))
        total_slippage = float(np.sum(position_changes * self.slippage_rate * equity_curve))

        metrics = BacktestMetrics(
            initial_capital=self.initial_capital,
            final_equity=round(float(equity_curve[-1]), 2),
            total_return_pct=round(total_return_pct, 2),
            benchmark_return_pct=round(benchmark_return_pct, 2),
            alpha_pct=round(alpha_pct, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            max_drawdown_pct=round(max_drawdown_pct, 2),
            total_trades=total_trades,
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            total_fees_paid=round(total_fees, 2),
            total_slippage_paid=round(total_slippage, 2),
        )

        history_df = pd.DataFrame({
            "timestamp": timestamps if timestamps is not None else np.arange(n),
            "price": prices,
            "probability_up": probabilities,
            "position": pos_held,
            "net_return": net_returns,
            "equity": equity_curve,
            "benchmark_equity": benchmark_curve,
            "drawdown": drawdowns,
        })

        logger.info(
            f"Backtest completed: Return={metrics.total_return_pct}% | Alpha={metrics.alpha_pct}% | "
            f"Sharpe={metrics.sharpe_ratio} | MDD={metrics.max_drawdown_pct}% | Trades={metrics.total_trades}"
        )
        return metrics, history_df
