"""Data loader for visualization dashboard.

Bridges existing modules (regime, strategies, risk, evolution) to the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd

from a_share_quant.data.providers import get_data_provider
from a_share_quant.regime import RegimeDetector, RegimeState


@dataclass
class MarketStateData:
    """Market state data for dashboard."""

    current_state: str
    current_state_label: str
    confidence: float
    state_distribution: dict[str, int]
    state_history: list[dict[str, Any]]
    recommended_strategy: str


@dataclass
class PositionData:
    """Position data for dashboard."""

    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    days_held: int = 0


@dataclass
class TradeSignalData:
    """Trade signal data for dashboard."""

    symbol: str
    action: str  # BUY, SELL, HOLD
    strategy: str
    reason: str
    confidence: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    rejected: bool = False
    reject_reason: str = ""


@dataclass
class EvolutionData:
    """Champion/Challenger data for dashboard."""

    champion_name: str
    challenger_name: Optional[str]
    champion_metrics: dict[str, float]
    challenger_metrics: dict[str, float]
    promotion_progress: float  # 0-1
    min_trades_required: int
    current_trades: int


class DashboardDataLoader:
    """Load and transform data for dashboard display."""

    def __init__(self, provider_type: str = "mock"):
        self.provider = get_data_provider(provider_type)
        self.regime_detector = RegimeDetector()
        self._cache: dict[str, Any] = {}

    def get_market_state(
        self,
        symbol: str = "510300.SSE",
        lookback_days: int = 120,
    ) -> MarketStateData:
        """Get current market state and history.

        Args:
            symbol: Index symbol to analyze
            lookback_days: Days of history to analyze
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        # Get historical data
        df = self.provider.get_daily_bars(symbol, start_date, end_date)

        if df.empty:
            return MarketStateData(
                current_state="FLAT_MEDIUM",
                current_state_label="震荡 + 中波动",
                confidence=0.5,
                state_distribution={},
                state_history=[],
                recommended_strategy="RANGE_MEAN_REVERSION",
            )

        # Detect current regime
        current_regime = self.regime_detector.detect(df, end_date)

        # Build state history
        state_history = []
        state_distribution: dict[str, int] = {}

        # Sample every 5 days for efficiency
        sample_dates = df["trade_date"].iloc[::5].tolist()
        for d in sample_dates:
            regime = self.regime_detector.detect(df, d)
            state_key = f"{regime.direction.value}_{regime.oscillation.value}"
            state_history.append({
                "date": d,
                "state": state_key,
                "direction": regime.direction.value,
                "oscillation": regime.oscillation.value,
            })
            state_distribution[state_key] = state_distribution.get(state_key, 0) + 1

        # Current state
        current_state = f"{current_regime.direction.value}_{current_regime.oscillation.value}"

        # Recommended strategy
        from a_share_quant.regime import REGIME_STRATEGY_MAP
        recommended_list = REGIME_STRATEGY_MAP.get(current_regime.state_id, ["CASH_DEFENSE"])
        recommended = recommended_list[0] if recommended_list else "CASH_DEFENSE"

        return MarketStateData(
            current_state=current_state,
            current_state_label=self._get_state_label(current_state),
            confidence=0.75,  # Placeholder
            state_distribution=state_distribution,
            state_history=state_history,
            recommended_strategy=recommended,
        )

    def get_positions(self) -> list[PositionData]:
        """Get current positions.

        Returns mock data for now - will connect to PaperBroker later.
        """
        # Mock positions for demonstration
        return [
            PositionData(
                symbol="510300.SSE",
                quantity=10000,
                avg_cost=3.85,
                current_price=4.02,
                market_value=40200,
                unrealized_pnl=1700,
                unrealized_pnl_pct=0.044,
                stop_loss=3.65,
                take_profit=4.30,
                days_held=15,
            ),
            PositionData(
                symbol="159915.SZSE",
                quantity=5000,
                avg_cost=2.15,
                current_price=2.08,
                market_value=10400,
                unrealized_pnl=-350,
                unrealized_pnl_pct=-0.033,
                stop_loss=2.00,
                take_profit=2.40,
                days_held=8,
            ),
        ]

    def get_trade_signals(self) -> list[TradeSignalData]:
        """Get today's trade signals.

        Returns mock data for now - will connect to strategy engine later.
        """
        return [
            TradeSignalData(
                symbol="510300.SSE",
                action="HOLD",
                strategy="TREND_HOLD",
                reason="趋势延续，继续持有",
                confidence=0.8,
            ),
            TradeSignalData(
                symbol="159915.SZSE",
                action="SELL",
                strategy="PULLBACK_SWING",
                reason="触及止损线",
                confidence=0.9,
                stop_loss=2.00,
            ),
            TradeSignalData(
                symbol="510500.SSE",
                action="BUY",
                strategy="RANGE_MEAN_REVERSION",
                reason="回调至支撑位",
                confidence=0.7,
                target_price=6.20,
                stop_loss=5.80,
                rejected=True,
                reject_reason="MAX_POSITIONS_REACHED",
            ),
        ]

    def get_evolution_data(self) -> EvolutionData:
        """Get Champion/Challenger evolution data.

        Returns mock data for now - will connect to EvolutionManager later.
        """
        return EvolutionData(
            champion_name="TREND_HOLD_v1.2",
            challenger_name="TREND_HOLD_v1.3",
            champion_metrics={
                "sharpe": 1.25,
                "max_drawdown": -0.12,
                "win_rate": 0.58,
                "total_return": 0.185,
                "trades": 45,
            },
            challenger_metrics={
                "sharpe": 1.42,
                "max_drawdown": -0.09,
                "win_rate": 0.62,
                "total_return": 0.215,
                "trades": 38,
            },
            promotion_progress=0.76,
            min_trades_required=50,
            current_trades=38,
        )

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio summary metrics."""
        positions = self.get_positions()
        total_value = sum(p.market_value for p in positions)
        total_pnl = sum(p.unrealized_pnl for p in positions)
        total_cost = sum(p.avg_cost * p.quantity for p in positions)

        return {
            "total_market_value": total_value,
            "total_unrealized_pnl": total_pnl,
            "total_unrealized_pnl_pct": total_pnl / total_cost if total_cost > 0 else 0,
            "position_count": len(positions),
            "cash": 50000,  # Mock
            "total_assets": total_value + 50000,
        }

    @staticmethod
    def _get_state_label(state: str) -> str:
        """Get Chinese label for state code."""
        labels = {
            "UP_LOW": "上涨 + 低波动",
            "UP_MEDIUM": "上涨 + 中波动",
            "UP_HIGH": "上涨 + 高波动",
            "FLAT_LOW": "震荡 + 低波动",
            "FLAT_MEDIUM": "震荡 + 中波动",
            "FLAT_HIGH": "震荡 + 高波动",
            "DOWN_LOW": "下跌 + 低波动",
            "DOWN_MEDIUM": "下跌 + 中波动",
            "DOWN_HIGH": "下跌 + 高波动",
        }
        return labels.get(state, state)
