"""Profit & Loss attribution framework.

Decomposes returns into:
1. Market attribution (beta exposure)
2. Sector/industry attribution
3. Factor attribution
4. Timing attribution
5. Exit attribution
6. Cost attribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


# =============================================================================
# Attribution Breakdown
# =============================================================================


@dataclass
class AttributionBreakdown:
    """Breakdown of P&L into components."""

    # Core components
    market_attribution: float = 0.0  # From market movement (beta)
    sector_attribution: float = 0.0  # From sector selection
    factor_attribution: float = 0.0  # From factor exposure
    timing_attribution: float = 0.0  # From entry/exit timing
    selection_attribution: float = 0.0  # From stock selection
    exit_attribution: float = 0.0  # From exit quality
    cost_attribution: float = 0.0  # Transaction costs (negative)

    # Metadata
    total_pnl: float = 0.0
    explanation_ratio: float = 0.0  # How much is explained

    def to_dict(self) -> dict[str, float]:
        return {
            "market": self.market_attribution,
            "sector": self.sector_attribution,
            "factor": self.factor_attribution,
            "timing": self.timing_attribution,
            "selection": self.selection_attribution,
            "exit": self.exit_attribution,
            "cost": self.cost_attribution,
            "total_pnl": self.total_pnl,
            "explained": self.explained,
            "unexplained": self.unexplained,
        }

    @property
    def explained(self) -> float:
        """Total explained P&L."""
        return (
            self.market_attribution
            + self.sector_attribution
            + self.factor_attribution
            + self.timing_attribution
            + self.selection_attribution
            + self.exit_attribution
            + self.cost_attribution
        )

    @property
    def unexplained(self) -> float:
        """Unexplained residual."""
        return self.total_pnl - self.explained


# =============================================================================
# Trade-Level Attribution
# =============================================================================


@dataclass
class ResearchTradeAttribution:
    """Attribution for a single trade (legacy research-grade model).

    This is the original trade-level attribution dataclass used by the
    research protocol tests.  For the 七星V3 strategy attribution engine
    see :class:`a_share_quant.attribution.engine.TradeAttribution`.
    """

    trade_id: str
    symbol: str
    entry_date: date
    exit_date: date
    side: str  # LONG, SHORT

    # P&L components
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    costs: float = 0.0

    # Attribution
    market_return: float = 0.0  # Benchmark return during holding
    alpha: float = 0.0  # Excess return
    timing_score: float = 0.0  # -1 to 1, quality of entry timing
    exit_score: float = 0.0  # -1 to 1, quality of exit

    # MFE/MAE (Maximum Favorable/Adverse Excursion)
    mfe: float = 0.0  # Max profit during trade
    mae: float = 0.0  # Max loss during trade
    mfe_capture: float = 0.0  # How much of MFE was captured

    breakdown: AttributionBreakdown | None = None


# =============================================================================
# Attribution Engine
# =============================================================================


class ResearchAttributionEngine:
    """Calculate P&L attribution (legacy research-grade engine).

    This is the original trade-level attribution engine used by the
    research protocol tests.  For the 七星V3 strategy attribution engine
    see :class:`a_share_quant.attribution.engine.AttributionEngine`.
    """

    def __init__(self, benchmark_returns: pd.Series | None = None):
        self._benchmark = benchmark_returns

    def attribute_trade(
        self,
        entry_price: float,
        exit_price: float,
        quantity: int,
        entry_date: date,
        exit_date: date,
        costs: float,
        prices_during_hold: pd.Series | None = None,
        benchmark_during_hold: pd.Series | None = None,
    ) -> ResearchTradeAttribution:
        """Attribute a single trade."""
        # Basic P&L
        gross_pnl = (exit_price - entry_price) * quantity
        net_pnl = gross_pnl - costs

        # Market attribution
        market_return = 0.0
        if benchmark_during_hold is not None and len(benchmark_during_hold) > 1:
            market_return = (
                (benchmark_during_hold.iloc[-1] / benchmark_during_hold.iloc[0] - 1)
                * entry_price
                * quantity
            )

        # Alpha (excess return)
        alpha = gross_pnl - market_return

        # MFE/MAE
        mfe, mae = 0.0, 0.0
        mfe_capture = 0.0
        if prices_during_hold is not None and len(prices_during_hold) > 0:
            max_price = prices_during_hold.max()
            min_price = prices_during_hold.min()
            mfe = (max_price - entry_price) * quantity
            mae = (entry_price - min_price) * quantity
            if mfe > 0:
                mfe_capture = gross_pnl / mfe

        # Timing score (entry quality)
        # Positive = entered at good time (price went up after)
        timing_score = 0.0
        if prices_during_hold is not None and len(prices_during_hold) > 1:
            avg_price_after = prices_during_hold.mean()
            timing_score = (avg_price_after - entry_price) / entry_price
            timing_score = np.clip(timing_score, -1, 1)

        # Exit score
        # Positive = exited at good time (price went down after)
        exit_score = 0.0
        if mfe > 0:
            exit_score = mfe_capture * 2 - 1  # Map 0-1 to -1 to 1

        return ResearchTradeAttribution(
            trade_id=f"{entry_date}_{exit_date}",
            symbol="",
            entry_date=entry_date,
            exit_date=exit_date,
            side="LONG",
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            costs=costs,
            market_return=market_return,
            alpha=alpha,
            timing_score=timing_score,
            exit_score=exit_score,
            mfe=mfe,
            mae=mae,
            mfe_capture=mfe_capture,
        )

    def attribute_portfolio(
        self,
        trades: list[ResearchTradeAttribution],
    ) -> AttributionBreakdown:
        """Aggregate attribution across all trades."""
        if not trades:
            return AttributionBreakdown()

        total_pnl = sum(t.net_pnl for t in trades)
        total_market = sum(t.market_return for t in trades)
        total_costs = sum(t.costs for t in trades)

        # Timing attribution (weighted by P&L)
        timing_attr = sum(t.timing_score * abs(t.gross_pnl) for t in trades)
        timing_attr = timing_attr / sum(abs(t.gross_pnl) for t in trades) if trades else 0

        # Exit attribution
        exit_attr = sum(t.exit_score * abs(t.gross_pnl) for t in trades)
        exit_attr = exit_attr / sum(abs(t.gross_pnl) for t in trades) if trades else 0

        breakdown = AttributionBreakdown(
            market_attribution=total_market,
            timing_attribution=timing_attr * abs(total_pnl) * 0.1,  # Scaled
            exit_attribution=exit_attr * abs(total_pnl) * 0.1,
            cost_attribution=-total_costs,
            total_pnl=total_pnl,
        )

        # Calculate explanation ratio
        if total_pnl != 0:
            breakdown.explanation_ratio = abs(breakdown.explained / total_pnl)

        return breakdown


# =============================================================================
# Failure Reason Codes
# =============================================================================

FAILURE_REASONS: dict[str, str] = {
    "STOP_LOSS": "触发止损",
    "TAKE_PROFIT": "触发止盈",
    "TIME_EXIT": "达到最大持有期",
    "REGIME_CHANGE": "市场状态变化",
    "SIGNAL_REVERSAL": "信号反转",
    "RISK_LIMIT": "触发风控限制",
    "MARGIN_CALL": "保证金不足",
    "LIQUIDITY": "流动性不足无法退出",
    "SUSPENSION": "停牌无法交易",
    "LIMIT_UP": "涨停无法买入",
    "LIMIT_DOWN": "跌停无法卖出",
    "DATA_ERROR": "数据错误",
    "SYSTEM_ERROR": "系统错误",
    "MANUAL": "手动干预",
    "UNKNOWN": "未知原因",
}


@dataclass
class FailureAnalysis:
    """Analysis of trade failures."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    failure_reasons: dict[str, int] = field(default_factory=dict)
    avg_loss_by_reason: dict[str, float] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0

    @property
    def top_failure_reason(self) -> str:
        if not self.failure_reasons:
            return "NONE"
        return max(self.failure_reasons, key=lambda k: self.failure_reasons[k])


def analyze_failures(trades: list[dict[str, Any]]) -> FailureAnalysis:
    """Analyze trade failures."""
    if not trades:
        return FailureAnalysis(0, 0, 0)

    winning = [t for t in trades if t.get("pnl", 0) > 0]
    losing = [t for t in trades if t.get("pnl", 0) <= 0]

    # Count failure reasons
    reasons: dict[str, int] = {}
    loss_by_reason: dict[str, float] = {}

    for t in losing:
        reason = t.get("exit_reason", "UNKNOWN")
        reasons[reason] = reasons.get(reason, 0) + 1
        loss_by_reason[reason] = loss_by_reason.get(reason, 0) + abs(t.get("pnl", 0))

    # Average loss by reason
    avg_loss = {r: loss_by_reason[r] / reasons[r] for r in reasons}

    return FailureAnalysis(
        total_trades=len(trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        failure_reasons=reasons,
        avg_loss_by_reason=avg_loss,
    )


# =============================================================================
# 七星V3 Strategy Attribution (P3-E3: 盈亏归因引擎)
# =============================================================================

from a_share_quant.attribution.engine import (  # noqa: E402
    AttributionEngine,
    AttributionReport,
    TradeAttribution,
    generate_html_report,
)

__all__ = [
    "FAILURE_REASONS",
    "AttributionBreakdown",
    "AttributionEngine",
    "AttributionReport",
    "FailureAnalysis",
    "ResearchAttributionEngine",
    "ResearchTradeAttribution",
    "TradeAttribution",
    "analyze_failures",
    "generate_html_report",
]
