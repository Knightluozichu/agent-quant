"""Report generation for backtest results.

Generates:
1. Performance summary reports
2. Trade logs
3. Equity curve charts (data export)
4. Risk metrics reports
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from a_share_quant.backtest import BacktestResult, PerformanceMetrics


@dataclass
class BacktestReport:
    """Complete backtest report."""

    run_id: str
    timestamp: datetime
    config: dict
    metrics: dict
    trades: list[dict]
    equity_curve: list[dict]
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Export report as JSON."""
        return json.dumps(
            {
                "run_id": self.run_id,
                "timestamp": self.timestamp.isoformat(),
                "config": self.config,
                "metrics": self.metrics,
                "trades": self.trades,
                "equity_curve": self.equity_curve,
                "metadata": self.metadata,
            },
            indent=2,
            ensure_ascii=False,
        )

    def to_dict(self) -> dict:
        """Export report as dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "config": self.config,
            "metrics": self.metrics,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "metadata": self.metadata,
        }


class ReportGenerator:
    """Generate reports from backtest results."""

    def __init__(self, output_dir: Path | str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        result: BacktestResult,
        run_id: Optional[str] = None,
    ) -> BacktestReport:
        """Generate a complete report from backtest result."""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert equity curve to list of dicts
        equity_data = []
        for dt, value in result.equity_curve.items():
            equity_data.append(
                {
                    "date": dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
                    "equity": float(value),
                }
            )

        report = BacktestReport(
            run_id=run_id,
            timestamp=datetime.now(),
            config={
                "start_date": result.config.start_date.isoformat(),
                "end_date": result.config.end_date.isoformat(),
                "initial_capital": result.config.initial_capital,
                "symbols": result.config.symbols,
                "benchmark": result.config.benchmark,
            },
            metrics=result.metrics.to_dict(),
            trades=result.trades,
            equity_curve=equity_data,
            metadata={
                "total_trading_days": len(result.equity_curve),
                "final_equity": float(result.equity_curve.iloc[-1])
                if len(result.equity_curve) > 0
                else 0,
            },
        )

        return report

    def save_json(self, report: BacktestReport, filename: Optional[str] = None) -> Path:
        """Save report as JSON file."""
        if filename is None:
            filename = f"backtest_{report.run_id}.json"

        filepath = self.output_dir / filename
        filepath.write_text(report.to_json(), encoding="utf-8")
        return filepath

    def save_csv(self, report: BacktestReport) -> dict[str, Path]:
        """Save report data as CSV files."""
        files = {}

        # Equity curve
        if report.equity_curve:
            equity_df = pd.DataFrame(report.equity_curve)
            equity_path = self.output_dir / f"equity_{report.run_id}.csv"
            equity_df.to_csv(equity_path, index=False)
            files["equity"] = equity_path

        # Trades
        if report.trades:
            trades_df = pd.DataFrame(report.trades)
            trades_path = self.output_dir / f"trades_{report.run_id}.csv"
            trades_df.to_csv(trades_path, index=False)
            files["trades"] = trades_path

        return files

    def generate_summary(self, report: BacktestReport) -> str:
        """Generate human-readable summary."""
        m = report.metrics
        c = report.config

        lines = [
            "=" * 60,
            f"回测报告: {report.run_id}",
            "=" * 60,
            "",
            "【配置】",
            f"  回测期间: {c['start_date']} → {c['end_date']}",
            f"  初始资金: {c['initial_capital']:,.0f}",
            f"  标的: {', '.join(c['symbols'])}",
            "",
            "【绩效指标】",
            f"  总收益率:   {m['total_return']:.2%}",
            f"  年化收益率: {m['annual_return']:.2%}",
            f"  波动率:     {m['volatility']:.2%}",
            f"  夏普比率:   {m['sharpe_ratio']:.2f}",
            f"  最大回撤:   {m['max_drawdown']:.2%}",
            f"  胜率:       {m['win_rate']:.2%}",
            f"  盈亏比:     {m['profit_factor']:.2f}",
            f"  交易次数:   {m['total_trades']}",
            f"  平均持有:   {m['avg_holding_days']:.1f} 天",
            "",
            "【资金】",
            f"  期末资金: {report.metadata.get('final_equity', 0):,.0f}",
            "",
            "=" * 60,
        ]

        return "\n".join(lines)

    def save_summary(self, report: BacktestReport, filename: Optional[str] = None) -> Path:
        """Save summary as text file."""
        if filename is None:
            filename = f"summary_{report.run_id}.txt"

        filepath = self.output_dir / filename
        filepath.write_text(self.generate_summary(report), encoding="utf-8")
        return filepath


def create_report(
    result: BacktestResult,
    output_dir: str = "reports",
    save: bool = True,
) -> BacktestReport:
    """Convenience function to create and optionally save a report."""
    generator = ReportGenerator(output_dir)
    report = generator.generate(result)

    if save:
        generator.save_json(report)
        generator.save_csv(report)
        generator.save_summary(report)

    return report
