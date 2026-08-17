"""Data quality rules and validation.

Ensures data integrity before it enters the research pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass
class QualityIssue:
    """A data quality issue found during validation."""

    rule_id: str
    severity: str  # "error" | "warning" | "info"
    symbol: str
    description: str
    affected_rows: int = 0
    details: dict = field(default_factory=dict)


@dataclass
class QualityReport:
    """Report of all quality checks for a dataset."""

    total_rows: int
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


class DataQualityChecker:
    """Check data quality for daily bar data."""

    def check_daily_bars(self, df: pd.DataFrame) -> QualityReport:
        """Run all quality checks on daily bar data."""
        report = QualityReport(total_rows=len(df))

        if len(df) == 0:
            report.issues.append(
                QualityIssue(
                    rule_id="EMPTY_DATA",
                    severity="warning",
                    symbol="*",
                    description="Empty DataFrame",
                )
            )
            return report

        self._check_missing_columns(df, report)
        self._check_price_validity(df, report)
        self._check_volume_validity(df, report)
        self._check_date_continuity(df, report)
        self._check_duplicates(df, report)
        self._check_price_jumps(df, report)

        return report

    def _check_missing_columns(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Check for required columns."""
        required = ["symbol", "trade_date", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            report.issues.append(
                QualityIssue(
                    rule_id="MISSING_COLUMNS",
                    severity="error",
                    symbol="*",
                    description=f"Missing required columns: {missing}",
                )
            )

    def _check_price_validity(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Check price data validity."""
        if "close" not in df.columns:
            return

        # Non-positive prices
        invalid = df[df["close"] <= 0]
        if len(invalid) > 0:
            report.issues.append(
                QualityIssue(
                    rule_id="NON_POSITIVE_PRICE",
                    severity="error",
                    symbol="*",
                    description="Non-positive close prices found",
                    affected_rows=len(invalid),
                )
            )

        # High < Low
        if "high" in df.columns and "low" in df.columns:
            invalid_hl = df[df["high"] < df["low"]]
            if len(invalid_hl) > 0:
                report.issues.append(
                    QualityIssue(
                        rule_id="HIGH_LESS_THAN_LOW",
                        severity="error",
                        symbol="*",
                        description="High price < Low price",
                        affected_rows=len(invalid_hl),
                    )
                )

    def _check_volume_validity(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Check volume data validity."""
        if "volume" not in df.columns:
            return

        negative = df[df["volume"] < 0]
        if len(negative) > 0:
            report.issues.append(
                QualityIssue(
                    rule_id="NEGATIVE_VOLUME",
                    severity="error",
                    symbol="*",
                    description="Negative volume found",
                    affected_rows=len(negative),
                )
            )

    def _check_date_continuity(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Check for date gaps (missing trading days)."""
        if "symbol" not in df.columns or "trade_date" not in df.columns:
            return

        for symbol in df["symbol"].unique():
            sym_df = df[df["symbol"] == symbol].sort_values("trade_date")
            if len(sym_df) < 2:
                continue

            dates = sym_df["trade_date"].tolist()
            # Check for large gaps (> 5 calendar days might indicate missing data)
            for i in range(1, len(dates)):
                gap = (dates[i] - dates[i - 1]).days
                if gap > 7:  # More than a week gap
                    report.issues.append(
                        QualityIssue(
                            rule_id="DATE_GAP",
                            severity="warning",
                            symbol=symbol,
                            description=f"Gap of {gap} days between {dates[i - 1]} and {dates[i]}",
                            details={"gap_days": gap},
                        )
                    )

    def _check_duplicates(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Check for duplicate rows."""
        if "symbol" not in df.columns or "trade_date" not in df.columns:
            return

        dupes = df.duplicated(subset=["symbol", "trade_date"], keep=False)
        n_dupes = dupes.sum()
        if n_dupes > 0:
            report.issues.append(
                QualityIssue(
                    rule_id="DUPLICATE_ROWS",
                    severity="error",
                    symbol="*",
                    description=f"Duplicate symbol/date rows found",
                    affected_rows=int(n_dupes),
                )
            )

    def _check_price_jumps(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Check for abnormal price jumps (> 20% daily change)."""
        if "symbol" not in df.columns or "close" not in df.columns:
            return

        for symbol in df["symbol"].unique():
            sym_df = df[df["symbol"] == symbol].sort_values("trade_date")
            if len(sym_df) < 2:
                continue

            returns = sym_df["close"].pct_change().abs()
            large_jumps = returns[returns > 0.20]
            if len(large_jumps) > 0:
                report.issues.append(
                    QualityIssue(
                        rule_id="LARGE_PRICE_JUMP",
                        severity="warning",
                        symbol=symbol,
                        description=f"{len(large_jumps)} days with >20% price change",
                        affected_rows=len(large_jumps),
                    )
                )
