"""DataProvider protocol and base classes.

All data sources must implement this protocol to ensure:
1. Strategy code never directly calls third-party SDKs
2. All data maps to unified internal schema
3. MockProvider enables development without external dependencies
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

    from a_share_quant.data.schemas import SecurityInfo


@runtime_checkable
class DataProvider(Protocol):
    """Protocol for data providers.

    All data providers must implement this interface.
    Data is returned as pandas DataFrames with standardized columns.
    """

    @property
    def name(self) -> str:
        """Provider name for logging and debugging."""
        ...

    def get_trading_calendar(
        self,
        exchange: str = "SSE",
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> pd.DataFrame:
        """Get trading calendar.

        Returns DataFrame with columns: [trade_date, is_open]
        """
        ...

    def get_security_list(
        self,
        security_type: str | None = None,
        exchange: str | None = None,
        include_delisted: bool = False,
    ) -> pd.DataFrame:
        """Get list of securities.

        Returns DataFrame with SecurityInfo columns.
        """
        ...

    def get_security_info(self, symbol: str) -> SecurityInfo | None:
        """Get info for a specific security."""
        ...

    def get_daily_bars(
        self,
        symbols: str | list[str],
        start_date: date | str,
        end_date: date | str,
        adjust: str = "qfq",  # qfq, hfq, none
    ) -> pd.DataFrame:
        """Get daily OHLCV bars.

        Returns DataFrame with columns:
        [symbol, trade_date, open, high, low, close, volume, amount, ...]
        """
        ...

    def get_adjustment_factors(
        self,
        symbols: str | list[str],
        start_date: date | str,
        end_date: date | str,
    ) -> pd.DataFrame:
        """Get price adjustment factors.

        Returns DataFrame with columns: [symbol, trade_date, factor]
        """
        ...

    def get_suspension_info(
        self,
        symbols: str | list[str],
        start_date: date | str,
        end_date: date | str,
    ) -> pd.DataFrame:
        """Get suspension information.

        Returns DataFrame with columns: [symbol, suspend_date, resume_date, reason]
        """
        ...

    def get_index_daily(
        self,
        symbols: str | list[str],
        start_date: date | str,
        end_date: date | str,
    ) -> pd.DataFrame:
        """Get index daily data.

        Returns DataFrame with daily bar columns.
        """
        ...

    def get_industry_classification(
        self,
        symbols: str | list[str] | None = None,
        level: int = 1,
    ) -> pd.DataFrame:
        """Get industry classification.

        Returns DataFrame with columns: [symbol, industry_code, industry_name, level]
        """
        ...

    def is_trading_day(self, check_date: date | str, exchange: str = "SSE") -> bool:
        """Check if a date is a trading day."""
        ...

    def get_next_trading_day(
        self,
        from_date: date | str,
        exchange: str = "SSE",
    ) -> date:
        """Get the next trading day after from_date."""
        ...

    def get_prev_trading_day(
        self,
        from_date: date | str,
        exchange: str = "SSE",
    ) -> date:
        """Get the previous trading day before from_date."""
        ...


class BaseDataProvider(ABC):
    """Base class for data providers with common utilities."""

    def __init__(self) -> None:
        self._calendar_cache: dict[str, pd.DataFrame] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...

    def _normalize_date(self, d: date | str) -> date:
        """Normalize date input to date object."""
        if isinstance(d, str):
            clean = d.replace("-", "")
            return date(int(clean[:4]), int(clean[4:6]), int(clean[6:8]))
        return d

    def _normalize_symbols(self, symbols: str | list[str]) -> list[str]:
        """Normalize symbols input to list."""
        if isinstance(symbols, str):
            return [symbols]
        return symbols

    def _validate_date_range(
        self,
        start_date: date | str,
        end_date: date | str,
    ) -> tuple[date, date]:
        """Validate and normalize date range."""
        start = self._normalize_date(start_date)
        end = self._normalize_date(end_date)
        if start > end:
            msg = f"start_date {start} > end_date {end}"
            raise ValueError(msg)
        return start, end
