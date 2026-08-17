"""Local Parquet data provider.

Reads data from local Parquet files organized in a layered storage structure:
- data/raw/       : Original downloaded data
- data/bronze/    : Cleaned, deduplicated
- data/silver/    : Standardized schema
- data/gold/      : Feature-enriched, ready for research
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from a_share_quant.data.providers.base import BaseDataProvider
from a_share_quant.data.schemas import SecurityInfo

if TYPE_CHECKING:
    from datetime import date


class LocalParquetProvider(BaseDataProvider):
    """Read data from local Parquet files.

    Expected directory structure:
        data_root/
        ├── silver/
        │   ├── daily_bars/
        │   │   ├── 510300.SSE.parquet
        │   │   └── ...
        │   ├── security_master.parquet
        │   ├── trading_calendar.parquet
        │   ├── adjustment_factors.parquet
        │   └── suspensions.parquet
        └── gold/
            └── ...
    """

    def __init__(self, data_root: Path | str = "data", layer: str = "silver") -> None:
        super().__init__()
        self._data_root = Path(data_root)
        self._layer = layer
        self._layer_path = self._data_root / layer

    @property
    def name(self) -> str:
        return f"local_parquet_{self._layer}"

    def _read_parquet(self, relative_path: str) -> pd.DataFrame:
        """Read a parquet file from the data layer."""
        path = self._layer_path / relative_path
        if not path.exists():
            msg = f"Parquet file not found: {path}"
            raise FileNotFoundError(msg)
        return pd.read_parquet(path)

    def _read_symbol_parquet(self, subdir: str, symbol: str) -> pd.DataFrame | None:
        """Read a per-symbol parquet file."""
        path = self._layer_path / subdir / f"{symbol}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def get_trading_calendar(
        self,
        exchange: str = "SSE",
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> pd.DataFrame:
        """Get trading calendar from parquet."""
        df = self._read_parquet("trading_calendar.parquet")

        if "exchange" in df.columns:
            df = df[df["exchange"] == exchange]

        if start_date:
            start = self._normalize_date(start_date)
            df = df[df["trade_date"] >= start]
        if end_date:
            end = self._normalize_date(end_date)
            df = df[df["trade_date"] <= end]

        return df.reset_index(drop=True)

    def get_security_list(
        self,
        security_type: str | None = None,
        exchange: str | None = None,
        include_delisted: bool = False,
    ) -> pd.DataFrame:
        """Get security list from parquet."""
        df = self._read_parquet("security_master.parquet")

        if security_type:
            df = df[df["security_type"] == security_type]
        if exchange:
            df = df[df["exchange"] == exchange]
        if not include_delisted and "delist_date" in df.columns:
            df = df[df["delist_date"].isna()]

        return df.reset_index(drop=True)

    def get_security_info(self, symbol: str) -> SecurityInfo | None:
        """Get security info."""
        try:
            df = self.get_security_list(include_delisted=True)
            row = df[df["symbol"] == symbol]
            if row.empty:
                return None
            return SecurityInfo(**row.iloc[0].to_dict())
        except FileNotFoundError:
            return None

    def get_daily_bars(
        self,
        symbols: str | list[str],
        start_date: date | str,
        end_date: date | str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """Get daily bars from per-symbol parquet files."""
        start, end = self._validate_date_range(start_date, end_date)
        symbol_list = self._normalize_symbols(symbols)

        dfs = []
        for symbol in symbol_list:
            df = self._read_symbol_parquet("daily_bars", symbol)
            if df is None:
                continue

            # Filter date range
            df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        return result.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    def get_adjustment_factors(
        self,
        symbols: str | list[str],
        start_date: date | str,
        end_date: date | str,
    ) -> pd.DataFrame:
        """Get adjustment factors."""
        start, end = self._validate_date_range(start_date, end_date)
        symbol_list = self._normalize_symbols(symbols)

        try:
            df = self._read_parquet("adjustment_factors.parquet")
            df = df[
                (df["symbol"].isin(symbol_list))
                & (df["trade_date"] >= start)
                & (df["trade_date"] <= end)
            ]
            return df.reset_index(drop=True)
        except FileNotFoundError:
            return pd.DataFrame(columns=["symbol", "trade_date", "factor"])

    def get_suspension_info(
        self,
        symbols: str | list[str],
        start_date: date | str,
        end_date: date | str,
    ) -> pd.DataFrame:
        """Get suspension info."""
        start, end = self._validate_date_range(start_date, end_date)
        symbol_list = self._normalize_symbols(symbols)

        try:
            df = self._read_parquet("suspensions.parquet")
            df = df[
                (df["symbol"].isin(symbol_list))
                & (df["suspend_date"] >= start)
                & (df["suspend_date"] <= end)
            ]
            return df.reset_index(drop=True)
        except FileNotFoundError:
            return pd.DataFrame(columns=["symbol", "suspend_date", "resume_date", "reason"])

    def get_index_daily(
        self,
        symbols: str | list[str],
        start_date: date | str,
        end_date: date | str,
    ) -> pd.DataFrame:
        """Get index daily data."""
        return self.get_daily_bars(symbols, start_date, end_date)

    def get_industry_classification(
        self,
        symbols: str | list[str] | None = None,
        level: int = 1,
    ) -> pd.DataFrame:
        """Get industry classification."""
        try:
            df = self._read_parquet("industry_classification.parquet")
            if symbols:
                symbol_list = self._normalize_symbols(symbols)
                df = df[df["symbol"].isin(symbol_list)]
            if "level" in df.columns:
                df = df[df["level"] == level]
            return df.reset_index(drop=True)
        except FileNotFoundError:
            return pd.DataFrame(columns=["symbol", "industry_code", "industry_name", "level"])

    def is_trading_day(self, check_date: date | str, exchange: str = "SSE") -> bool:
        """Check if a date is a trading day."""
        d = self._normalize_date(check_date)
        try:
            cal = self.get_trading_calendar(exchange, d, d)
            return len(cal) > 0 and cal.iloc[0]["is_open"]
        except FileNotFoundError:
            # Fallback: weekdays are trading days
            return d.weekday() < 5

    def get_next_trading_day(
        self,
        from_date: date | str,
        exchange: str = "SSE",
    ) -> date:
        """Get the next trading day."""
        from datetime import timedelta

        d = self._normalize_date(from_date)
        for i in range(1, 30):  # Search up to 30 days
            candidate = d + timedelta(days=i)
            if self.is_trading_day(candidate, exchange):
                return candidate
        msg = f"No trading day found within 30 days after {d}"
        raise ValueError(msg)

    def get_prev_trading_day(
        self,
        from_date: date | str,
        exchange: str = "SSE",
    ) -> date:
        """Get the previous trading day."""
        from datetime import timedelta

        d = self._normalize_date(from_date)
        for i in range(1, 30):
            candidate = d - timedelta(days=i)
            if self.is_trading_day(candidate, exchange):
                return candidate
        msg = f"No trading day found within 30 days before {d}"
        raise ValueError(msg)
