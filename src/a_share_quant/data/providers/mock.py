"""Mock data provider for development and testing.

Generates synthetic data that follows A-share market rules.
This enables full development and testing without external data sources.
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from a_share_quant.data.providers.base import BaseDataProvider
from a_share_quant.data.schemas import SecurityInfo


class MockProvider(BaseDataProvider):
    """Mock data provider generating synthetic A-share data.

    Features:
    - Deterministic data generation (same seed = same data)
    - Realistic price movements with trends and volatility
    - Proper trading calendar (weekends excluded)
    - Support for multiple securities
    - Configurable market regimes
    """

    def __init__(self, seed: int = 20260720) -> None:
        super().__init__()
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Mock securities
        self._securities: dict[str, SecurityInfo] = {
            "510300.SSE": SecurityInfo(
                symbol="510300.SSE",
                code="510300",
                exchange="SSE",
                name="沪深300ETF",
                security_type="etf",
                board="etf",
                list_date=date(2012, 5, 28),
            ),
            "510500.SSE": SecurityInfo(
                symbol="510500.SSE",
                code="510500",
                exchange="SSE",
                name="中证500ETF",
                security_type="etf",
                board="etf",
                list_date=date(2013, 3, 15),
            ),
            "159915.SZSE": SecurityInfo(
                symbol="159915.SZSE",
                code="159915",
                exchange="SZSE",
                name="创业板ETF",
                security_type="etf",
                board="etf",
                list_date=date(2011, 9, 20),
            ),
            "600519.SSE": SecurityInfo(
                symbol="600519.SSE",
                code="600519",
                exchange="SSE",
                name="贵州茅台",
                security_type="stock",
                board="main",
                list_date=date(2001, 8, 27),
            ),
            "000001.SZSE": SecurityInfo(
                symbol="000001.SZSE",
                code="000001",
                exchange="SZSE",
                name="平安银行",
                security_type="stock",
                board="main",
                list_date=date(1991, 4, 3),
            ),
        }

        # Generate trading calendar
        self._trading_days = self._generate_trading_calendar(
            date(2010, 1, 1), date(2026, 12, 31)
        )

    @property
    def name(self) -> str:
        return "mock"

    def _generate_trading_calendar(
        self, start: date, end: date
    ) -> list[date]:
        """Generate trading calendar excluding weekends."""
        days = []
        current = start
        while current <= end:
            # Exclude weekends (5=Saturday, 6=Sunday)
            if current.weekday() < 5:
                days.append(current)
            current += timedelta(days=1)
        return days

    def _get_price_seed(self, symbol: str, trade_date: date) -> int:
        """Generate deterministic seed for a symbol/date combination."""
        key = f"{symbol}_{trade_date.isoformat()}_{self._seed}"
        return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)  # noqa: S324

    def _generate_price_series(
        self,
        symbol: str,
        trading_days: list[date],
        initial_price: float = 100.0,
        annual_return: float = 0.08,
        annual_vol: float = 0.25,
    ) -> pd.DataFrame:
        """Generate synthetic price series."""
        n = len(trading_days)
        if n == 0:
            return pd.DataFrame()

        # Daily parameters
        daily_return = annual_return / 252
        daily_vol = annual_vol / np.sqrt(252)

        # Generate returns with some autocorrelation for realism
        rng = np.random.default_rng(self._get_price_seed(symbol, trading_days[0]))
        returns = rng.normal(daily_return, daily_vol, n)

        # Add some momentum (autocorrelation)
        for i in range(1, n):
            returns[i] += 0.1 * returns[i - 1]

        # Generate prices
        prices = initial_price * np.cumprod(1 + returns)

        # Generate OHLCV
        data = []
        prev_close = initial_price
        for i, (day, close) in enumerate(zip(trading_days, prices)):
            # Intraday range
            intraday_vol = abs(returns[i]) + daily_vol * 0.5
            high = close * (1 + rng.uniform(0, intraday_vol))
            low = close * (1 - rng.uniform(0, intraday_vol))
            open_price = prev_close * (1 + rng.normal(0, daily_vol * 0.3))

            # Ensure OHLC consistency
            high = max(high, open_price, close)
            low = min(low, open_price, close)

            # Volume and amount
            base_volume = 1_000_000 if "ETF" in self._securities.get(symbol, SecurityInfo(
                symbol=symbol, code="", exchange="", name="", security_type="stock"
            )).name else 500_000
            volume = base_volume * (1 + abs(returns[i]) * 10) * rng.uniform(0.5, 1.5)
            amount = volume * close

            # Price limits (10% for main board)
            limit_up = round(prev_close * 1.10, 2)
            limit_down = round(prev_close * 0.90, 2)

            data.append({
                "symbol": symbol,
                "trade_date": day,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": int(volume),
                "amount": round(amount, 2),
                "pre_close": round(prev_close, 2),
                "limit_up": limit_up,
                "limit_down": limit_down,
                "is_suspended": False,
                "turnover_rate": round(rng.uniform(0.5, 3.0), 2),
            })

            prev_close = close

        return pd.DataFrame(data)

    def get_trading_calendar(
        self,
        exchange: str = "SSE",
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> pd.DataFrame:
        """Get trading calendar."""
        start = self._normalize_date(start_date) if start_date else self._trading_days[0]
        end = self._normalize_date(end_date) if end_date else self._trading_days[-1]

        days = [d for d in self._trading_days if start <= d <= end]
        return pd.DataFrame({
            "trade_date": days,
            "is_open": [True] * len(days),
        })

    def get_security_list(
        self,
        security_type: str | None = None,
        exchange: str | None = None,
        include_delisted: bool = False,
    ) -> pd.DataFrame:
        """Get list of securities."""
        securities = list(self._securities.values())

        if security_type:
            securities = [s for s in securities if s.security_type == security_type]
        if exchange:
            securities = [s for s in securities if s.exchange == exchange]

        return pd.DataFrame([s.model_dump() for s in securities])

    def get_security_info(self, symbol: str) -> Optional[SecurityInfo]:
        """Get info for a specific security."""
        return self._securities.get(symbol)

    def get_daily_bars(
        self,
        symbols: str | list[str],
        start_date: date | str,
        end_date: date | str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """Get daily OHLCV bars."""
        start, end = self._validate_date_range(start_date, end_date)
        symbol_list = self._normalize_symbols(symbols)

        trading_days = [d for d in self._trading_days if start <= d <= end]

        dfs = []
        for symbol in symbol_list:
            if symbol not in self._securities:
                continue

            # Different initial prices for different securities
            initial_prices = {
                "510300.SSE": 4.0,
                "510500.SSE": 6.0,
                "159915.SZSE": 2.5,
                "600519.SSE": 1800.0,
                "000001.SZSE": 15.0,
            }
            initial_price = initial_prices.get(symbol, 100.0)

            df = self._generate_price_series(symbol, trading_days, initial_price)
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def get_adjustment_factors(
        self,
        symbols: str | list[str],
        start_date: date | str,
        end_date: date | str,
    ) -> pd.DataFrame:
        """Get adjustment factors (mock: all 1.0)."""
        start, end = self._validate_date_range(start_date, end_date)
        symbol_list = self._normalize_symbols(symbols)
        trading_days = [d for d in self._trading_days if start <= d <= end]

        data = []
        for symbol in symbol_list:
            for day in trading_days:
                data.append({
                    "symbol": symbol,
                    "trade_date": day,
                    "factor": 1.0,
                })

        return pd.DataFrame(data)

    def get_suspension_info(
        self,
        symbols: str | list[str],
        start_date: date | str,
        end_date: date | str,
    ) -> pd.DataFrame:
        """Get suspension info (mock: no suspensions)."""
        return pd.DataFrame(columns=["symbol", "suspend_date", "resume_date", "reason"])

    def get_index_daily(
        self,
        symbols: str | list[str],
        start_date: date | str,
        end_date: date | str,
    ) -> pd.DataFrame:
        """Get index daily data."""
        # Treat indices like securities for mock
        return self.get_daily_bars(symbols, start_date, end_date)

    def get_industry_classification(
        self,
        symbols: str | list[str] | None = None,
        level: int = 1,
    ) -> pd.DataFrame:
        """Get industry classification."""
        if symbols is None:
            symbols = list(self._securities.keys())
        symbol_list = self._normalize_symbols(symbols)

        data = []
        for symbol in symbol_list:
            sec = self._securities.get(symbol)
            if sec:
                industry = "金融" if "银行" in sec.name else "消费" if "茅台" in sec.name else "宽基指数"
                data.append({
                    "symbol": symbol,
                    "industry_code": "MOCK01",
                    "industry_name": industry,
                    "level": level,
                })

        return pd.DataFrame(data)

    def is_trading_day(self, check_date: date | str, exchange: str = "SSE") -> bool:
        """Check if a date is a trading day."""
        d = self._normalize_date(check_date)
        return d in self._trading_days

    def get_next_trading_day(
        self,
        from_date: date | str,
        exchange: str = "SSE",
    ) -> date:
        """Get the next trading day."""
        d = self._normalize_date(from_date)
        for day in self._trading_days:
            if day > d:
                return day
        msg = f"No trading day found after {d}"
        raise ValueError(msg)

    def get_prev_trading_day(
        self,
        from_date: date | str,
        exchange: str = "SSE",
    ) -> date:
        """Get the previous trading day."""
        d = self._normalize_date(from_date)
        for day in reversed(self._trading_days):
            if day < d:
                return day
        msg = f"No trading day found before {d}"
        raise ValueError(msg)
