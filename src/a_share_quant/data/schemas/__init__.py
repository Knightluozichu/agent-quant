"""Data schemas for the quant system.

Defines the internal data format that all providers must map to.
This ensures strategy code never directly depends on third-party SDKs.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field


class SecurityInfo(BaseModel):
    """Basic security information."""

    symbol: str = Field(description="Symbol like '600519.SSE'")
    code: str = Field(description="6-digit code")
    exchange: str = Field(description="Exchange: SSE, SZSE, BSE")
    name: str = Field(description="Security name")
    security_type: str = Field(description="stock, etf, index, etc.")
    board: str = Field(default="main", description="Trading board")
    list_date: Optional[date] = Field(default=None, description="IPO date")
    delist_date: Optional[date] = Field(default=None, description="Delisting date")
    is_st: bool = Field(default=False, description="Is ST stock")


class DailyBar(BaseModel):
    """Daily OHLCV bar."""

    symbol: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float  # 成交量（股/份）
    amount: float  # 成交额（元）
    # Optional fields
    pre_close: Optional[float] = None  # 前收盘价
    limit_up: Optional[float] = None  # 涨停价
    limit_down: Optional[float] = None  # 跌停价
    is_suspended: bool = False  # 是否停牌
    turnover_rate: Optional[float] = None  # 换手率


class AdjustmentFactor(BaseModel):
    """Price adjustment factor for dividends/splits."""

    symbol: str
    trade_date: date
    factor: float  # 复权因子
    adjustment_type: str = "qfq"  # qfq=前复权, hfq=后复权


class TradingCalendar(BaseModel):
    """Trading calendar entry."""

    exchange: str
    trade_date: date
    is_open: bool


class SuspensionInfo(BaseModel):
    """Suspension information."""

    symbol: str
    suspend_date: date
    resume_date: Optional[date] = None
    reason: Optional[str] = None


class IndexInfo(BaseModel):
    """Index information."""

    symbol: str
    name: str
    exchange: str
    base_date: date
    base_point: float


class IndustryInfo(BaseModel):
    """Industry classification."""

    symbol: str
    industry_code: str
    industry_name: str
    level: int = 1  # 行业分级


# DataFrame column specifications
DAILY_BAR_COLUMNS = [
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pre_close",
    "limit_up",
    "limit_down",
    "is_suspended",
    "turnover_rate",
]

SECURITY_INFO_COLUMNS = [
    "symbol",
    "code",
    "exchange",
    "name",
    "security_type",
    "board",
    "list_date",
    "delist_date",
    "is_st",
]


def validate_daily_bar_df(df: pd.DataFrame) -> list[str]:
    """Validate a daily bar DataFrame.

    Returns list of validation errors (empty if valid).
    """
    errors = []

    # Check required columns
    required = ["symbol", "trade_date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
        return errors  # Return early if required columns are missing

    if len(df) == 0:
        return errors

    # Check price validity
    if (df["high"] < df["low"]).any():
        errors.append("High price < Low price detected")

    if (df["close"] <= 0).any():
        errors.append("Non-positive close price detected")

    if (df["volume"] < 0).any():
        errors.append("Negative volume detected")

    # Check for duplicates
    dupes = df.duplicated(subset=["symbol", "trade_date"], keep=False)
    if dupes.any():
        errors.append(f"Duplicate rows detected: {dupes.sum()}")

    return errors
