"""Core domain types for the A-share quant system.

This module defines fundamental value types used throughout the system:
- TradingDate: A trading date in the A-share market
- Symbol: A stock/ETF symbol with exchange information
- Currency: Supported currencies
- Money: Amount with currency
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import Field, field_validator


class Exchange(StrEnum):
    """A-share exchanges."""

    SSE = "SSE"  # 上海证券交易所
    SZSE = "SZSE"  # 深圳证券交易所
    BSE = "BSE"  # 北京证券交易所


class SecurityType(StrEnum):
    """Security types."""

    STOCK = "stock"  # 股票
    ETF = "etf"  # 交易所交易基金
    INDEX = "index"  # 指数
    BOND = "bond"  # 债券
    FUND = "fund"  # 基金


class Board(StrEnum):
    """Trading boards with different rules."""

    MAIN = "main"  # 主板
    GEM = "gem"  # 创业板 (ChiNext)
    STAR = "star"  # 科创板
    BSE_BOARD = "bse"  # 北交所
    ETF_BOARD = "etf"  # ETF


class Currency(StrEnum):
    """Supported currencies."""

    CNY = "CNY"  # 人民币
    HKD = "HKD"  # 港币
    USD = "USD"  # 美元


class TradingDate:
    """A trading date in the A-share market.

    Wraps a date to ensure it represents a valid trading day.
    Validation against actual trading calendar is done at data layer.
    """

    __slots__ = ("_date",)

    def __init__(self, value: date | str) -> None:
        if isinstance(value, str):
            # Support formats: 20260720, 2026-07-20
            clean = value.replace("-", "")
            if len(clean) != 8:
                msg = f"Invalid date format: {value}"
                raise ValueError(msg)
            self._date = date(int(clean[:4]), int(clean[4:6]), int(clean[6:8]))
        else:
            self._date = value

    @property
    def date(self) -> date:
        """Return the underlying date."""
        return self._date

    def to_str(self, fmt: str = "%Y%m%d") -> str:
        """Format as string."""
        return self._date.strftime(fmt)

    def __repr__(self) -> str:
        return f"TradingDate({self._date.isoformat()})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TradingDate):
            return self._date == other._date
        if isinstance(other, date):
            return self._date == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._date)

    def __lt__(self, other: TradingDate) -> bool:
        return self._date < other._date

    def __le__(self, other: TradingDate) -> bool:
        return self._date <= other._date

    def __gt__(self, other: TradingDate) -> bool:
        return self._date > other._date

    def __ge__(self, other: TradingDate) -> bool:
        return self._date >= other._date


class Symbol:
    """A tradeable symbol in the A-share market.

    Examples:
        - 600519.SSE (贵州茅台)
        - 000001.SZSE (平安银行)
        - 510300.SSE (沪深300ETF)
    """

    __slots__ = ("_code", "_exchange", "_security_type")

    def __init__(
        self,
        code: str,
        exchange: Exchange | str,
        security_type: SecurityType | str = SecurityType.STOCK,
    ) -> None:
        self._code = code
        self._exchange = Exchange(exchange) if isinstance(exchange, str) else exchange
        self._security_type = (
            SecurityType(security_type) if isinstance(security_type, str) else security_type
        )

    @classmethod
    def from_string(cls, symbol_str: str) -> Symbol:
        """Parse from string like '600519.SSE' or '000001.SZSE'."""
        parts = symbol_str.split(".")
        if len(parts) != 2:
            msg = f"Invalid symbol format: {symbol_str}. Expected 'CODE.EXCHANGE'"
            raise ValueError(msg)
        code, exchange = parts
        return cls(code=code, exchange=exchange)

    @property
    def code(self) -> str:
        """Return the symbol code."""
        return self._code

    @property
    def exchange(self) -> Exchange:
        """Return the exchange."""
        return self._exchange

    @property
    def security_type(self) -> SecurityType:
        """Return the security type."""
        return self._security_type

    @property
    def board(self) -> Board:
        """Infer the trading board from the symbol code."""
        if self._security_type == SecurityType.ETF:
            return Board.ETF_BOARD
        if self._exchange == Exchange.BSE:
            return Board.BSE_BOARD
        if self._code.startswith("688"):
            return Board.STAR
        if self._code.startswith("300") or self._code.startswith("301"):
            return Board.GEM
        return Board.MAIN

    def to_string(self) -> str:
        """Return string representation like '600519.SSE'."""
        return f"{self._code}.{self._exchange.value}"

    def __repr__(self) -> str:
        return f"Symbol({self.to_string()})"

    def __str__(self) -> str:
        return self.to_string()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Symbol):
            return self._code == other._code and self._exchange == other._exchange
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._code, self._exchange))


class Money:
    """Amount with currency.

    Uses Decimal for precise financial calculations.
    """

    __slots__ = ("_amount", "_currency")

    def __init__(
        self, amount: float | int | Decimal | str, currency: Currency | str = Currency.CNY
    ) -> None:
        self._amount = Decimal(str(amount))
        self._currency = Currency(currency) if isinstance(currency, str) else currency

    @property
    def amount(self) -> Decimal:
        """Return the amount."""
        return self._amount

    @property
    def currency(self) -> Currency:
        """Return the currency."""
        return self._currency

    def __add__(self, other: Money) -> Money:
        if self._currency != other._currency:
            msg = f"Cannot add {self._currency} and {other._currency}"
            raise ValueError(msg)
        return Money(self._amount + other._amount, self._currency)

    def __sub__(self, other: Money) -> Money:
        if self._currency != other._currency:
            msg = f"Cannot subtract {self._currency} and {other._currency}"
            raise ValueError(msg)
        return Money(self._amount - other._amount, self._currency)

    def __mul__(self, factor: float | int | Decimal) -> Money:
        return Money(self._amount * Decimal(str(factor)), self._currency)

    def __repr__(self) -> str:
        return f"Money({self._amount}, {self._currency.value})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Money):
            return self._amount == other._amount and self._currency == other._currency
        return NotImplemented


# Type aliases for common patterns
SymbolStr = Annotated[str, Field(pattern=r"^\d{6}\.(SSE|SZSE|BSE)$")]
DateStr = Annotated[str, Field(pattern=r"^\d{8}$")]
