"""JoinQuant (JQData) data provider.

IMPORTANT: Quota management
- Trial: 50万条/天, 3个月有效期
- Cache aggressively to minimize API calls
- Batch requests when possible

Symbol format conversion:
- Our format: 510300.SSE, 000001.SZSE
- JQ format: 510300.XSHG, 000001.XSHE
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from a_share_quant.data.providers.base import BaseDataProvider
from a_share_quant.data.schemas import SecurityInfo
from a_share_quant.settings import get_settings

logger = logging.getLogger(__name__)

# 交易日相关时间一律使用 Asia/Shanghai, 不依赖运行环境本地时区
_SH_TZ: tzinfo
try:
    _SH_TZ = ZoneInfo("Asia/Shanghai")
except Exception:  # 极简环境缺 tzdata 时回退固定 +08:00 (1991 年后上海无夏令时)
    _SH_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")


def _today_sh() -> date:
    """返回 Asia/Shanghai 时区的当前日期."""
    return datetime.now(_SH_TZ).date()


# =============================================================================
# Symbol Conversion
# =============================================================================


def to_jq_symbol(symbol: str) -> str:
    """Convert our symbol format to JQ format.

    510300.SSE -> 510300.XSHG
    000001.SZSE -> 000001.XSHE
    """
    code, exchange = symbol.split(".")
    if exchange == "SSE":
        return f"{code}.XSHG"
    if exchange == "SZSE":
        return f"{code}.XSHE"
    if exchange == "BSE":
        return f"{code}.XBJE"
    return symbol


def from_jq_symbol(jq_symbol: str) -> str:
    """Convert JQ symbol format to our format.

    510300.XSHG -> 510300.SSE
    000001.XSHE -> 000001.SZSE
    """
    code, exchange = jq_symbol.split(".")
    if exchange == "XSHG":
        return f"{code}.SSE"
    if exchange == "XSHE":
        return f"{code}.SZSE"
    if exchange == "XBJE":
        return f"{code}.BSE"
    return jq_symbol


# =============================================================================
# Quota Tracker
# =============================================================================


class QuotaTracker:
    """Track daily API quota usage."""

    def __init__(self, daily_limit: int = 500_000):
        self.daily_limit = daily_limit
        self._used_today = 0
        self._last_reset = _today_sh()

    def _maybe_reset(self) -> None:
        today = _today_sh()
        if today != self._last_reset:
            self._used_today = 0
            self._last_reset = today

    def consume(self, count: int) -> bool:
        """Record API usage. Returns False if quota exceeded."""
        self._maybe_reset()
        if self._used_today + count > self.daily_limit:
            logger.warning(f"Quota limit approaching: {self._used_today}/{self.daily_limit}")
            return False
        self._used_today += count
        return True

    @property
    def remaining(self) -> int:
        self._maybe_reset()
        return max(0, self.daily_limit - self._used_today)

    @property
    def used(self) -> int:
        self._maybe_reset()
        return self._used_today


# =============================================================================
# JoinQuant Provider
# =============================================================================


class JoinQuantProvider(BaseDataProvider):
    """JoinQuant JQData provider.

    Features:
    - Automatic authentication
    - Quota tracking
    - Local caching to minimize API calls
    - Symbol format conversion
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        cache_dir: Path | str = "data/jq_cache",
        daily_quota: int = 500_000,
    ):
        super().__init__()
        self._username = username
        self._password = password
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._quota = QuotaTracker(daily_quota)
        self._authenticated = False
        self._jq = None

    @property
    def name(self) -> str:
        return "joinquant"

    def _ensure_auth(self) -> None:
        """Ensure authenticated with JQData."""
        if self._authenticated:
            return

        try:
            import jqdatasdk

            self._jq = jqdatasdk

            # Get credentials
            username = self._username
            password = self._password

            if not username or not password:
                settings = get_settings()
                username = settings.joinquant_username
                password = settings.joinquant_password

            if not username or not password:
                raise ValueError(
                    "JoinQuant credentials not configured. "
                    "Set JOINQUANT_USERNAME and JOINQUANT_PASSWORD in .env"
                )

            jqdatasdk.auth(username, password)
            self._authenticated = True
            logger.info("JQData authenticated successfully")

            # Log remaining quota
            try:
                quota = jqdatasdk.get_query_count()
                logger.info(f"JQData quota: {quota}")
            except Exception as e:
                # 配额查询失败不影响认证流程, 仅记录
                logger.debug(f"JQData quota query failed: {e}")

        except ImportError as e:
            raise ImportError("jqdatasdk not installed. Run: pip install jqdatasdk") from e

    def _get_cache_path(self, category: str, key: str) -> Path:
        """Get cache file path."""
        safe_key = key.replace("/", "_").replace(".", "_")
        return self._cache_dir / category / f"{safe_key}.parquet"

    def _read_cache(self, category: str, key: str) -> pd.DataFrame | None:
        """Read from cache if exists and fresh."""
        path = self._get_cache_path(category, key)
        if path.exists():
            # Cache valid for 1 day
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=_SH_TZ)
            if datetime.now(_SH_TZ) - mtime < timedelta(days=1):
                try:
                    return pd.read_parquet(path)
                except Exception as e:
                    # 缓存损坏时回退到 API 拉取, 仅记录
                    logger.debug(f"Cache read failed for {path}: {e}")
        return None

    def _write_cache(self, category: str, key: str, df: pd.DataFrame) -> None:
        """Write to cache."""
        path = self._get_cache_path(category, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(path, index=False)
        except Exception as e:
            logger.warning(f"Failed to cache: {e}")

    # -------------------------------------------------------------------------
    # Trading Calendar
    # -------------------------------------------------------------------------

    def get_trading_calendar(
        self,
        exchange: str = "SSE",
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> pd.DataFrame:
        """Get trading calendar."""
        self._ensure_auth()

        # Check cache
        cache_key = f"calendar_{exchange}"
        cached = self._read_cache("calendar", cache_key)
        if cached is not None:
            df = cached
        else:
            # API call
            start = start_date or "2005-01-01"
            end = end_date or (_today_sh() + timedelta(days=365)).isoformat()

            if not self._quota.consume(1000):
                raise RuntimeError("Daily quota exceeded")

            trade_days = self._jq.get_trade_days(start_date=start, end_date=end)
            df = pd.DataFrame(
                {
                    "trade_date": trade_days,
                    "exchange": exchange,
                    "is_open": True,
                }
            )
            self._write_cache("calendar", cache_key, df)

        # Filter
        if start_date:
            start = self._normalize_date(start_date)
            df = df[df["trade_date"] >= start]
        if end_date:
            end = self._normalize_date(end_date)
            df = df[df["trade_date"] <= end]

        return df.reset_index(drop=True)

    def is_trading_day(self, check_date: date | str, exchange: str = "SSE") -> bool:
        """Check if date is trading day."""
        d = self._normalize_date(check_date)
        cal = self.get_trading_calendar(exchange, d, d)
        return len(cal) > 0

    # -------------------------------------------------------------------------
    # Security List
    # -------------------------------------------------------------------------

    def get_security_list(
        self,
        security_type: str | None = None,
        exchange: str | None = None,
        include_delisted: bool = False,
    ) -> pd.DataFrame:
        """Get security list."""
        self._ensure_auth()

        cache_key = f"securities_{security_type or 'all'}"
        cached = self._read_cache("securities", cache_key)

        if cached is not None:
            df = cached
        else:
            if not self._quota.consume(10000):
                raise RuntimeError("Daily quota exceeded")

            # Get all securities
            jq_type = None
            if security_type == "stock":
                jq_type = "stock"
            elif security_type == "fund":
                jq_type = "fund"
            elif security_type == "index":
                jq_type = "index"

            df = self._jq.get_all_securities(types=jq_type)
            df = df.reset_index()
            df.columns = ["jq_symbol", "display_name", "name", "start_date", "end_date", "type"]

            # Convert symbols
            df["symbol"] = df["jq_symbol"].apply(from_jq_symbol)
            df["exchange"] = df["symbol"].apply(lambda x: x.split(".")[1])

            self._write_cache("securities", cache_key, df)

        # Filter
        if exchange:
            df = df[df["exchange"] == exchange]
        if not include_delisted and "end_date" in df.columns:
            today = _today_sh()
            df = df[df["end_date"].isna() | (df["end_date"] >= today)]

        return df.reset_index(drop=True)

    def get_security_info(self, symbol: str) -> SecurityInfo | None:
        """Get security info."""
        self._ensure_auth()

        jq_symbol = to_jq_symbol(symbol)

        if not self._quota.consume(1):
            raise RuntimeError("Daily quota exceeded")

        try:
            info = self._jq.get_security_info(jq_symbol)
            return SecurityInfo(
                symbol=symbol,
                display_name=info.display_name,
                name=info.abbrev_code,
                security_type=info.type,
                exchange=symbol.split(".")[1],
                list_date=info.start_date,
                delist_date=info.end_date,
            )
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Daily Bars
    # -------------------------------------------------------------------------

    def get_daily_bars(
        self,
        symbols: str | list[str],
        start_date: date | str,
        end_date: date | str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """Get daily bars."""
        self._ensure_auth()

        start, end = self._validate_date_range(start_date, end_date)
        symbol_list = self._normalize_symbols(symbols)

        # Check cache for each symbol
        dfs = []
        symbols_to_fetch = []

        for symbol in symbol_list:
            cache_key = f"daily_{symbol}_{start}_{end}_{adjust}"
            cached = self._read_cache("daily", cache_key)
            if cached is not None:
                dfs.append(cached)
            else:
                symbols_to_fetch.append(symbol)

        # Fetch missing symbols
        if symbols_to_fetch:
            jq_symbols = [to_jq_symbol(s) for s in symbols_to_fetch]

            # Estimate quota usage
            days = (end - start).days
            estimated_rows = len(jq_symbols) * days
            if not self._quota.consume(estimated_rows):
                raise RuntimeError(
                    f"Daily quota exceeded. Need ~{estimated_rows}, "
                    f"remaining {self._quota.remaining}"
                )

            # API call
            # JQ fq parameter: None, 'pre', 'post', 'none'
            fq_map = {"qfq": "pre", "hfq": "post", "none": None}
            jq_fq = fq_map.get(adjust, "pre")

            jq_df = self._jq.get_price(
                jq_symbols,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                frequency="daily",
                panel=False,
                fq=jq_fq,
            )

            if jq_df is not None and not jq_df.empty:
                # Convert symbol format
                jq_df["symbol"] = jq_df["code"].apply(from_jq_symbol)
                jq_df = jq_df.rename(columns={"time": "trade_date"})

                # Select columns
                cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume"]
                if "amount" in jq_df.columns:
                    cols.append("amount")
                jq_df = jq_df[[c for c in cols if c in jq_df.columns]]

                # Cache per symbol
                for symbol in symbols_to_fetch:
                    sym_df = jq_df[jq_df["symbol"] == symbol]
                    if not sym_df.empty:
                        cache_key = f"daily_{symbol}_{start}_{end}_{adjust}"
                        self._write_cache("daily", cache_key, sym_df)
                        dfs.append(sym_df)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        return result.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Index Data
    # -------------------------------------------------------------------------

    def get_index_daily(
        self,
        symbols: str | list[str],
        start_date: date | str,
        end_date: date | str,
    ) -> pd.DataFrame:
        """Get index daily data."""
        return self.get_daily_bars(symbols, start_date, end_date, adjust="none")

    def get_index_stocks(
        self,
        index_symbol: str,
        trade_date: date | str | None = None,
    ) -> list[str]:
        """Get index constituent stocks."""
        self._ensure_auth()

        jq_symbol = to_jq_symbol(index_symbol)
        d = trade_date or _today_sh()

        cache_key = f"index_stocks_{index_symbol}_{d}"
        cached = self._read_cache("index", cache_key)

        if cached is not None:
            return cached["symbol"].tolist()

        if not self._quota.consume(1000):
            raise RuntimeError("Daily quota exceeded")

        stocks = self._jq.get_index_stocks(jq_symbol, date=d)
        symbols = [from_jq_symbol(s) for s in stocks]

        df = pd.DataFrame({"symbol": symbols})
        self._write_cache("index", cache_key, df)

        return symbols

    # -------------------------------------------------------------------------
    # Quota Info
    # -------------------------------------------------------------------------

    def get_quota_info(self) -> dict:
        """Get quota usage info."""
        self._ensure_auth()

        info = {
            "tracked_used": self._quota.used,
            "tracked_remaining": self._quota.remaining,
        }

        try:
            jq_quota = self._jq.get_query_count()
            info["jq_total"] = jq_quota.get("total", 0)
            info["jq_used"] = jq_quota.get("used", 0)
            info["jq_remaining"] = jq_quota.get("spare", 0)
        except Exception as e:
            # 配额信息仅用于展示, 查询失败不阻塞
            logger.debug(f"JQData quota info query failed: {e}")

        return info
