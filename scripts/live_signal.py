"""七星 V3-G/V4 实盘信号生成器 (生产级).

核心保证: 直接 import run_qixing_v3 的 select_target, 实盘与回测逻辑100%一致.

用法:
  uv run python scripts/live_signal.py --init 100000   # 首次: 初始化账户(10万本金)
  uv run python scripts/live_signal.py                 # 每日: 更新数据+生成信号
  uv run python scripts/live_signal.py --status        # 查看当前持仓(不更新数据)
  uv run python scripts/live_signal.py --dry-run       # 预演今日信号(不改动状态)
  uv run python scripts/live_signal.py --sync-only     # 仅更新行情(夜间补齐当日K线, 不出信号)
  uv run python scripts/live_signal.py --set-bark      # 设置Bark推送(交互式输入)
     或: BARK_KEY=xxx uv run python scripts/live_signal.py --set-bark

推荐工作流:
  每个交易日 14:50 后运行 → 若为调仓日且有换仓信号 → 15:00 收盘前执行
  (或次日开盘执行, 周频策略隔日影响很小)

调仓频率: 每5个交易日检查一次 (与回测一致).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import stat
import sys
import time
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# --------------------------------------------------------------------------- #
# 时区辅助: 显式 Asia/Shanghai, 不依赖服务器本地时区 (避免迁移失效)
# --------------------------------------------------------------------------- #
_SH_TZ: tzinfo
try:
    _SH_TZ = ZoneInfo("Asia/Shanghai")
except Exception:  # 极简环境缺 tzdata 时回退固定 +08:00 (1991 年后上海无夏令时)
    _SH_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")


def _today_sh() -> date:
    """返回 Asia/Shanghai 时区的当前日期 (用于全部实盘日期判断)."""
    return datetime.now(_SH_TZ).date()


# === 保证与回测100%一致: 直接复用回测的核心逻辑 ===
sys.path.insert(0, str(Path(__file__).parent))
import qixing_v4 as v4  # noqa: E402
from notify import load_config, push_bark, save_config, set_bark_key, test_push  # noqa: E402
from risk_overrides import ACTION_EMERGENCY, RiskDecision  # noqa: E402
from risk_overrides import assess as risk_assess  # noqa: E402
from run_qixing_v3 import (  # noqa: E402
    A_SHARE_MA,
    DATA_DIR,
    DEFENSE,
    DROP_LOOKBACK,
    DROP_THRESHOLD,
    ETF_POOL,
    FEE,
    MOM_PERIODS,
    MOM_WEIGHTS,
    REBALANCE_DAYS,
    SLIPPAGE,
    calc_momentum_score,
    load_data,
    select_target,
)

LIVE_DIR = DATA_DIR.parent / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = LIVE_DIR / "state.json"
STATE_TMP_FILE = STATE_FILE.parent / (STATE_FILE.name + ".tmp")
LOCK_FILE = LIVE_DIR / "quant_state.lock"
STRATEGY_MODE_FILE = LIVE_DIR / "strategy_mode.json"
SNAPSHOT_DIR = LIVE_DIR / "decision_snapshots"
MAX_BACKUPS = 7

ALL_CODES = [*list(ETF_POOL.keys()), DEFENSE]


def name_of(code: str) -> str:
    return ETF_POOL.get(code, "货币基金")


def sina_symbol(code: str) -> str:
    prefix = "sh" if code.startswith(("5", "6")) else "sz"
    return f"{prefix}{code}"


# --------------------------------------------------------------------------- #
# 数据更新 (增量追加, 不碰已复权的历史数据 → 保证实盘==回测)
# --------------------------------------------------------------------------- #
def update_data(data: dict) -> dict:
    """增量拉取最新行情, 追加到已复权的历史缓存.

    历史数据保持不变(与回测一致), 只追加新交易日.
    若检测到疑似份额拆分(单日跳变>40%)会告警.
    """
    import akshare as ak

    today = _today_sh()
    updated = []

    for code in ALL_CODES:
        if code not in data:
            continue
        df = data[code]
        last_date = df["trade_date"].max()
        if last_date >= today:
            continue  # 已是最新

        try:
            raw = ak.fund_etf_hist_sina(symbol=sina_symbol(code))
            if raw is None or raw.empty:
                continue
            raw = raw.rename(columns={"date": "trade_date"})
            raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.date
            new = raw[raw["trade_date"] > last_date].copy()
            if new.empty:
                continue

            # 份额拆分安全检查: 新数据首日 vs 缓存末日 收盘价跳变>40%
            last_close = float(df.iloc[-1]["close"])
            first_new_close = float(new.iloc[0]["close"])
            jump = (first_new_close - last_close) / last_close
            if abs(jump) > 0.40:
                print(
                    f"  ⚠️  {code} {name_of(code)} 检测到异常跳变 {jump:+.0%}, "
                    f"疑似份额拆分! 请人工复权后再用 (切勿直接交易)"
                )
                continue

            new["symbol"] = code
            for col in ["open", "close", "high", "low", "volume"]:
                if col not in new.columns:
                    new[col] = 0.0
            new = new[["trade_date", "open", "close", "high", "low", "volume", "symbol"]]

            df = pd.concat([df, new], ignore_index=True)
            df = df.sort_values("trade_date").drop_duplicates("trade_date").reset_index(drop=True)
            df.to_parquet(DATA_DIR / f"{code}.parquet", index=False)
            data[code] = df
            updated.append(f"{code}+{len(new)}天")
            time.sleep(1.0)
        except Exception as e:
            print(f"  ⚠️  {code} 更新失败: {e}")

    if updated:
        print(f"  数据更新: {', '.join(updated)}")
    else:
        _td = get_trading_dates(data)
        print(f"  数据已是最新 (最新数据: {_td[-1] if _td else '无'})")
    check_data_freshness(data)
    return data


def check_data_freshness(data: dict) -> None:
    """数据新鲜度检查: 若缺失上一个已完成交易日的数据则告警.

    新浪等数据源的当日K线有发布延迟。若上一交易日数据缺失,
    信号将基于过期数据, 此处打印告警(不阻断)供日志排查。
    """
    try:
        import akshare as ak

        cal = ak.tool_trade_date_hist_sina()
        cal_dates = sorted(pd.to_datetime(cal["trade_date"]).dt.date)
        today = _today_sh()
        past = [d for d in cal_dates if d < today]
        if not past:
            return
        expected = past[-1]  # 上一个已完成交易日
        trading_dates = get_trading_dates(data)
        if not trading_dates:
            return
        latest = trading_dates[-1]
        if latest < expected:
            print(f"  ⚠️  数据滞后: 最新数据 {latest}, 缺失上一交易日 {expected} 的数据!")
            print("      信号将基于过期数据, 请检查数据源 (可能为发布延迟)")
    except Exception as e:
        print(f"  (数据新鲜度检查跳过: {e})")


# 已知份额拆分 (前复权: 拆分前 OHLC 除以比例, 与回测缓存一致)
KNOWN_SPLITS = {
    "513100": [("2022-01-14", 5.1153)],  # 纳指ETF 1:5.1153
    "511220": [("2023-01-16", 10.0663)],  # 城投债ETF 1:10.0663
}


def bootstrap_data() -> None:
    """全量拉取历史并前复权 (服务器首次部署 / 灾难恢复用).

    从新浪源拉取全部历史, 对已知拆分做前复权, 重建缓存.
    注意: 只应在空缓存时运行; 已有缓存请用增量 update_data.
    """
    import akshare as ak

    print("  全量数据初始化 (拉取完整历史 + 份额拆分复权)...")
    for code in ALL_CODES:
        print(f"  拉取 {code} {name_of(code)}...", end=" ", flush=True)
        try:
            raw = ak.fund_etf_hist_sina(symbol=sina_symbol(code))
            if raw is None or raw.empty:
                print("无数据")
                continue
            df = raw.rename(columns={"date": "trade_date"})
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            df["symbol"] = code
            for col in ["open", "close", "high", "low", "volume"]:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[["trade_date", "open", "close", "high", "low", "volume", "symbol"]]
            df = df.sort_values("trade_date").reset_index(drop=True)

            # 数值列统一转 float64 (避免拆分复权后与 int64 schema 冲突)
            for col in ["open", "close", "high", "low", "volume"]:
                df[col] = df[col].astype("float64")

            # 份额拆分前复权
            for split_date, ratio in KNOWN_SPLITS.get(code, []):
                split_d = pd.to_datetime(split_date).date()
                mask = df["trade_date"] < split_d
                for col in ["open", "close", "high", "low"]:
                    df.loc[mask, col] = df.loc[mask, col].astype(float) / ratio
                df.loc[mask, "volume"] = df.loc[mask, "volume"].astype(float) * ratio

            df.to_parquet(DATA_DIR / f"{code}.parquet", index=False)
            print(f"{len(df)}天 ({df['trade_date'].min()} ~ {df['trade_date'].max()})")
            time.sleep(1.0)
        except Exception as e:
            print(f"失败: {e}")
    print("  ✓ 数据初始化完成")


def _append_risk_log(state: dict, events: list) -> None:
    """追加风控事件到 risk_log, 保留最近 500 条防膨胀 (R3 二期)."""
    log = state.setdefault("risk_log", [])
    log.extend(events)
    if len(log) > 500:
        del log[:-500]


# --------------------------------------------------------------------------- #
# 状态管理
# --------------------------------------------------------------------------- #
def default_state(capital: float) -> dict:
    return {
        "initial_capital": capital,
        "cash": capital,
        "holding": None,
        "shares": 0,
        "entry_date": None,
        "entry_price": 0.0,
        "last_rebalance_date": None,
        "last_run_date": None,
        "pending_order": None,
        "trade_log": [],
        # V32 尾部风控状态 (与回测 risk_overrides 同构)
        "peak_equity": capital,
        "risk_exposure": 1.0,
        "cooldown_until": None,
        "risk_log": [],
        # V3-G H3 放行止损状态
        "h3_holding": None,
        "h3_peak": 0.0,
        "v4_state": default_v4_state(),
        "last_decision": None,
    }


def default_v4_state() -> dict:
    return {
        "schema_version": v4.STATE_SCHEMA_VERSION,
        "strategy_id": v4.STRATEGY_ID,
        "config_hash": v4.CONFIG_HASH,
        "candidate_history": [],
        "last_early_rotation_date": None,
    }


def _normalize_state(state: dict) -> dict:
    state.setdefault("_version", 0)
    state.setdefault("peak_equity", state.get("initial_capital", 0.0))
    state.setdefault("risk_exposure", 1.0)
    state.setdefault("cooldown_until", None)
    state.setdefault("risk_log", [])
    state.setdefault("h3_holding", None)
    state.setdefault("h3_peak", 0.0)
    state.setdefault("last_decision", None)
    runtime = state.setdefault("v4_state", default_v4_state())
    runtime.setdefault("schema_version", v4.STATE_SCHEMA_VERSION)
    runtime.setdefault("strategy_id", v4.STRATEGY_ID)
    runtime.setdefault("config_hash", v4.CONFIG_HASH)
    runtime.setdefault("candidate_history", [])
    runtime.setdefault("last_early_rotation_date", None)
    return state


def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    with open(STATE_FILE) as f:
        state = json.load(f)
    return _normalize_state(state)


def get_strategy_mode() -> str:
    """Return active strategy mode; environment override is for dry-run canaries."""
    mode = os.environ.get("QIXING_STRATEGY_MODE", "").strip().upper()
    if not mode and STRATEGY_MODE_FILE.exists():
        with open(STRATEGY_MODE_FILE) as f:
            mode = str(json.load(f).get("mode", "")).strip().upper()
    if not mode:
        mode = "V3-G"
    if mode not in {"V3-G", "V4_SHADOW", "V4"}:
        raise ValueError(f"未知策略模式: {mode}")
    return mode


def set_strategy_mode(mode: str) -> None:
    mode = mode.strip().upper()
    if mode not in {"V3-G", "V4_SHADOW", "V4"}:
        raise ValueError("策略模式必须为 V3-G、V4_SHADOW 或 V4")
    STRATEGY_MODE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = STRATEGY_MODE_FILE.with_suffix(".tmp")
    with open(temp, "w") as f:
        json.dump({"mode": mode}, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(STRATEGY_MODE_FILE)
    _fsync_dir(STRATEGY_MODE_FILE.parent)


def _fsync_dir(path: Path) -> None:
    """fsync 目录, 确保目录条目变更落盘 (macOS/Linux). 不支持则忽略."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _rotate_backups() -> None:
    """轮转备份: state.json.bak(最新) → .bak.1 → ... → .bak.6(最旧), 最多保留 MAX_BACKUPS 个."""
    if not STATE_FILE.exists():
        return
    bak = STATE_FILE.parent / f"{STATE_FILE.name}.bak"
    # 丢弃最旧备份 (.bak.6)
    oldest = STATE_FILE.parent / f"{STATE_FILE.name}.bak.{MAX_BACKUPS - 1}"
    if oldest.exists():
        oldest.unlink()
    # 依次后移: .bak.(N-1) → .bak.N
    for i in range(MAX_BACKUPS - 2, -1, -1):
        src = bak if i == 0 else STATE_FILE.parent / f"{STATE_FILE.name}.bak.{i}"
        dst = STATE_FILE.parent / f"{STATE_FILE.name}.bak.{i + 1}"
        if src.exists():
            src.replace(dst)
    # 当前 state.json 复制为最新备份
    shutil.copy2(STATE_FILE, bak)


def _state_file_metadata() -> tuple[int, int, int] | None:
    """读取当前 state.json 元数据, 供 root cron 保留 quant 服务权限."""
    try:
        info = STATE_FILE.stat()
    except OSError:
        return None
    return info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)


def _restore_state_file_metadata(
    path: Path,
    metadata: tuple[int, int, int] | None,
) -> None:
    """将临时状态文件恢复为原文件属主/权限, 避免原子替换后权限漂移."""
    if metadata is None:
        return
    uid, gid, mode = metadata
    try:
        # root cron 需要把文件交还给 quant; 非 root 进程无需 chown.
        if os.geteuid() == 0:
            os.chown(path, uid, gid)
    except OSError:
        pass
    with contextlib.suppress(OSError):
        path.chmod(mode)


def save_state_atomic(state: dict) -> None:
    """原子写状态文件 (事务锁 + 临时文件 + fsync + 原子替换 + 自动备份).

    流程:
      1. fcntl.flock 锁定 data/live/quant_state.lock (跨进程互斥, 不受 PrivateTmp 隔离)
      2. 轮转备份 (保留最多 7 个)
      3. 写临时文件 state.json.tmp → flush + fsync
      4. os.replace 原子替换 state.json
      5. fsync 父目录 (目录条目落盘)
      6. _version 递增
    """
    state["_version"] = int(state.get("_version", 0)) + 1
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        metadata = _state_file_metadata()
        _rotate_backups()
        with open(STATE_TMP_FILE, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        _restore_state_file_metadata(STATE_TMP_FILE, metadata)
        STATE_TMP_FILE.replace(STATE_FILE)
        _fsync_dir(STATE_FILE.parent)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


@contextlib.contextmanager
def state_transaction():
    """完整事务: 加锁 → 读取 → 校验版本 → (调用方修改) → 原子写入 → 解锁.

    用法:
        with state_transaction() as state:
            state["cash"] -= 100
            # 退出 with 块时自动原子写入
    """
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # 在锁内读取最新状态
        if not STATE_FILE.exists():
            raise ValueError("账户未初始化")
        with open(STATE_FILE) as f:
            state = _normalize_state(json.load(f))
        metadata = _state_file_metadata()
        state.setdefault("_version", 0)
        old_version = state["_version"]

        yield state

        # 校验版本未被其他进程修改
        if state["_version"] != old_version:
            raise RuntimeError(f"状态版本冲突: 期望 {old_version}, 实际 {state['_version']}")
        # 原子写入
        state["_version"] = old_version + 1
        _rotate_backups()
        with open(STATE_TMP_FILE, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        _restore_state_file_metadata(STATE_TMP_FILE, metadata)
        STATE_TMP_FILE.replace(STATE_FILE)
        _fsync_dir(STATE_FILE.parent)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def save_state(state: dict) -> None:
    """保存状态 (原子写, 委托 save_state_atomic)."""
    save_state_atomic(state)


def is_paper_mode() -> bool:
    """模拟记账模式: 信号按理论价自动记账 (用于尚无法真实下单的阶段)."""
    return bool(load_config().get("paper_mode", False))


def set_paper_mode(on: bool) -> None:
    cfg = load_config()
    cfg["paper_mode"] = bool(on)
    save_config(cfg)


# --------------------------------------------------------------------------- #
# 交易日历
# --------------------------------------------------------------------------- #
def get_trading_dates(data: dict) -> list:
    """公共交易日历 (与回测一致)."""
    common: set = set()
    for code in ALL_CODES:
        if code not in data:
            continue
        dates = set(data[code]["trade_date"].tolist())
        common = dates if not common else common & dates
    return sorted(common)


def price_on(data: dict, code: str, td) -> float | None:
    if code not in data:
        return None
    row = data[code][data[code]["trade_date"] == td]
    if row.empty:
        return None
    return float(row.iloc[0]["close"])


_REALTIME_CACHE: dict[str, object] = {"time": 0.0, "spot": {}}


def _fetch_tencent_spot() -> dict[str, dict]:
    """腾讯实时行情 (qt.gtimg.cn), 覆盖全部场内ETF+LOF+QDII, 缓存1分钟。

    东方财富 fund_etf_spot_em 不覆盖 LOF (如501018南方原油),
    腾讯源无此限制。返回 {code: {"price": float, "prev_close": float}}。
    """
    import requests

    now = time.time()
    cached_spot: dict[str, dict] = _REALTIME_CACHE["spot"]  # type: ignore[assignment]
    if cached_spot and (now - _REALTIME_CACHE["time"]) <= 60:  # type: ignore[operator]
        return cached_spot

    tencent_codes = [f"{'sh' if c.startswith(('5', '6')) else 'sz'}{c}" for c in ALL_CODES]
    url = f"https://qt.gtimg.cn/q={','.join(tencent_codes)}"
    try:
        resp = requests.get(url, timeout=10)
        resp.encoding = "gbk"
    except Exception as e:
        print(f"  ⚠️  腾讯行情获取失败: {e}")
        return cached_spot

    spot: dict[str, dict] = {}
    for line in resp.text.strip().split(";"):
        if "~" not in line:
            continue
        parts = line.split("~")
        if len(parts) < 5:
            continue
        code = parts[2]
        price = float(parts[3]) if parts[3] else 0
        prev_close = float(parts[4]) if parts[4] else 0
        exchange_time = parts[30] if len(parts) > 30 else ""
        try:
            # 显式使用 Asia/Shanghai 时区解析, 避免服务器迁移时失效
            dt = datetime.strptime(exchange_time, "%Y%m%d%H%M%S").replace(tzinfo=_SH_TZ)
            exchange_ts = dt.timestamp()
        except (TypeError, ValueError):
            exchange_ts = 0.0
        if price > 0:
            spot[code] = {
                "price": price,
                "prev_close": prev_close,
                "ts": exchange_ts,
                "exchange_time": exchange_time,
                "fetched_at": now,
            }

    if spot:
        _REALTIME_CACHE["spot"] = spot
        _REALTIME_CACHE["time"] = now
    return spot


def get_realtime_price(code: str) -> float | None:
    """获取ETF/LOF当日实时价 (腾讯源, 覆盖全部品种含LOF, 缓存1分钟)。

    用于实时急跌保护: 盘中急跌>3%时排除候选。
    失败返回None(回退到历史价)。
    """
    info = _fetch_tencent_spot().get(code)
    return info["price"] if info else None


def account_value(state: dict, data: dict, td) -> float:
    """账户总值 = 现金 + 持仓市值.

    停牌回退: td 当日无价时, 用 td 当天或之前的最后已知收盘价估值
    (仅服务估值/推送, 不用于成交; 严格按 td 截断, 绝不使用未来数据).
    """
    total = state["cash"]
    if state["holding"]:
        p = price_on(data, state["holding"], td)
        if not p:
            # 停牌回退: td 当天或之前的最后已知收盘价
            df = data.get(state["holding"])
            if df is not None:
                hist = df[df["trade_date"] <= td]
                if not hist.empty:
                    p = float(hist["close"].iloc[-1])
        if p:
            total += state["shares"] * p
    return total


def build_etf_data_at_date(data: dict, td, warmup: int = 130) -> dict:
    """构造 select_target 需要的 {code: 当日索引} (与回测一致)."""
    idx_map = {}
    for code in ALL_CODES:
        if code not in data:
            continue
        df = data[code]
        mask = df["trade_date"] <= td
        if mask.sum() < warmup:
            continue
        idx_map[code] = mask.sum() - 1
    return idx_map


# --------------------------------------------------------------------------- #
# 报告输出
# --------------------------------------------------------------------------- #
def fmt_money(x: float) -> str:
    return f"{x:,.0f}"


def print_header(td) -> None:
    print("\n" + "=" * 56)
    print(f"  七星V4 实盘信号  |  {td}")
    print("=" * 56)


def print_account(state: dict, data: dict, td) -> None:
    holding = state["holding"]
    cash = state["cash"]
    mv = 0.0
    if holding:
        p = price_on(data, holding, td)
        if p:
            mv = state["shares"] * p
    total = cash + mv
    init = state["initial_capital"]
    print(f"\n  【账户】 总值 {fmt_money(total)} 元 (累计 {(total / init - 1) * 100:+.1f}%)")
    if holding:
        p = price_on(data, holding, td)
        cost = state["entry_price"]
        pnl = (p / cost - 1) * 100 if cost else 0
        print(
            f"    持仓: {name_of(holding)} ({holding}) "
            f"{state['shares']}股 @ {cost:.3f} → 现价 {p:.3f} ({pnl:+.1f}%)"
        )
        print(f"    现金: {fmt_money(cash)} 元")
    else:
        print(f"    持仓: 空仓 (全部现金 {fmt_money(cash)} 元)")


def print_momentum_board(data: dict, td, holding: str | None, target: str) -> None:
    """动量排行 + 每个ETF的状态."""
    idx_map = build_etf_data_at_date(data, td)
    rows = []
    for code in ETF_POOL:
        if code not in idx_map:
            continue
        df = data[code]
        close = df["close"].values[: idx_map[code] + 1].astype(float)
        if len(close) < 121:
            continue
        score = calc_momentum_score(close)
        # 单日跌幅过滤状态
        drop_flag = ""
        for i in range(-DROP_LOOKBACK, 0):
            r = (close[i] - close[i - 1]) / close[i - 1]
            if r < DROP_THRESHOLD:
                drop_flag = f"⛔近{DROP_LOOKBACK}日暴跌"
                break
        rows.append((code, score, drop_flag))
    rows.sort(key=lambda x: -x[1])

    w_label = "+".join(f"{p}日×{w}" for p, w in zip(MOM_PERIODS, MOM_WEIGHTS, strict=False))
    print(f"\n  【动量排行】({w_label})")
    for rank, (code, score, flag) in enumerate(rows, 1):
        tag = ""
        if code == holding:
            tag += " ◀持仓"
        if code == target:
            tag += " ◀目标"
        status = flag if flag else ("入选" if score > 0 else "动量<0")
        print(f"    {rank}. {name_of(code):<8} {score * 100:+6.2f}%  [{status}]{tag}")


def print_orders(sell: tuple | None, buy: tuple | None) -> None:
    print(f"\n  {'─' * 56}")
    print("  📋 今日操作指令")
    print(f"  {'─' * 56}")
    if sell:
        code, shares, price, amount = sell
        print(f"    ① 卖出  {name_of(code)} ({code})")
        print(f"       {shares} 股 @ 约 {price:.3f} 元 ≈ {fmt_money(amount)} 元")
    if buy:
        code, shares, price, amount = buy
        print(f"    ② 买入  {name_of(code)} ({code})")
        print(f"       {shares} 股 @ 约 {price:.3f} 元 ≈ {fmt_money(amount)} 元")
    if not sell and not buy:
        print("    无操作")
    print(f"  {'─' * 56}")


def translate_risk_event(etype: str) -> str:
    """风控事件类型 → 大白话 (供 Bark 推送展示)."""
    return {
        "改进-H1/H2降仓": "持仓亏损超15% 或 当日大跌超5%",
        "改进-高波动衰减降仓": "波动过大且趋势转弱",
        "H3放行止损-降仓": "放行品种自高点回撤超2%",
        "熔断-30%清仓": "总资产从高点回撤30%, 强制清仓",
        "熔断-25%告警": "总资产从高点回撤25%, 预警",
        "熔断-12%降仓": "总资产从高点回撤12%",
        "目标不可交易": "目标品种停牌/涨跌停",
        "H1/H2硬触发": "持仓亏损超15% 或 当日大跌超5%",
        "层2衰减退出": "动量持续衰减, 退出持仓",
    }.get(etype, etype)


def notify_hold(td, holding: str | None, state: dict, data: dict) -> None:
    """调仓日继续持有 → 推送一条平安通知."""
    total = account_value(state, data, td)
    ret = (total / state["initial_capital"] - 1) * 100
    holding_disp = f"【{holding}】{name_of(holding)}" if holding else "现金"
    body = (
        f"📅 {td} 调仓日 · 无操作\n"
        f"继续持有 {holding_disp}\n"
        f"💰 账户 {fmt_money(total)} 元 ({ret:+.1f}%)"
    )
    push_bark("✅ 今日不调仓", body, level="active")


def notify_trade(td, sell_order, buy_order, reason: str, state: dict, data: dict) -> None:
    """调仓日换仓 → 推送紧急买卖指令 (突破勿扰)."""
    total = account_value(state, data, td)
    ret = (total / state["initial_capital"] - 1) * 100
    lines = [f"📅 {td} 调仓日", ""]
    if sell_order:
        code, shares, price, amount = sell_order
        lines.append(
            f"① 卖出 【{code}】{name_of(code)}  {shares}股 @{price:.3f} ≈ {fmt_money(amount)}元"
        )
    if buy_order:
        code, shares, price, amount = buy_order
        lines.append(
            f"② 买入 【{code}】{name_of(code)}  {shares}股 @{price:.3f} ≈ {fmt_money(amount)}元"
        )
    lines += ["", f"💡 {reason}", f"💰 账户 {fmt_money(total)} 元 ({ret:+.1f}%)", ""]
    lines.append("👉 易淘金App按代码下单:")
    if sell_order:
        lines.append(f"   卖出 {sell_order[0]} ({name_of(sell_order[0])}) {sell_order[1]}股")
    if buy_order:
        lines.append(f"   买入 {buy_order[0]} ({name_of(buy_order[0])}) {buy_order[1]}股")
    lines.append("完成后打开记账网页确认成交")
    push_bark("🔄 调仓指令 · 请操作", "\n".join(lines), level="timeSensitive", sound="alarm")


def notify_risk_reduction(risk: RiskDecision) -> None:
    """仅在实际暴露小于 100% 时发送降仓 Bark, 纯告警留在日志."""
    if not risk.events or risk.action == ACTION_EMERGENCY or risk.exposure >= 1.0:
        return
    try:
        evs = risk.events[:3]
        lines = [f"⚠️ 自动降仓: 仓位从100% → {risk.exposure:.0%}", ""]
        for ev in evs:
            lines.append(f"• {translate_risk_event(ev['type'])}")
        lines += [
            "",
            "✅ 当前持仓不动，今天不卖",
            f"📌 下次调仓买入按{risk.exposure:.0%}仓位执行，保留{1 - risk.exposure:.0%}现金",
        ]
        push_bark(
            f"⚠️ 自动降仓至{risk.exposure:.0%}仓位",
            "\n".join(lines),
            level="timeSensitive",
            sound="alarm",
        )
    except Exception as e:
        print(f"  ⚠️ 降仓 Bark 推送失败: {e}")


# --------------------------------------------------------------------------- #
# P1 安全校验 (R4 交易日历 / R7 数据完整性 / R8 实时数据双通道)
# --------------------------------------------------------------------------- #
def is_trading_day(td) -> bool:
    """判断 td 是否为 A 股交易日 (R4, 基于新浪交易日历).

    非交易日 (周末/节假日) 跳过信号生成。日历获取失败时 fail-closed，
    防止工作日节假日误生成订单。
    """
    try:
        import akshare as ak

        cal = ak.tool_trade_date_hist_sina()
        cal_dates = set(pd.to_datetime(cal["trade_date"]).dt.date)
        return td in cal_dates
    except Exception as e:
        print(f"  ⚠️  交易日历获取失败, fail-closed: {e}")
        return False


def check_data_availability(data: dict, td) -> tuple[bool, list[str]]:
    """检查所有 ETF_POOL 代码在 td 当日是否有数据 (R7, fail-closed).

    不允许静默缩小候选池: 任何一只 ETF 缺数据都视为数据不完整。
    注意: 只检查 ETF_POOL (8只ETF), 不检查 DEFENSE (货币基金, 无需动量)。
    """
    missing = []
    for code in ETF_POOL:
        if code not in data:
            missing.append(code)
            continue
        df = data[code]
        if (df["trade_date"] == td).sum() == 0:
            missing.append(code)
    return (len(missing) == 0, missing)


def validate_realtime_data(spot_map: dict) -> tuple[bool, str]:
    """校验实时行情数据有效性 (R8, 策略信号通道, fail-closed).

    展示通道 (/api/status) 允许昨收兜底并标记 stale=true,
    但策略信号通道要求: 8只ETF全部有效才允许计算信号。
      1. 行情时间: spot 必须携带 ts 且在 5 分钟内 (非过期缓存)
      2. 价格: price 为有限数且 > 0
    任何一项不满足 → fail-closed (不生成信号)。
    """
    import math

    if not spot_map:
        return (False, "实时行情为空")
    now = time.time()
    for code in ETF_POOL:
        info = spot_map.get(code)
        if not info:
            return (False, f"{code} {name_of(code)} 实时行情缺失")
        ts = info.get("ts", 0)
        if not ts or (now - ts) > 300:
            return (False, f"{code} {name_of(code)} 行情时间过期")
        price = info.get("price", 0)
        try:
            price = float(price)
        except (TypeError, ValueError):
            return (False, f"{code} {name_of(code)} 实时价格无效: {price}")
        if not math.isfinite(price) or price <= 0:
            return (False, f"{code} {name_of(code)} 实时价格无效: {price}")
    return (True, "")


# --------------------------------------------------------------------------- #
# P2-O7: 数据源交叉校验 (腾讯 prev_close vs 新浪历史 close)
# --------------------------------------------------------------------------- #
# 不同类型 ETF 的允许偏差阈值 (QDII/LOF 因汇率和时区差异, 阈值更大)
_CROSS_VALIDATE_THRESHOLDS: dict[str, float] = {
    "513100": 0.02,  # 纳指ETF (QDII, 海外市场, 汇率影响)
    "501018": 0.015,  # 南方原油 (LOF, 估值差异)
    "161226": 0.015,  # 白银LOF (LOF, 估值差异)
    "518880": 0.01,  # 黄金ETF (商品, T+0)
    "159985": 0.01,  # 豆粕ETF (商品, T+0)
    "159915": 0.005,  # 创业板ETF (A股, T+1)
    "511220": 0.005,  # 城投债ETF (债券, 波动小)
    "511880": 0.005,  # 货币基金 (极低波动)
}


def cross_validate_data_sources(data: dict, spot_map: dict) -> tuple[bool, list[str]]:
    """交叉校验: 腾讯 prev_close vs 新浪历史上一交易日 close (P2-O7).

    比较逻辑 (非直接比较盘中实时价 vs 历史收盘):
      - 腾讯返回的 prev_close = 上一交易日收盘价
      - 新浪 parquet 的最后一条 close = 上一交易日收盘价
      - 两者应一致, 偏差超阈值 → 数据源冲突 → fail-closed

    Args:
        data: parquet 历史数据 {code: DataFrame}
        spot_map: 腾讯实时行情 {code: {"price", "prev_close", "ts"}}

    Returns:
        (ok, conflicts): ok=True 表示全部通过, conflicts 为冲突描述列表
    """
    conflicts: list[str] = []
    for code in ETF_POOL:
        if code not in data or code not in spot_map:
            continue
        tencent_prev = spot_map[code].get("prev_close", 0)
        if tencent_prev <= 0:
            continue
        df = data[code]
        if len(df) == 0:
            continue
        sina_last_close = float(df.iloc[-1]["close"])
        if sina_last_close <= 0:
            continue
        diff = abs(tencent_prev - sina_last_close) / sina_last_close
        threshold = _CROSS_VALIDATE_THRESHOLDS.get(code, 0.01)
        if diff > threshold:
            conflicts.append(
                f"{code} {name_of(code)}: 腾讯prev_close={tencent_prev:.4f} "
                f"vs 新浪close={sina_last_close:.4f} (偏差{diff:.2%}>阈值{threshold:.2%})"
            )
    return (len(conflicts) == 0, conflicts)


# --------------------------------------------------------------------------- #
# 实时行情注入 (解决14:50信号看不到当天数据的问题)
# --------------------------------------------------------------------------- #
def inject_realtime(data: dict, spot_map: dict | None = None) -> dict:
    """将当日实时行情注入内存数据 (不写parquet, 仅用于信号计算).

    数据源: 腾讯行情接口 qt.gtimg.cn (免费/无认证/覆盖全部场内ETF+LOF+QDII).
    复用 _fetch_tencent_spot(), 与实时急跌保护同源, 保证一致性。

    R8 策略信号通道: 8只ETF全部有效才注入, 任何一只缺失/无效则 fail-closed
    (不注入、不生成信号), 不用昨收填充。DEFENSE (货币基金) 仍用昨收 (非策略通道)。
    """
    today = _today_sh()

    # 空集合守卫: data 为空或不含交易池标的时 all() 恒真, 会静默跳过注入
    pool_codes = [c for c in ALL_CODES if c in data]
    if not pool_codes:
        print("  ⚠️  无交易池行情数据, 无法注入实时数据 (fail-closed)")
        return data

    # 检查是否已有今天数据 (需全部 ETF 都有今天数据才跳过)
    has_today = all(data[c]["trade_date"].max() >= today for c in pool_codes)
    if has_today:
        return data  # 已有今天数据, 无需注入

    # 腾讯实时行情 (复用 _fetch_tencent_spot, 与急跌保护同源)
    spot_map = spot_map if spot_map is not None else _fetch_tencent_spot()
    if not spot_map:
        print("  ⚠️  腾讯行情获取失败, 未注入实时数据 (fail-closed)")
        return data

    # R8 策略信号通道: 全部 ETF 有效才注入, 不用昨收填充
    ok, reason = validate_realtime_data(spot_map)
    if not ok:
        print(f"  ⚠️  实时行情校验失败: {reason}, 未注入实时数据 (fail-closed)")
        return data

    # P2-O7: 数据源交叉校验 (腾讯 prev_close vs 新浪历史 close)
    ok, conflicts = cross_validate_data_sources(data, spot_map)
    if not ok:
        msg = "数据源冲突:\n" + "\n".join(f"  - {c}" for c in conflicts)
        print(f"  ⚠️  {msg}")
        print("  ⚠️  未注入实时数据 (fail-closed), 不生成信号")
        push_bark("⚠️ 行情数据冲突", msg, level="timeSensitive", sound="alarm")
        return data

    injected = []
    for code in list(data.keys()):
        df = data[code]
        if code in spot_map:
            s = spot_map[code]
            new_row = {
                "trade_date": today,
                "open": s["prev_close"],  # 用昨收近似开盘
                "close": s["price"],  # 实时价作为当天收盘
                "high": max(s["price"], s["prev_close"]),
                "low": min(s["price"], s["prev_close"]),
                "volume": 0.0,
            }
        elif code in ETF_POOL:
            # R8 策略通道: ETF 缺实时数据 → fail-closed, 不用昨收填充
            print(f"  ⚠️  {code} {name_of(code)} 实时数据缺失, 未注入实时数据 (fail-closed)")
            return data
        else:
            # DEFENSE (货币基金) 无盘中实时价, 用昨收填充 (非策略信号通道)
            last_row = df.iloc[-1]
            new_row = {
                "trade_date": today,
                "open": float(last_row["close"]),
                "close": float(last_row["close"]),
                "high": float(last_row["close"]),
                "low": float(last_row["close"]),
                "volume": 0.0,
            }
        if "symbol" in df.columns:
            new_row["symbol"] = code
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df = df.sort_values("trade_date").drop_duplicates("trade_date").reset_index(drop=True)
        data[code] = df
        injected.append(code)

    if injected:
        print(f"  ⚡ 已注入当日实时行情 ({today}): {len(injected)}只ETF [腾讯源]")
        print("     注: 使用实时价作为当日收盘, 与回测(真实收盘)有微小差异")
    return data


def evaluate_v4_overlay(
    *,
    data: dict,
    idx_map: dict[str, int],
    trading_dates: list,
    td,
    holding: str | None,
    base_target: str,
    candidates: list[tuple[str, float]],
    scheduled_rebalance: bool,
    state: dict,
    mode: str,
) -> dict:
    """Evaluate the shared V4 core and return an auditable live decision."""
    factor_codes = (*tuple(ETF_POOL), DEFENSE)
    factors = v4.compute_factors(data, idx_map, factor_codes)
    candidate_codes = {code for code, _score in candidates}
    factors = {
        code: factor if code in candidate_codes else replace(factor, eligible=False)
        for code, factor in factors.items()
    }
    raw = v4.raw_candidate(holding, factors)
    raw_target = raw.target if raw.triggered else None
    runtime = state.get("v4_state", default_v4_state())
    history, signal_hits = v4.update_candidate_history(
        runtime.get("candidate_history", []),
        trade_date=td,
        raw_target=raw_target,
        trading_dates=trading_dates,
    )
    days_since_early = v4.trading_days_since(
        runtime.get("last_early_rotation_date"), td, trading_dates
    )
    held_factor = factors.get(holding) if holding else None
    proposed_target = base_target
    proposed_rebalance = scheduled_rebalance
    scheduled_lock = bool(
        scheduled_rebalance
        and base_target != holding
        and days_since_early < v4.V4_PARAMS.minimum_hold_days
        and held_factor is not None
        and held_factor.slow_momentum > 0.0
    )
    if scheduled_lock:
        proposed_target = holding or base_target

    decision = v4.FullPoolDecision(False)
    if not scheduled_rebalance:
        decision = v4.decide_full_pool_handoff(
            holding=holding,
            factors=factors,
            params=v4.V4_PARAMS,
            signal_hits=signal_hits,
            days_since_rotation=days_since_early,
        )
        if decision.triggered and decision.target:
            proposed_target = decision.target
            proposed_rebalance = True

    active = mode == "V4"
    final_target = proposed_target if active else base_target
    final_rebalance = proposed_rebalance if active else scheduled_rebalance
    return {
        "mode": mode,
        "active": active,
        "history": history,
        "raw_target": raw_target,
        "signal_hits": signal_hits,
        "days_since_early_rotation": days_since_early,
        "scheduled_lock": scheduled_lock,
        "decision": decision,
        "base_target": base_target,
        "proposed_target": proposed_target,
        "target": final_target,
        "is_rebalance": final_rebalance,
        "factors": factors,
    }


def _decision_snapshot(
    *,
    td,
    holding: str | None,
    candidates: list[tuple[str, float]],
    overlay: dict,
    risk: RiskDecision,
    final_target: str,
    spot_map: dict,
) -> dict:
    factors = overlay["factors"]
    snapshot = {
        "strategy_name": v4.STRATEGY_NAME,
        "strategy_id": v4.STRATEGY_ID,
        "config_hash": v4.CONFIG_HASH,
        "mode": overlay["mode"],
        "trade_date": str(td),
        "created_at": datetime.now(_SH_TZ).isoformat(timespec="seconds"),
        "holding": holding,
        "v3g_target": overlay["base_target"],
        "raw_v4_target": overlay["raw_target"],
        "confirmation_hits": overlay["signal_hits"],
        "confirmation_required": v4.V4_PARAMS.confirmation_hits,
        "days_since_early_rotation": overlay["days_since_early_rotation"],
        "scheduled_lock": overlay["scheduled_lock"],
        "v4_triggered": overlay["decision"].triggered,
        "v4_blocked_by": overlay["decision"].blocked_by,
        "v4_reasons": list(overlay["decision"].reasons),
        "v4_proposed_target": overlay["proposed_target"],
        "risk_action": risk.action,
        "risk_exposure": risk.exposure,
        "final_target": final_target,
        "is_rebalance": overlay["is_rebalance"] or risk.action == ACTION_EMERGENCY,
        "eligible_candidates": [code for code, _score in candidates],
        "candidate_scores": dict(candidates),
        "factors": v4.factors_payload(factors),
        "spot_hash": hashlib.sha256(
            json.dumps(spot_map, sort_keys=True, default=str).encode()
        ).hexdigest(),
    }
    identity = dict(snapshot)
    identity.pop("created_at")
    snapshot["decision_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()
    return snapshot


def _record_v4_state(state: dict, overlay: dict, snapshot: dict) -> None:
    runtime = state.setdefault("v4_state", default_v4_state())
    runtime["schema_version"] = v4.STATE_SCHEMA_VERSION
    runtime["strategy_id"] = v4.STRATEGY_ID
    runtime["config_hash"] = v4.CONFIG_HASH
    if overlay["mode"] in {"V4", "V4_SHADOW"}:
        runtime["candidate_history"] = overlay["history"]
    state["last_decision"] = snapshot


def _save_decision_snapshot(snapshot: dict) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{snapshot['trade_date']}.json"
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
        if existing.get("decision_id") != snapshot.get("decision_id"):
            raise RuntimeError(f"{path} 已存在不同决策, 拒绝覆盖")
        return
    temp = path.with_suffix(".tmp")
    with open(temp, "w") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)
    _fsync_dir(path.parent)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(dry_run: bool = False) -> int:
    """生成当日信号. 返回 0=成功, 1=数据错误/失败."""
    state = load_state()
    if state is None:
        print("  ❌ 尚未初始化账户, 请先运行:")
        print("     uv run python scripts/live_signal.py --init 100000")
        return 1

    # I-V4A-01: 待确认订单只抑制【新交易信号】, 不得阻塞风控监控与熔断告警。
    # 历史教训: 旧版在此直接 return, 用户不点确认期间 -30% 组合熔断发不出去。
    pending = state.get("pending_order")
    has_pending = bool(pending and pending.get("status") == "pending")
    if has_pending:
        assert pending is not None  # has_pending 蕴含 pending 非空 (类型收窄)
        print(f"  ⚠️ 尚有 {pending.get('date')} 待确认订单: 新交易信号将被抑制")
        print("     风控监控照常运行; 请尽快在网页确认成交或明确跳过")

    mode = get_strategy_mode()
    runtime = state.get("v4_state", default_v4_state())
    if (
        mode in {"V4", "V4_SHADOW"}
        and runtime.get("candidate_history")
        and runtime.get("config_hash") != v4.CONFIG_HASH
    ):
        print("  ❌ V4 参数哈希变化但确认历史非空, 拒绝混用状态")
        return 1
    print(f"  策略模式: {mode} ({v4.STRATEGY_ID})")

    print("  加载数据...")
    data = load_data()
    if not data:
        print("  ❌ 无数据缓存, 请先运行: uv run python scripts/live_signal.py --bootstrap")
        return 1
    if not dry_run:
        data = update_data(data)

    today = _today_sh()

    # R4: 交易日历校验 - 非交易日 (周末/节假日) 跳过信号生成
    if not is_trading_day(today):
        print(f"\n  😴 今日 ({today}) 非交易日 (周末/节假日), 跳过信号生成")
        return 0

    # 14:50只获取一次腾讯快照，注入、急跌过滤和审计共用同一份数据。
    spot_map = _fetch_tencent_spot()
    ok, reason = validate_realtime_data(spot_map)
    if not ok:
        msg = f"{today} 腾讯实时行情不可用: {reason}"
        print(f"\n  ⚠️  {msg}")
        if not dry_run:
            push_bark("⚠️ 实时行情获取失败", msg, level="timeSensitive", sound="alarm")
        return 1
    data = inject_realtime(data, spot_map)

    trading_dates = get_trading_dates(data)
    td = trading_dates[-1]  # 最新交易日 (注入后=今天)

    # R8: 今日为交易日但实时数据注入失败 (数据停在昨日) → fail-closed
    if td < today:
        msg = f"{today} 实时行情注入失败 (数据停在 {td}), 不生成信号, 保持原持仓"
        print(f"\n  ⚠️  {msg}")
        push_bark("⚠️ 实时行情获取失败", msg, level="timeSensitive", sound="alarm")
        return 1

    print_header(td)

    # 幂等: 今日已处理过
    if not dry_run and state.get("last_run_date") == str(td):
        print("\n  ✓ 今日已生成过信号, 以下为当前状态:")
        print_account(state, data, td)
        print("\n  (如需强制重算, 使用 --dry-run 预演)")
        return 0

    # R7: 数据缺失 fail-closed - 检查所有 ETF_POOL 当日数据完整性
    ok, missing = check_data_availability(data, td)
    if not ok:
        msg = f"{td} 数据缺失: {', '.join(missing)}\n不生成新信号, 保持原持仓 (DATA_UNAVAILABLE)"
        print(f"\n  ⚠️  {msg}")
        push_bark("⚠️ 数据不完整·暂停信号", msg, level="timeSensitive", sound="alarm")
        return 1

    # 是否为调仓日: 绝对网格 (与回测一致: trading_dates[warmup:][::5])
    # 回测逻辑: 从公共日期第130天起, 每隔5天为一个调仓日
    # 实盘必须用同一网格, 否则调仓日会与回测逐渐错位
    warmup = 130
    all_trading = trading_dates  # 公共交易日历
    if len(all_trading) > warmup:
        post_warmup = all_trading[warmup:]  # 与回测 trading_dates 一致
        rebalance_grid = set(post_warmup[::REBALANCE_DAYS])  # 绝对网格
        is_rebalance = td in rebalance_grid
        # 计算距下次调仓还有几天
        if is_rebalance:
            days_since = REBALANCE_DAYS
        else:
            try:
                td_idx = post_warmup.index(td)
                days_since = td_idx % REBALANCE_DAYS
            except ValueError:
                days_since = 0
                is_rebalance = True  # 找不到就当作调仓日
    else:
        is_rebalance = True
        days_since = REBALANCE_DAYS

    holding = state["holding"]

    # === 核心决策 (与回测同一函数) ===
    idx_map = build_etf_data_at_date(data, td)
    target, candidates, _best_score, a_share_weak = select_target(data, idx_map, holding)

    # === 实时急跌保护 (V3-G): 检查当天盘中是否已暴跌>3% ===
    # 同步门控逻辑: ret60>=0.01 或 动量≤0 → 排除; 否则放行 (假摔)
    realtime_dropped = []
    for code, _score in candidates:
        spot = spot_map.get(code)
        if not spot:
            continue
        rt_price = spot["price"]
        prev_close = spot["prev_close"]
        if prev_close and prev_close > 0:
            intraday_ret = (rt_price - prev_close) / prev_close
            if intraday_ret < -0.03:
                # 检查门控条件: 用历史数据计算 ret60 和动量
                df = data[code]
                idx = int((df["trade_date"] <= td).sum())
                if idx > 61:
                    close = df["close"].values[:idx].astype(float)
                    ret60 = (close[-1] - close[-61]) / close[-61]
                    r10 = (close[-1] - close[-11]) / close[-11]
                    r20 = (close[-1] - close[-21]) / close[-21]
                    mom = 0.5 * r10 + 0.5 * r20
                    if ret60 >= 0.01 or mom <= 0:
                        realtime_dropped.append((code, intraday_ret))
                else:
                    realtime_dropped.append((code, intraday_ret))
    if realtime_dropped:
        dropped_codes = {c for c, _ in realtime_dropped}
        for code, ret in realtime_dropped:
            print(f"  ⚡ 实时急跌: {name_of(code)} 当日{ret:+.1%}, 已排除")
        print("     注: 此过滤为实盘保护, 回测中不存在(回测用真实收盘价自然触发)")
        # 从candidates中移除, 重新选目标
        candidates = [(c, s) for c, s in candidates if c not in dropped_codes]
        target = candidates[0][0] if candidates else DEFENSE

    overlay = evaluate_v4_overlay(
        data=data,
        idx_map=idx_map,
        trading_dates=trading_dates,
        td=td,
        holding=holding,
        base_target=target,
        candidates=candidates,
        scheduled_rebalance=is_rebalance,
        state=state,
        mode=mode,
    )
    target = overlay["target"]
    is_rebalance = overlay["is_rebalance"]
    if overlay["raw_target"]:
        print(
            f"  V4共振候选: {name_of(overlay['raw_target'])} "
            f"({overlay['signal_hits']}/{v4.V4_PARAMS.confirmation_hits}日确认)"
        )
    if overlay["scheduled_lock"] and mode == "V4":
        print("  🔒 V4提前轮动3交易日锁: 本次网格换仓保持原持仓")
    decision = overlay["decision"]
    if decision.triggered and decision.target:
        verb = "执行" if mode == "V4" else "影子观察"
        print(f"  ⚡ V4严格快慢共振{verb}: → {name_of(decision.target)}")

    # === V32 尾部风控层 (与回测 risk_overrides 同一纯函数, 零副作用) ===
    cur_val = account_value(state, data, td)
    if cur_val > state.get("peak_equity", 0.0):
        state["peak_equity"] = cur_val
    risk = risk_assess(
        target=target,
        holding=holding,
        state=state,
        data=data,
        td=td,
        idx_map=idx_map,
        is_rebalance=is_rebalance,
        common_dates=trading_dates,
        spot_map=spot_map,
    )
    if risk.events:
        for ev in risk.events:
            print(f"  🛡️ 风控: {ev['type']} | {ev.get('reason', '')}")
        # V32/V3-G: 只有真实降仓才发二级告警; 关闭降仓层时仅保留日志审计.
        notify_risk_reduction(risk)
    target = risk.final_target or DEFENSE
    if risk.action == ACTION_EMERGENCY:
        print(f"  🛡️ 组合熔断: 强制切换防御 {risk.final_target} (非调仓日也执行)")
        is_rebalance = True  # 复用既有调仓交易链路

    snapshot = _decision_snapshot(
        td=td,
        holding=holding,
        candidates=candidates,
        overlay=overlay,
        risk=risk,
        final_target=target,
        spot_map=spot_map,
    )
    snapshot["is_rebalance"] = is_rebalance

    print_account(state, data, td)
    print_momentum_board(data, td, holding, target)

    if a_share_weak:
        print(f"\n  ⚠️  A股走弱 (创业板<MA{A_SHARE_MA}), 已排除创业板ETF")

    if not is_rebalance:
        wait = REBALANCE_DAYS - days_since
        print(f"\n  😴 今日非调仓日 (距下次还有 {wait} 个交易日)")
        print(f"     继续持有 {name_of(holding) if holding else '现金'}, 无操作")
        if not dry_run:
            with state_transaction() as st:
                st["last_run_date"] = str(td)
                st["peak_equity"] = state.get("peak_equity", st["initial_capital"])
                st["risk_exposure"] = risk.exposure
                st["cooldown_until"] = str(risk.cooldown_until) if risk.cooldown_until else None
                if risk.events:
                    _append_risk_log(st, risk.events)
                _record_v4_state(st, overlay, snapshot)
            _save_decision_snapshot(snapshot)
        return 0

    # === 调仓日 ===
    print(f"\n  🔄 今日为调仓日 (每{REBALANCE_DAYS}个交易日)")

    if target == holding:
        print(f"     信号: 继续持有 {name_of(target) if target else '现金'}, 无操作")
        if risk.exposure < 1.0:
            print(f"  ⚠️ 风控暴露 {risk.exposure:.0%}: 降仓待下次换仓生效 (当前继续持有)")
        if not dry_run:
            with state_transaction() as st:
                st["last_rebalance_date"] = str(td)
                st["last_run_date"] = str(td)
                st["peak_equity"] = state.get("peak_equity", st["initial_capital"])
                st["risk_exposure"] = risk.exposure
                st["cooldown_until"] = str(risk.cooldown_until) if risk.cooldown_until else None
                if risk.events:
                    _append_risk_log(st, risk.events)
                _record_v4_state(st, overlay, snapshot)
                state = st
            _save_decision_snapshot(snapshot)
            notify_hold(td, target, state, data)
        return 0

    # === 需要换仓: 生成订单 ===
    sell_order = None
    buy_order = None
    cash = state["cash"]

    # 1. 卖出当前持仓
    if holding:
        p_sell = price_on(data, holding, td)
        if p_sell:
            amount = state["shares"] * p_sell * (1 - FEE - SLIPPAGE)
            sell_order = (holding, state["shares"], p_sell, amount)
            cash += amount

    # 2. 买入目标 (预留1%现金, 按100股整数; V32风控: 按 exposure 折算降仓)
    if target:
        p_buy = price_on(data, target, td)
        if p_buy:
            shares = int(cash * risk.exposure * 0.99 / p_buy / 100) * 100
            if shares > 0:
                cost = shares * p_buy * (1 + FEE + SLIPPAGE)
                buy_order = (target, shares, p_buy, cost)

    print_orders(sell_order, buy_order)

    why = dict(candidates).get(target, 0)
    cur = dict(candidates).get(holding, None) if holding else None
    if risk.action == ACTION_EMERGENCY:
        reason = f"组合回撤熔断, 强制切防御 {name_of(target)} (若来不及, 次日开盘执行亦可)"
    elif decision.triggered and decision.target and mode == "V4":
        reason = (
            f"V4全池严格快慢共振连续{overlay['signal_hits']}日确认, 提前换仓至 {name_of(target)}"
        )
    elif target == DEFENSE:
        reason = "所有ETF动量转弱, 切入货币基金防御"
    else:
        reason = f"{name_of(target)} 动量 {why * 100:+.1f}% 为最强" + (
            f" (当前持仓 {cur * 100:+.1f}%)" if cur is not None else ""
        )
    if 0 < risk.exposure < 1.0 and risk.action != "emergency_defense":
        print(f"  ⚠️ 风控暴露 {risk.exposure:.0%} (本次买入按此比例折算)")
        reason += f" | 风控暴露 {risk.exposure:.0%}"
    print(f"  原因: {reason}")

    if dry_run:
        print("\n  [预演模式] 以上为模拟信号, 未改动账户状态")
        return 0

    # === I-V4A-01: 有待确认订单 → 抑制新订单, 但风控结论必须落盘并告警 ===
    # 故意不写 last_run_date: 用户在网页处理完 pending 后, 当日重跑可补生成信号.
    # 代价是重复运行会重复推送抑制告警 (cron 每日只跑一次, 可接受).
    if has_pending:
        assert pending is not None  # has_pending 蕴含 pending 非空
        msg = (
            f"{td} 本日信号 ({reason}) 被 {pending.get('date')} 的待确认订单抑制。\n"
            "请先在网页确认/跳过旧订单; 处理后重新运行即可生成今日信号。"
        )
        print(f"\n  ⛔ {msg}")
        with state_transaction() as st:
            st["peak_equity"] = state.get("peak_equity", st["initial_capital"])
            st["risk_exposure"] = risk.exposure
            st["cooldown_until"] = str(risk.cooldown_until) if risk.cooldown_until else None
            if risk.events:
                _append_risk_log(st, risk.events)
            _record_v4_state(st, overlay, snapshot)
        _save_decision_snapshot(snapshot)
        if risk.action == ACTION_EMERGENCY:
            push_bark(
                "🚨 组合熔断触发, 但被待确认订单阻塞",
                msg + "\n请立即人工处理持仓!",
                level="timeSensitive",
                sound="alarm",
            )
            return 1  # 熔断未执行 = 未解决的关键状态, 让 cron 日志与非零退出可见
        push_bark("⚠️ 新调仓信号被待确认订单抑制", msg)
        return 0

    # === 模拟记账模式: 信号按理论价自动成交 (尚无法真实下单的阶段) ===
    if is_paper_mode():
        with state_transaction() as st:
            if sell_order:
                st["cash"] += sell_order[3]
                st["trade_log"].append(
                    {
                        "date": str(td),
                        "action": "sell",
                        "code": holding,
                        "name": name_of(holding),
                        "shares": st["shares"],
                        "price": sell_order[2],
                        "amount": sell_order[3],
                    }
                )
                st["holding"] = None
                st["shares"] = 0
                st["entry_price"] = 0.0
            if buy_order:
                st["cash"] -= buy_order[3]
                st["holding"] = target
                st["shares"] = buy_order[1]
                st["entry_price"] = buy_order[2]
                st["entry_date"] = str(td)
                st["trade_log"].append(
                    {
                        "date": str(td),
                        "action": "buy",
                        "code": target,
                        "name": name_of(target),
                        "shares": buy_order[1],
                        "price": buy_order[2],
                        "amount": buy_order[3],
                    }
                )
            st["pending_order"] = None
            st["last_rebalance_date"] = str(td)
            st["last_run_date"] = str(td)
            st["peak_equity"] = state.get("peak_equity", st["initial_capital"])
            st["risk_exposure"] = 1.0  # 交易后重置 (与回测 exposure=1.0 语义一致)
            st["cooldown_until"] = str(risk.cooldown_until) if risk.cooldown_until else None
            if risk.events:
                _append_risk_log(st, risk.events)
            if decision.triggered and buy_order and target != holding:
                st["v4_state"]["last_early_rotation_date"] = str(td)
            _record_v4_state(st, overlay, snapshot)
            state = st
        _save_decision_snapshot(snapshot)
        print(f"\n  ✓ [模拟记账] 已按理论价自动记录: {STATE_FILE}")
        notify_trade(td, sell_order, buy_order, reason + " [模拟记账]", state, data)
        return 0

    # === 保存为【待确认】订单 (不自动成交, 以用户在网页填入的真实成交为准) ===
    if not dry_run:
        with state_transaction() as st:
            order_id = hashlib.sha256(f"{snapshot['decision_id']}:{target}".encode()).hexdigest()[
                :24
            ]
            st["pending_order"] = {
                "order_id": order_id,
                "decision_id": snapshot["decision_id"],
                "expected_state_version": st.get("_version", 0) + 1,
                "strategy_id": v4.STRATEGY_ID,
                "config_hash": v4.CONFIG_HASH,
                "decision_kind": (
                    "v4_early_rotation"
                    if decision.triggered and not snapshot["scheduled_lock"]
                    else "scheduled_or_risk"
                ),
                "date": str(td),
                "sell": {
                    "code": sell_order[0],
                    "name": name_of(sell_order[0]),
                    "shares": sell_order[1],
                    "price": sell_order[2],
                    "amount": sell_order[3],
                }
                if sell_order
                else None,
                "buy": {
                    "code": buy_order[0],
                    "name": name_of(buy_order[0]),
                    "shares": buy_order[1],
                    "price": buy_order[2],
                    "amount": buy_order[3],
                }
                if buy_order
                else None,
                "reason": reason,
                "status": "pending",
                "created_at": datetime.now(_SH_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            }
            st["last_rebalance_date"] = str(td)
            st["last_run_date"] = str(td)
            st["peak_equity"] = state.get("peak_equity", st["initial_capital"])
            st["risk_exposure"] = 1.0  # 交易后重置
            st["cooldown_until"] = str(risk.cooldown_until) if risk.cooldown_until else None
            if risk.events:
                _append_risk_log(st, risk.events)
            _record_v4_state(st, overlay, snapshot)
            state = st
        _save_decision_snapshot(snapshot)
        print(f"\n  ✓ 信号已保存为【待确认】: {STATE_FILE}")
        print("    👉 请在网页填入真实成交后确认 (持仓状态以网页确认为准)")

    # === 推送换仓指令到手机 ===
    notify_trade(td, sell_order, buy_order, reason, state, data)
    return 0


def show_status() -> None:
    state = load_state()
    if state is None:
        print("  ❌ 尚未初始化账户")
        return
    data = load_data()
    trading_dates = get_trading_dates(data)
    td = trading_dates[-1]
    print_header(td)
    print(f"  策略模式: {get_strategy_mode()} | {v4.STRATEGY_ID}")
    print_account(state, data, td)
    log = state.get("trade_log", [])
    if log:
        print(f"\n  【最近交易】(共{len(log)}笔)")
        for t in log[-6:]:
            arrow = "卖出" if t["action"] == "sell" else "买入"
            print(
                f"    {t['date']}  {arrow} {t['name']} "
                f"{t['shares']}股 @ {t['price']:.3f} ≈ {fmt_money(t['amount'])}元"
            )


def sync_only() -> int:
    """仅更新行情数据 (不生成信号/不改账户状态), 供夜间定时任务补齐当日K线.

    新浪当日K线在收盘后才陆续发布, 14:50/16:30 的定时任务拿不到当日数据。
    本函数幂等 (只追加新交易日), 可由夜间 cron 安全重跑。

    Returns:
        0=成功, 1=数据错误
    """
    print("  加载数据...")
    data = load_data()
    if not data:
        print("  ❌ 无数据缓存, 请先运行: uv run python scripts/live_signal.py --bootstrap")
        return 1
    update_data(data)  # 内部已写回 parquet
    print("  ✓ 数据同步完成")
    return 0


# --------------------------------------------------------------------------- #
# 网页确认成交 (实盘以用户填入的真实成交为准)
# --------------------------------------------------------------------------- #
def confirm_order(
    real_sell: dict | None,
    real_buy: dict | None,
    *,
    order_id: str | None = None,
    expected_state_version: int | None = None,
    idempotency_key: str | None = None,
) -> dict:
    """确认待确认订单, 用真实成交数据更新持仓状态.

    real_sell: {"shares": int, "price": float} 或 None (卖出当前持仓)
    real_buy:  {"code": str, "shares": int, "price": float} 或 None

    校验:
      - pending_order 必须存在且 status=pending
      - 卖出: code 必须匹配当前持仓, shares ≤ 持仓数, price > 0
      - 买入: code 必须匹配 pending_order 的 buy_code, shares > 0, price > 0, 现金充足
      - 部分卖出: 保留剩余持仓, 不清零
    """
    with state_transaction() as state:
        pending = state.get("pending_order")
        if not pending:
            raise ValueError("无待确认订单")
        if pending.get("status") != "pending":
            raise ValueError(f"订单状态非 pending: {pending.get('status')}")
        if order_id and pending.get("order_id") != order_id:
            raise ValueError("待确认订单已变化, 请刷新页面后重试")
        if expected_state_version is not None and state.get("_version") != expected_state_version:
            raise ValueError(
                f"状态版本已变化: 页面 {expected_state_version}, 当前 {state.get('_version')}"
            )
        receipts = state.setdefault("confirm_receipts", {})
        if idempotency_key and idempotency_key in receipts:
            raise ValueError("重复提交 (该确认请求已处理)")
        td = pending["date"]

        expected_sell = pending.get("sell")
        expected_buy = pending.get("buy")
        if bool(real_sell) != bool(expected_sell) or bool(real_buy) != bool(expected_buy):
            raise ValueError("确认成交必须完整包含待确认订单的全部买卖腿")

        if real_sell and expected_sell:
            if state["holding"] != expected_sell.get("code"):
                raise ValueError("当前持仓与待卖出资产不一致")
            code = state["holding"]
            shares = int(real_sell["shares"])
            price = float(real_sell["price"])
            if price <= 0:
                raise ValueError(f"卖出价格必须 > 0, 实际: {price}")
            if shares <= 0:
                raise ValueError(f"卖出数量必须 > 0, 实际: {shares}")
            if shares > state["shares"]:
                raise ValueError(f"卖出数量 {shares} 超过持仓 {state['shares']}")
            # 注: A股卖出允许零股一次性清仓, 不做整百校验;
            # 下一行的完整成交校验 (shares == 待确认数量) 已兜底
            if shares != int(expected_sell.get("shares", state["shares"])):
                raise ValueError("换仓卖出必须按待确认订单数量完整成交")
            amount = shares * price * (1 - FEE - SLIPPAGE)
            state["cash"] += amount
            state["trade_log"].append(
                {
                    "date": td,
                    "action": "sell",
                    "code": code,
                    "name": name_of(code),
                    "shares": shares,
                    "price": price,
                    "amount": amount,
                }
            )
            # 部分卖出: 保留剩余持仓
            remaining = state["shares"] - shares
            if remaining > 0:
                state["shares"] = remaining
                # holding 和 entry_price 保持不变
            else:
                state["holding"] = None
                state["shares"] = 0
                state["entry_price"] = 0.0

        if real_buy and expected_buy:
            code = real_buy["code"]
            shares = int(real_buy["shares"])
            price = float(real_buy["price"])
            if price <= 0:
                raise ValueError(f"买入价格必须 > 0, 实际: {price}")
            if shares <= 0:
                raise ValueError(f"买入数量必须 > 0, 实际: {shares}")
            if shares % 100 != 0:
                raise ValueError(f"买入股数必须是100的整数倍: {shares}")
            # 校验买入代码匹配 pending_order
            expected_code = expected_buy.get("code")
            if expected_code and code != expected_code:
                raise ValueError(f"买入代码 {code} 与待确认订单 {expected_code} 不匹配")
            amount = shares * price * (1 + FEE + SLIPPAGE)
            if state["cash"] - amount < 0:
                raise ValueError(f"现金不足: 需要 {amount:.2f}, 可用 {state['cash']:.2f}")
            state["cash"] -= amount
            state["holding"] = code
            state["shares"] = shares
            state["entry_price"] = price
            state["entry_date"] = td
            state["trade_log"].append(
                {
                    "date": td,
                    "action": "buy",
                    "code": code,
                    "name": name_of(code),
                    "shares": shares,
                    "price": price,
                    "amount": amount,
                }
            )

        pending["status"] = "confirmed"
        pending["confirmed_at"] = datetime.now(_SH_TZ).strftime("%Y-%m-%d %H:%M:%S")
        if pending.get("decision_kind") == "v4_early_rotation" and real_buy:
            state["v4_state"]["last_early_rotation_date"] = td
        if idempotency_key:
            receipts[idempotency_key] = {
                "order_id": pending.get("order_id"),
                "confirmed_at": pending["confirmed_at"],
            }
            if len(receipts) > 100:
                for key in list(receipts)[:-100]:
                    receipts.pop(key, None)
    return state


def skip_pending() -> dict:
    """标记待确认订单为'本次不操作'."""
    with state_transaction() as state:
        pending = state.get("pending_order")
        if not pending:
            raise ValueError("无待确认订单")
        if pending.get("status") != "pending":
            raise ValueError(f"订单状态非 pending: {pending.get('status')}")
        pending["status"] = "skipped"
        pending["skipped_at"] = datetime.now(_SH_TZ).strftime("%Y-%m-%d %H:%M:%S")
    return state


def record_manual_trade(
    action: str, code: str, shares: int, price: float, td: str | None = None
) -> dict:
    """手动记录一笔成交 (用于修正或非信号交易).

    校验:
      - 卖出: code 必须匹配持仓, shares ≤ 持仓数, price > 0, 部分卖出保留剩余
      - 买入: shares > 0, price > 0, 现金充足
    """
    with state_transaction() as state:
        td = td or str(_today_sh())
        shares = int(shares)
        price = float(price)
        if price <= 0:
            raise ValueError(f"价格必须 > 0, 实际: {price}")
        if shares <= 0:
            raise ValueError(f"数量必须 > 0, 实际: {shares}")

        if action == "sell":
            if state["holding"] != code:
                raise ValueError(f"卖出代码 {code} 与持仓 {state['holding']} 不匹配")
            if shares > state["shares"]:
                raise ValueError(f"卖出数量 {shares} 超过持仓 {state['shares']}")
            amount = shares * price * (1 - FEE - SLIPPAGE)
            state["cash"] += amount
            state["trade_log"].append(
                {
                    "date": td,
                    "action": "sell",
                    "code": code,
                    "name": name_of(code),
                    "shares": shares,
                    "price": price,
                    "amount": amount,
                }
            )
            remaining = state["shares"] - shares
            if remaining > 0:
                state["shares"] = remaining
            else:
                state["holding"] = None
                state["shares"] = 0
                state["entry_price"] = 0.0
        elif action == "buy":
            if shares % 100 != 0:
                raise ValueError(f"买入股数必须是100的整数倍: {shares}")
            amount = shares * price * (1 + FEE + SLIPPAGE)
            if state["cash"] - amount < 0:
                raise ValueError(f"现金不足: 需要 {amount:.2f}, 可用 {state['cash']:.2f}")
            state["cash"] -= amount
            state["holding"] = code
            state["shares"] = shares
            state["entry_price"] = price
            state["entry_date"] = td
            state["trade_log"].append(
                {
                    "date": td,
                    "action": "buy",
                    "code": code,
                    "name": name_of(code),
                    "shares": shares,
                    "price": price,
                    "amount": amount,
                }
            )
        else:
            raise ValueError(f"未知操作: {action}")
    return state


def momentum_board_data(data: dict, td, holding: str | None, target: str) -> list[dict]:
    """动量排行 (返回结构化数据, 供网页显示)."""
    idx_map = build_etf_data_at_date(data, td)
    rows = []
    for code in ETF_POOL:
        if code not in idx_map:
            continue
        df = data[code]
        close = df["close"].values[: idx_map[code] + 1].astype(float)
        if len(close) < 121:
            continue
        score = calc_momentum_score(close)
        dropped = False
        for i in range(-DROP_LOOKBACK, 0):
            r = (close[i] - close[i - 1]) / close[i - 1]
            if r < DROP_THRESHOLD:
                dropped = True
                break
        rows.append(
            {
                "code": code,
                "name": name_of(code),
                "score": round(float(score) * 100, 2),
                "dropped": dropped,
                "is_holding": code == holding,
                "is_target": code == target,
                "eligible": bool(score > 0) and not dropped,
            }
        )
    rows.sort(key=lambda x: -x["score"])  # type: ignore[operator]
    return rows


def build_equity_curve(state: dict, data: dict) -> list[dict]:
    """根据已确认成交记录, 重建账户净值曲线."""
    log = state.get("trade_log", [])
    if not log:
        return []
    trading_dates = get_trading_dates(data)
    start_date = log[0]["date"]
    trades_by_date: dict[str, list] = {}
    for t in log:
        trades_by_date.setdefault(t["date"], []).append(t)
    cash = state["initial_capital"]
    holding = None
    shares = 0
    curve = []
    for td in trading_dates:
        td_str = str(td)
        if td_str < start_date:
            continue
        for t in trades_by_date.get(td_str, []):
            if t["action"] == "sell":
                cash += t["amount"]
                holding = None
                shares = 0
            elif t["action"] == "buy":
                cash -= t["amount"]
                holding = t["code"]
                shares = t["shares"]
        value = cash
        if holding and holding in data:
            p = price_on(data, holding, td)
            if p:
                value += shares * p
        curve.append({"date": td_str, "value": round(value, 2)})
    return curve


def init_account(capital: float) -> None:
    if STATE_FILE.exists():
        print(f"  ⚠️  账户已存在: {STATE_FILE}")
        print("     如需重置, 请先手动删除该文件")
        return
    state = default_state(capital)
    save_state(state)
    print(f"  ✓ 账户已初始化: 本金 {fmt_money(capital)} 元")
    print(f"    状态文件: {STATE_FILE}")
    print("\n  下一步: 运行 uv run python scripts/live_signal.py 生成首个信号")


def main() -> None:
    parser = argparse.ArgumentParser(description="七星 V3-G/V4 实盘信号生成器")
    parser.add_argument("--init", type=float, metavar="CAPITAL", help="初始化账户本金")
    parser.add_argument("--status", action="store_true", help="查看当前持仓状态")
    parser.add_argument("--dry-run", action="store_true", help="预演信号(不改动状态)")
    parser.add_argument(
        "--set-bark",
        action="store_true",
        help="设置 Bark 设备 Key (从 stdin 或 BARK_KEY 环境变量读取)",
    )
    parser.add_argument("--notify-test", action="store_true", help="发送一条测试推送")
    parser.add_argument(
        "--bootstrap", action="store_true", help="全量拉取历史数据 (服务器首次部署)"
    )
    parser.add_argument(
        "--sync-only", action="store_true", help="仅更新行情数据 (夜间补齐当日K线, 不生成信号)"
    )
    parser.add_argument(
        "--paper-mode",
        metavar="on/off",
        help="模拟记账: on=信号按理论价自动记账, off=网页确认真实成交",
    )
    parser.add_argument(
        "--strategy-mode",
        choices=("V3-G", "V4_SHADOW", "V4"),
        help="切换持久化策略模式",
    )
    args = parser.parse_args()

    if args.strategy_mode:
        set_strategy_mode(args.strategy_mode)
        print(f"  ✓ 策略模式已切换为 {args.strategy_mode}")
        print(f"    策略ID: {v4.STRATEGY_ID}")
        print(f"    参数哈希: {v4.CONFIG_HASH}")
    elif args.bootstrap:
        bootstrap_data()
    elif args.set_bark:
        import os

        key = os.environ.get("BARK_KEY", "").strip()
        if not key:
            import getpass

            key = getpass.getpass("请输入 Bark 设备 Key: ").strip()
        if not key:
            print("  ❌ Key 不能为空")
            sys.exit(1)
        set_bark_key(key)
        print("  ✓ Bark Key 已保存")
        print("    下一步: uv run python scripts/live_signal.py --notify-test 验证能否收到")
    elif args.notify_test:
        test_push()
    elif args.paper_mode:
        on = args.paper_mode.strip().lower() in ("on", "1", "true", "yes")
        set_paper_mode(on)
        print(
            "  ✓ 模拟记账模式已"
            f"{'开启 (信号按理论价自动记账)' if on else '关闭 (网页确认真实成交)'}"
        )
    elif args.init:
        init_account(args.init)
    elif args.status:
        show_status()
    elif args.sync_only:
        sys.exit(sync_only())
    else:
        sys.exit(run(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
