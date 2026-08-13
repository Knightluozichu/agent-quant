"""8+8 asset-pool test for server V3-G with same-class early rotation.

The production eight assets each receive one long-history peer.  Selection
keeps the V3-G gate/buffer rules and the server risk path.  The only overlay is
the previously fixed rule: on a non-grid day, rotate fully when the global
leader is in the same declared class, leads by at least 0.5 percentage points,
and the previous early rotation is at least three trading days old.

Peer data is stored separately from production caches so this experiment never
rewrites the server's live data.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts import risk_overrides as ro
    from scripts import run_qixing_v3 as rq
    from scripts.exp_v3g_relative_rotation import RotationParams, run_strategy
except ModuleNotFoundError:
    import risk_overrides as ro
    import run_qixing_v3 as rq
    from exp_v3g_relative_rotation import RotationParams, run_strategy

PROJECT_ROOT = Path(__file__).parent.parent
PEER_DATA_DIR = PROJECT_ROOT / "data" / "cross_asset_expanded_peers"
OUTPUT = PROJECT_ROOT / "data" / "v9_results" / "v3g_expanded_pairs.json"

INITIAL_CAPITAL = 100_000.0
WARMUP = 130

ORIGINAL_POOL = dict(rq.ETF_POOL)
ORIGINAL_CODES = (*tuple(ORIGINAL_POOL), rq.DEFENSE)

PEER_OF = {
    "518880": "159934",  # 黄金ETF华安 -> 黄金ETF易方达
    "159985": "159980",  # 豆粕期货 -> 有色金属期货 (商品期货代理)
    "501018": "160723",  # 南方原油 -> 嘉实原油
    "161226": "518800",  # 白银 -> 黄金ETF国泰 (贵金属低波动代理)
    "513100": "159941",  # 纳指ETF国泰 -> 纳指ETF广发
    "159915": "159952",  # 创业板ETF易方达 -> 创业板ETF广发
    "511220": "511260",  # 城投债 -> 十年国债
    "511880": "511990",  # 银华日利 -> 华宝添益
}

PEER_NAMES = {
    "159934": "黄金ETF易方达",
    "159980": "有色金属期货ETF",
    "160723": "嘉实原油LOF",
    "518800": "黄金ETF国泰",
    "159941": "纳指ETF广发",
    "159952": "创业板ETF广发",
    "511260": "十年国债ETF",
    "511990": "华宝添益ETF",
}

EXTENDED_POOL = {**ORIGINAL_POOL, **PEER_NAMES}

# Gold and silver are one economic handoff class; the remaining peers are
# paired by underlying exposure or the closest long-history domestic proxy.
CLASS_GROUPS = (
    ("518880", "159934", "161226", "518800"),
    ("159985", "159980"),
    ("501018", "160723"),
    ("513100", "159941"),
    ("159915", "159952"),
    ("511220", "511260"),
    ("511880", "511990"),
)

KNOWN_SPLITS: dict[str, tuple[tuple[str, float], ...]] = {
    "159941": (("2022-07-05", 4.0),),
}


def apply_known_splits(code: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Forward-adjust pre-split OHLC and volume without touching later rows."""
    adjusted = frame.copy()
    for split_date, ratio in KNOWN_SPLITS.get(code, ()):
        split_day = pd.Timestamp(split_date).date()
        mask = adjusted["trade_date"] < split_day
        for column in ("open", "high", "low", "close"):
            adjusted.loc[mask, column] = adjusted.loc[mask, column].astype(float) / ratio
        adjusted.loc[mask, "volume"] = (
            adjusted.loc[mask, "volume"].astype(float) * ratio
        )
    return adjusted


def sina_symbol(code: str) -> str:
    prefix = "sh" if code.startswith(("5", "6")) else "sz"
    return f"{prefix}{code}"


def fetch_peer_data() -> None:
    """Fetch full peer history into an experiment-only cache."""
    import akshare as ak

    PEER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for code, name in PEER_NAMES.items():
        print(f"fetch {code} {name}...", end=" ", flush=True)
        raw = ak.fund_etf_hist_sina(symbol=sina_symbol(code))
        if raw is None or raw.empty:
            raise RuntimeError(f"No data returned for {code} {name}")
        frame = raw.rename(columns={"date": "trade_date"}).copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
        frame["symbol"] = code
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
        frame = frame[
            ["trade_date", "open", "close", "high", "low", "volume", "symbol"]
        ].dropna(subset=["trade_date", "close"])
        frame = frame.sort_values("trade_date").drop_duplicates("trade_date")
        frame = apply_known_splits(code, frame).reset_index(drop=True)
        frame.to_parquet(PEER_DATA_DIR / f"{code}.parquet", index=False)
        print(f"{len(frame)} rows {frame.trade_date.min()}..{frame.trade_date.max()}")
        time.sleep(0.2)


def load_expanded_data() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    base = rq.load_data()
    missing_base = sorted(set(ORIGINAL_CODES) - set(base))
    if missing_base:
        raise RuntimeError(f"Missing production data: {missing_base}")
    data = dict(base)
    missing_peers: list[str] = []
    for code in PEER_NAMES:
        path = PEER_DATA_DIR / f"{code}.parquet"
        if not path.exists():
            missing_peers.append(code)
            continue
        frame = pd.read_parquet(path)
        data[code] = frame.sort_values("trade_date").reset_index(drop=True)
    if missing_peers:
        raise RuntimeError(
            f"Missing peer data {missing_peers}; rerun with --fetch"
        )
    return base, data


def select_target_expanded(
    data: dict[str, pd.DataFrame],
    etf_data_at_date: dict[str, int],
    holding: str | None,
) -> tuple[str, list[tuple[str, float]], float, bool]:
    """V3-G selector extended to 15 candidates plus the original defense."""
    a_share_weak = rq.check_a_share_weak(
        data, etf_data_at_date.get("159915", 0)
    )
    candidates: list[tuple[str, float]] = []
    for code in EXTENDED_POOL:
        if code not in etf_data_at_date:
            continue
        if code in {"159915", "159952"} and a_share_weak:
            continue
        idx = etf_data_at_date[code]
        close = data[code]["close"].values[: idx + 1].astype(float)
        if len(close) < 121:
            continue
        if rq.USE_SHORT_MOM_FILTER and not rq.check_short_momentum(close):
            continue
        if rq.USE_DROP_FILTER and not rq.check_single_day_drop(close):
            continue
        score = float(rq.calc_momentum_score(close))
        if score > 0:
            candidates.append((code, score))
    candidates.sort(key=lambda item: -item[1])

    best_target = candidates[0][0] if candidates else rq.DEFENSE
    best_score = candidates[0][1] if candidates else 0.0
    threshold = 0.0 if best_score > 0.10 else 0.05

    if holding and holding in etf_data_at_date and holding != rq.DEFENSE:
        idx = etf_data_at_date[holding]
        held_close = data[holding]["close"].values[: idx + 1].astype(float)
        if len(held_close) > rq.DROP_LOOKBACK + 1 and len(held_close) > 61:
            recent_drop = any(
                held_close[i] / held_close[i - 1] - 1.0 < rq.DROP_THRESHOLD
                for i in range(-rq.DROP_LOOKBACK, 0)
            )
            ret60 = held_close[-1] / held_close[-61] - 1.0
            if recent_drop and ret60 < 0.01:
                threshold = 0.0

    if holding and holding != rq.DEFENSE:
        current_score = dict(candidates).get(holding, -999.0)
        if current_score > 0:
            target = (
                best_target
                if best_score > current_score + threshold
                else holding
            )
        else:
            target = best_target
    else:
        target = best_target
    return target, candidates, best_score, a_share_weak


@contextmanager
def expanded_runtime():
    original_pool = dict(rq.ETF_POOL)
    original_select = rq.select_target
    rq.ETF_POOL.clear()
    rq.ETF_POOL.update(EXTENDED_POOL)
    rq.select_target = select_target_expanded
    try:
        yield
    finally:
        rq.ETF_POOL.clear()
        rq.ETF_POOL.update(original_pool)
        rq.select_target = original_select


def raw_common_dates(data: dict[str, pd.DataFrame]) -> list[Any]:
    codes = [*tuple(EXTENDED_POOL), rq.DEFENSE]
    return sorted(set.intersection(*[set(data[code]["trade_date"]) for code in codes]))


def yearly_report(result: dict[str, Any]) -> dict[str, Any]:
    curve = result["equity_curve"].copy()
    curve["year"] = curve["trade_date"].dt.year
    trades = result["trades"]
    rotations = result["rotation_events"]
    previous_value = INITIAL_CAPITAL
    report: dict[str, Any] = {}
    for year in sorted(curve["year"].unique()):
        rows = curve[curve["year"] == year]
        end_value = float(rows["equity"].iloc[-1])
        values = np.concatenate(([previous_value], rows["equity"].astype(float).values))
        high_water = np.maximum.accumulate(values)
        max_drawdown = float(np.min(values / high_water - 1.0))
        year_trades = [row for row in trades if int(row["date"][:4]) == int(year)]
        year_rotations = [row for row in rotations if int(row["date"][:4]) == int(year)]
        holding_days = Counter(rows["holding"].tolist())
        top_holding, top_days = holding_days.most_common(1)[0]
        report[str(int(year))] = {
            "start_value": previous_value,
            "end_value": end_value,
            "return": end_value / previous_value - 1.0,
            "max_drawdown": max_drawdown,
            "trade_legs": len(year_trades),
            "early_rotations": len(year_rotations),
            "top_holding": top_holding,
            "top_holding_name": EXTENDED_POOL.get(top_holding, "货币ETF"),
            "top_holding_days": top_days,
        }
        previous_value = end_value
    return report


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "params": result["params"],
        "metrics": result["metrics"],
        "yearly": yearly_report(result),
        "rotation_events": result["rotation_events"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V3-G 8+8 expanded-pool test")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if args.fetch:
        fetch_peer_data()
    base_data, expanded_data = load_expanded_data()
    if ro.EXPO_REDUCE != 1.0 or ro.H3_EXPO_REDUCE != 1.0:
        raise RuntimeError("Server V3-G requires all downsize layers disabled")

    baseline_params = RotationParams(enabled=False)
    scheme_params = RotationParams(
        relative_gap=0.005,
        holding_drawdown=0.0,
        persistence_hits=1,
        fast_drawdown=None,
        minimum_hold_days=3,
        scope_groups=CLASS_GROUPS,
    )

    original = run_strategy(base_data, baseline_params)
    cache: dict[tuple[bool, float], dict[str, Any]] = {}
    with expanded_runtime():
        for multiplier in (1.0, 2.0, 3.0):
            cache[(False, multiplier)] = run_strategy(
                expanded_data, baseline_params, cost_multiplier=multiplier
            )
            cache[(True, multiplier)] = run_strategy(
                expanded_data, scheme_params, cost_multiplier=multiplier
            )

    extended_base = cache[(False, 1.0)]
    scheme = cache[(True, 1.0)]
    dates = raw_common_dates(expanded_data)
    raw_years = (dates[-1] - dates[0]).days / 365.25
    effective_years = (dates[-1] - dates[WARMUP]).days / 365.25

    print("\nasset peers")
    for original_code, peer in PEER_OF.items():
        original_name = ORIGINAL_POOL.get(original_code, "货币ETF")
        print(f"{original_code} {original_name:<10} -> {peer} {PEER_NAMES[peer]}")
    print(
        f"\ncommon data {dates[0]}..{dates[-1]} ({raw_years:.2f}y); "
        f"effective {dates[WARMUP]}..{dates[-1]} ({effective_years:.2f}y)"
    )

    print("\noverall")
    for label, result in (
        ("original server V3-G", original),
        ("expanded 8+8 V3-G", extended_base),
        ("expanded + class rotation", scheme),
    ):
        metrics = result["metrics"]
        print(
            f"{label:<28} final={metrics['final_value']:>11,.0f} "
            f"CAGR={metrics['cagr']:>6.1%} Sharpe={metrics['sharpe']:>5.2f} "
            f"MDD={metrics['max_drawdown']:>7.1%} trades={metrics['trade_legs']:>3} "
            f"early={metrics['early_rotations']:>2}"
        )

    print("\nyearly: expanded V3-G + class rotation")
    for year, row in yearly_report(scheme).items():
        print(
            f"{year}: end={row['end_value']:>11,.0f} ret={row['return']:>7.1%} "
            f"MDD={row['max_drawdown']:>7.1%} trades={row['trade_legs']:>3} "
            f"early={row['early_rotations']:>2} top={row['top_holding_name']}"
        )

    print("\ncost pressure")
    for multiplier in (1.0, 2.0, 3.0):
        base_metrics = cache[(False, multiplier)]["metrics"]
        scheme_metrics = cache[(True, multiplier)]["metrics"]
        print(
            f"{multiplier:.0f}x: expanded={base_metrics['final_value']:,.0f}/"
            f"{base_metrics['max_drawdown']:.1%} scheme="
            f"{scheme_metrics['final_value']:,.0f}/"
            f"{scheme_metrics['max_drawdown']:.1%}"
        )

    payload = {
        "meta": {
            "initial_capital": INITIAL_CAPITAL,
            "execution": "server V3-G T-day 14:50 close approximation",
            "downsize_layers": "disabled",
            "raw_common_start": str(dates[0]),
            "raw_common_end": str(dates[-1]),
            "raw_span_years": raw_years,
            "effective_start": str(dates[WARMUP]),
            "effective_span_years": effective_years,
            "peer_of": PEER_OF,
            "peer_names": PEER_NAMES,
            "class_groups": CLASS_GROUPS,
        },
        "original_server_v3g": compact(original),
        "expanded_v3g": compact(extended_base),
        "expanded_v3g_class_rotation": compact(scheme),
        "cost_pressure": {
            f"{multiplier:.0f}x": {
                "expanded_v3g": compact(cache[(False, multiplier)]),
                "expanded_v3g_class_rotation": compact(cache[(True, multiplier)]),
            }
            for multiplier in (1.0, 2.0, 3.0)
        },
    }
    if args.save:
        OUTPUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
        )
        print(f"\nsaved: {OUTPUT}")


if __name__ == "__main__":
    main()
