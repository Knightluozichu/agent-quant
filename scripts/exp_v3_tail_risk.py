"""V3 尾部风险量化分析 — 资产尾部特征 + 回撤段诊断 + 预警信号验证.

数据支撑:
  1. 各资产日收益分布: 偏度/峰度/VaR/ES/最大单日跌 (2019-12 ~ 2026-08)
  2. V3 同日口径回测 (14:50) 净值曲线 → 回撤段识别 + 恢复时间
  3. 深回撤段起点前信号: 持仓资产动量/波动率/距高点回撤/全池动量一致性
  4. 深回撤频率: >15% / >25% / >35% 段数、年化频率、时间占比
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_qixing_v3 import (
    DEFENSE,
    ETF_POOL,
    load_data,
    select_target,
)

OUTPUT = PROJECT_ROOT / "data" / "v9_results" / "v3_tail_analysis.json"

ASSET_NAMES = {
    "518880": "黄金ETF",
    "159985": "豆粕ETF",
    "501018": "南方原油",
    "161226": "白银LOF",
    "513100": "纳指ETF",
    "159915": "创业板ETF",
    "511220": "城投债ETF",
    "511880": "货币基金",
}
COMMODITY = ["161226", "501018", "159985", "518880"]  # 商品类
STOCKS = ["159915", "513100"]  # 股票类


def asset_tail_stats(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """各资产日收益尾部统计 (回测区间 2019-12 ~ 2026-08)."""
    rows = []
    for code in list(ETF_POOL) + [DEFENSE]:
        df = data[code]
        close = df["close"].astype(float).values
        ret = np.diff(close) / close[:-1]
        rows.append(
            {
                "code": code,
                "name": ASSET_NAMES[code],
                "n_days": len(ret),
                "ann_ret": (close[-1] / close[0]) ** (252 / len(ret)) - 1,
                "ann_vol": ret.std() * np.sqrt(252),
                "skew": float(pd.Series(ret).skew()),
                "kurt": float(pd.Series(ret).kurt()),  # 超额峰度
                "var95": float(np.percentile(ret, 5)),
                "es95": float(ret[ret <= np.percentile(ret, 5)].mean()),
                "var99": float(np.percentile(ret, 1)),
                "es99": float(ret[ret <= np.percentile(ret, 1)].mean()),
                "max_daily_drop": float(ret.min()),
                "drop_gt3pct": int((ret < -0.03).sum()),
                "drop_gt5pct": int((ret < -0.05).sum()),
                "max_consec_down": _max_consecutive_negative(ret),
            }
        )
    t = pd.DataFrame(rows).set_index("code")

    # 类别对比: 商品 vs 股票
    cat = {}
    for name, codes in [("商品(银油粕金)", COMMODITY), ("股票(创业板/纳指)", STOCKS)]:
        r = t.loc[
            codes,
            [
                "ann_vol",
                "skew",
                "kurt",
                "var99",
                "es99",
                "max_daily_drop",
                "drop_gt3pct",
                "drop_gt5pct",
            ],
        ].mean()
        cat[name] = r.to_dict()
    # 相关性 (日收益, 对齐公共日期)
    common_dates: set | None = None
    for code in ETF_POOL:
        dates = set(data[code]["trade_date"])
        common_dates = dates if common_dates is None else (common_dates & dates)
    common_dates = sorted(common_dates)
    rets = {}
    for code in ETF_POOL:
        df = data[code].set_index("trade_date").loc[common_dates, "close"].astype(float)
        rets[ASSET_NAMES[code]] = df.pct_change().values
    corr = pd.DataFrame(rets).corr()
    return t, pd.DataFrame(cat).T, corr


def _max_consecutive_negative(ret: np.ndarray) -> int:
    best = cur = 0
    for r in ret:
        cur = cur + 1 if r < 0 else 0
        best = max(best, cur)
    return best


def _idx_at(df: pd.DataFrame, td: str) -> int | None:
    mask = df["trade_date"] <= pd.Timestamp(td).date()
    return int(mask.sum()) - 1 if mask.sum() > 0 else None


def snapshot_features(data: dict, td: str) -> dict:
    """T日收盘的全部诊断特征 (用 <= T 的数据)."""
    feat: dict = {}
    per_asset = {}
    for code in ETF_POOL:
        if code not in data:
            continue
        idx = _idx_at(data[code], td)
        if idx is None or idx < 21:
            continue
        close = data[code]["close"].values[: idx + 1].astype(float)
        mom10 = (close[-1] - close[-11]) / close[-11]
        mom20 = (close[-1] - close[-21]) / close[-21]
        score = 0.5 * mom10 + 0.5 * mom20
        r = np.diff(close[-21:]) / close[-21:-1]
        vol20 = r.std() * np.sqrt(252)
        window = min(len(close), 252)
        hi = close[-window:].max()
        dd52w = close[-1] / hi - 1 if hi > 0 else 0.0
        per_asset[code] = {
            "mom10": mom10,
            "mom20": mom20,
            "score": score,
            "vol20": vol20,
            "dd_52w": dd52w,
        }
    feat["per_asset"] = per_asset

    # 全池动量一致性 (按 V3 过滤后的候选集合)
    etf_data_at_date = {
        c: _idx_at(data[c], td) for c in ETF_POOL if _idx_at(data[c], td) is not None
    }
    if len(etf_data_at_date) >= 5:
        _t, candidates, best_score, a_share_weak = select_target(data, etf_data_at_date, None)
        scores = sorted((s for _, s in candidates), reverse=True)
        feat["n_candidates"] = len(candidates)
        feat["best_score"] = best_score
        feat["top_margin"] = (scores[0] - scores[1]) if len(scores) >= 2 else np.nan
        feat["pool_mean"] = float(np.mean(scores)) if scores else 0.0
        feat["a_share_weak"] = bool(a_share_weak)
    return feat


def run_backtest(data: dict) -> pd.DataFrame:
    """同日口径回测 (对齐 14:50 执行), 返回每日净值+持仓."""
    from scripts.run_qixing_v3 import run_qixing_v3_same_day

    result = run_qixing_v3_same_day(data)
    eq = result["equity_curve"]
    eq["trade_date"] = pd.to_datetime(eq["trade_date"])
    return eq


def find_drawdowns(eq: pd.DataFrame, min_depth: float = 0.03) -> list[dict]:
    """基于净值曲线的回撤段识别 (dd<0 连续区间, 与 v3_tail_risk.json 口径对齐).

    段首 = 净值从峰值回落首日 (通常为调仓日, 即段首持仓=当日买入资产);
    段末 = 净值回到峰值日 (创新高). 恢复天数 = 谷底→段末 的交易日数.
    """
    equity = eq["equity"].values
    dates = eq["trade_date"].values
    holdings = eq["holding"].values
    cummax = np.maximum.accumulate(equity)
    dd = equity / cummax - 1
    in_dd = dd < 0
    segments: list[dict] = []
    i = 0
    n = len(equity)
    while i < n:
        if in_dd[i]:
            j = i
            while j < n and in_dd[j]:
                j += 1
            seg_dd = dd[i:j]
            trough = i + int(np.argmin(seg_dd))
            depth = float(seg_dd.min())
            if depth <= -min_depth:
                first_hold = str(holdings[i])
                trough_hold = str(holdings[trough])
                recovered = j < n  # 段末在数据内 => 已回到峰值
                segments.append(
                    {
                        "start": str(pd.Timestamp(dates[i]).date()),
                        "end": str(pd.Timestamp(dates[j - 1]).date()),
                        "trough": str(pd.Timestamp(dates[trough]).date()),
                        "depth": depth,
                        "days": j - i,
                        "recovery_days_from_trough": (j - 1 - trough) if recovered else None,
                        "recovered": recovered,
                        "holding": first_hold,  # 段首持仓 (买入时点)
                        "holding_name": ASSET_NAMES.get(first_hold, ""),
                        "trough_holding": trough_hold,  # 谷底持仓 (最深时点)
                        "trough_holding_name": ASSET_NAMES.get(trough_hold, ""),
                    }
                )
            i = j
        else:
            i += 1
    return segments


def enrich_segments(segments: list[dict], data: dict, eq: pd.DataFrame) -> list[dict]:
    """给回撤段附加起点前信号."""
    dates = eq["trade_date"]
    for seg in segments:
        start_dt = pd.Timestamp(seg["start"])
        # 段起点前最多5个交易日 (不含段内)
        prev = dates[dates < start_dt]
        if len(prev) == 0:
            seg["signal"] = None
            continue
        hist_dates = prev.tail(5)
        feats = [snapshot_features(data, str(d.date())) for d in hist_dates]
        valid = [f for f in feats if f.get("per_asset")]
        if not valid:
            seg["signal"] = None
            continue
        # 持仓资产特征 (段内主导持仓)
        code = seg["holding"]
        vals = {k: [] for k in ("mom10", "mom20", "score", "vol20", "dd_52w")}
        for f in valid:
            pa = f["per_asset"].get(code)
            if pa is None:
                continue
            for k in vals:
                vals[k].append(pa[k])
        sig = {
            "hold_mom10": float(np.mean(vals["mom10"])) if vals["mom10"] else None,
            "hold_mom20": float(np.mean(vals["mom20"])) if vals["mom20"] else None,
            "hold_slope": None,  # mom10 - mom20
            "hold_vol20": float(np.mean(vals["vol20"])) if vals["vol20"] else None,
            "hold_dd52w": float(np.mean(vals["dd_52w"])) if vals["dd_52w"] else None,
            "n_candidates": int(np.mean([f["n_candidates"] for f in valid])),
            "top_margin": float(np.nanmean([f["top_margin"] for f in valid])),
            "pool_mean": float(np.mean([f["pool_mean"] for f in valid])),
            "a_share_weak": bool(sum(f["a_share_weak"] for f in valid) > len(valid) / 2),
        }
        if sig["hold_mom10"] is not None and sig["hold_mom20"] is not None:
            sig["hold_slope"] = sig["hold_mom10"] - sig["hold_mom20"]
        seg["signal"] = sig
    return segments


def baseline_signals(data: dict, eq: pd.DataFrame) -> dict:
    """正常期基准: 非回撤 (equity>=0.995*cummax) 持仓日的信号分布."""
    equity = eq["equity"].values
    cummax = np.maximum.accumulate(equity)
    dates = eq["trade_date"]
    holdings = eq["holding"].values
    ok = equity >= 0.995 * cummax
    rows = []
    sampled = 0
    for i in np.where(ok)[0]:
        if sampled >= 400:
            break
        h = holdings[i]
        if h == DEFENSE or h not in ETF_POOL:
            continue
        f = snapshot_features(data, str(pd.Timestamp(dates[i]).date()))
        pa = f.get("per_asset", {}).get(h)
        if pa is None or not f.get("n_candidates"):
            continue
        rows.append(
            {
                "hold_mom10": pa["mom10"],
                "hold_mom20": pa["mom20"],
                "hold_slope": pa["mom10"] - pa["mom20"],
                "hold_vol20": pa["vol20"],
                "hold_dd52w": pa["dd_52w"],
                "n_candidates": f["n_candidates"],
                "top_margin": f["top_margin"],
                "pool_mean": f["pool_mean"],
            }
        )
        sampled += 1
    df = pd.DataFrame(rows)
    out = {}
    for c in df.columns:
        out[c] = {
            "mean": float(df[c].mean()),
            "median": float(df[c].median()),
            "p25": float(df[c].quantile(0.25)),
            "p75": float(df[c].quantile(0.75)),
        }
    return out


def main() -> None:
    data = load_data()
    print(f"数据加载: {len(data)} 只")

    # 1. 资产尾部统计
    stats, cat_cmp, corr = asset_tail_stats(data)
    print("\n=== 资产日收益尾部统计 ===")
    print(stats.round(4).to_string())
    print("\n=== 商品 vs 股票 ===")
    print(cat_cmp.round(4).to_string())
    print("\n=== 相关性 (日收益) ===")
    print(corr.round(2).to_string())

    # 2. 回测 + 回撤段
    eq = run_backtest(data)
    segs = find_drawdowns(eq, min_depth=0.03)
    segs.sort(key=lambda s: s["depth"])
    segs = enrich_segments(segs, data, eq)

    # 3. 深回撤频率
    n_years = (eq["trade_date"].iloc[-1] - eq["trade_date"].iloc[0]).days / 365.25
    dd_time = (eq["equity"] / eq["equity"].cummax() - 1).values
    freq = {}
    for thr in (0.15, 0.25, 0.35):
        segs_t = [s for s in segs if s["depth"] <= -thr]
        time_pct = float((dd_time <= -thr).mean())
        freq[f"gt{int(thr * 100)}"] = {
            "n_segments": len(segs_t),
            "annual_freq": len(segs_t) / n_years,
            "time_pct": time_pct,
            "depth_range": [
                float(min(s["depth"] for s in segs_t)),
                float(max(s["depth"] for s in segs_t)),
            ]
            if segs_t
            else None,
            "median_days": float(np.median([s["days"] for s in segs_t])) if segs_t else None,
            "recovery_ok": [s["recovery_days_from_trough"] for s in segs_t if s["recovered"]],
            "not_recovered": [s["start"] for s in segs_t if not s["recovered"]],
        }
        freq[f"gt{int(thr * 100)}"]["median_recovery_days"] = (
            float(np.median(freq[f"gt{int(thr * 100)}"]["recovery_ok"]))
            if freq[f"gt{int(thr * 100)}"]["recovery_ok"]
            else None
        )

    print("\n=== 深回撤频率 ===")
    print(json.dumps(freq, indent=2))

    # 4. 深回撤段明细 (深度>15%)
    deep = [s for s in segs if s["depth"] <= -0.15]
    print(f"\n=== 深回撤段明细 (深度>15%, 共{len(deep)}段) ===")
    for s in sorted(deep, key=lambda x: x["depth"]):
        sig = s["signal"] or {}
        rec = f"恢复{s['recovery_days_from_trough']}天" if s["recovered"] else "未恢复"
        print(
            f"{s['start']}~{s['end']} {s['days']}天 {s['depth']:+.1%} "
            f"买:{s['holding_name']} 谷底:{s['trough_holding_name']} "
            f"谷底日:{s['trough']} {rec}"
        )
        if sig:
            print(
                f"  段首前5日信号: mom10={sig.get('hold_mom10'):+.3f} mom20={sig.get('hold_mom20'):+.3f} "
                f"slope={sig.get('hold_slope'):+.3f} vol20={sig.get('hold_vol20'):.2f} "
                f"dd52w={sig.get('hold_dd52w'):+.1%} n_cand={sig.get('n_candidates')} "
                f"margin={sig.get('top_margin'):+.3f} pool={sig.get('pool_mean'):+.3f} "
                f"a弱={int(sig.get('a_share_weak', 0))}"
            )

    # 4b. 商品齐跌情境 (银油粕 20日收益同时为负的时间占比)
    print("\n=== 商品齐跌情境 (20日收益同负) ===")
    comm = ["161226", "501018", "159985"]
    common_dates: set | None = None
    for code in comm:
        dates = set(data[code]["trade_date"])
        common_dates = dates if common_dates is None else (common_dates & dates)
    cd = sorted(common_dates)
    r20 = {}
    for code in comm:
        s = data[code].set_index("trade_date").loc[cd, "close"].astype(float)
        r20[code] = s.pct_change(20).values
    r20df = pd.DataFrame(r20, index=cd)
    all_neg = (r20df < 0).all(axis=1)
    any_neg = (r20df < 0).any(axis=1)
    print(f"银油粕20日齐跌时间占比: {all_neg.mean():.1%} ({(cd[-1] - cd[0]).days / 365.25:.1f}年)")
    print(f"至少一个为负: {any_neg.mean():.1%}")
    # 策略持仓期间商品齐跌的暴露
    eq_by_date = eq.set_index("trade_date")["holding"]
    eq_by_date.index = pd.to_datetime(eq_by_date.index)
    r20df.index = pd.to_datetime(r20df.index)
    comm_neg = r20df[r20df < 0].all(axis=1)  # 银油粕20日齐跌日
    aligned = pd.concat([comm_neg, eq_by_date], axis=1, join="inner")
    aligned.columns = ["comm_all_neg", "holding"]
    expos = aligned[aligned["comm_all_neg"]]["holding"].value_counts(normalize=True)
    print("商品齐跌期间策略持仓分布:")
    for k, v in expos.items():
        print(f"  {ASSET_NAMES.get(k, k)}: {v:.1%}")

    # 5. 正常期基准对比
    base = baseline_signals(data, eq)
    print("\n=== 正常期基准信号 (非回撤持仓日) ===")
    print(json.dumps(base, indent=2))

    # 6. 回撤时间占比重验
    for thr in (0.05, 0.10, 0.20):
        print(f"回撤超{thr:.0%}时间占比: {float((dd_time <= -thr).mean()):.1%}")

    out = {
        "asset_stats": stats.round(6).to_dict("index"),
        "category_compare": cat_cmp.round(6).to_dict("index"),
        "correlation": corr.round(4).to_dict("index"),
        "drawdown_segments": segs,
        "deep_frequency": freq,
        "baseline_signals": base,
        "commodity_all_neg_20d": float(all_neg.mean()),
        "commodity_exposure_during_all_neg": expos.to_dict(),
        "n_years": n_years,
        "backtest": {
            "final": float(eq["equity"].iloc[-1]),
            "ann_return": float(
                (eq["equity"].iloc[-1] / eq["equity"].iloc[0])
                ** (365.25 / (eq["trade_date"].iloc[-1] - eq["trade_date"].iloc[0]).days)
                - 1
            ),
            "max_dd": float(dd_time.min()),
        },
    }
    OUTPUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n结果已保存: {OUTPUT}")


if __name__ == "__main__":
    main()
