"""被甩事件全景分析: 6.5年38次 DROP_FILTER 甩出持仓的正反数据.

目标: 对"38次被甩, 49%误杀(后5日反弹)"做多维度数据剖析:
  1. 误杀组 vs 真跌组 的特征对比 (资产/年份/暴跌幅度/回撤深度/趋势位置/动量/波动率)
  2. 各维度分桶的误杀率 (找"误杀高发区"与"真跌高发区")
  3. 机会成本: 被甩后换入品种 vs 原品种 的后续收益差
  4. 恢复时间: 误杀品种多久回到被甩价
  5. 市场环境: 牛熊/同池强弱下的误杀率差异

输出: 控制台专家团分析 + data/v9_results/drop_analysis.json
用法: uv run python scripts/exp_drop_analysis.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

import run_qixing_v3 as rq  # noqa: E402
import exp_drop_gate_h3 as h3  # noqa: E402

OUTPUT_DIR = Path(rq.PROJECT_ROOT) / "data" / "v9_results"
WARMUP = 130
HORIZONS = (5, 10, 20, 30)


def collect_events(data: dict) -> list[dict]:
    """收集全部被甩事件 (基线持仓触发暴跌过滤被换走) + 完整特征."""
    log: list[tuple] = []
    ORIG_SELECT = rq.select_target

    def wrapped(data_, etf_data_at_date, holding):
        t, c, s, a = ORIG_SELECT(data_, etf_data_at_date, holding)
        td = None
        for code in rq.ETF_POOL:
            if code in etf_data_at_date:
                td = data_[code].iloc[etf_data_at_date[code]]["trade_date"]
                break
        log.append((td, holding, t))
        return t, c, s, a

    rq.check_single_day_drop = h3.ORIG_CHECK
    rq.select_target = wrapped
    h3.select_target = wrapped
    h3.H3_ENABLED = False
    h3.run_v3_risk_h3(data)
    rq.check_single_day_drop = h3.ORIG_CHECK
    rq.select_target = ORIG_SELECT
    h3.select_target = ORIG_SELECT
    h3.H3_ENABLED = False

    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    ci = {d: i for i, d in enumerate(dates)}

    events = []
    for td, holding, target in log:
        if target == holding or not holding or holding == rq.DEFENSE:
            continue
        df = data[holding]
        mask = df["trade_date"] <= td
        if mask.sum() < WARMUP:
            continue
        idx = mask.sum() - 1
        close = df["close"].values[: idx + 1].astype(float)
        drop = any(
            (close[i] - close[i - 1]) / close[i - 1] < rq.DROP_THRESHOLD
            for i in range(-rq.DROP_LOOKBACK, 0)
        )
        if not drop:
            continue

        # 特征
        day_drop = float((close[-1] - close[-2]) / close[-2])
        ret60 = (close[-1] - close[-61]) / close[-61] if len(close) > 61 else 0.0
        r10 = (close[-1] - close[-11]) / close[-11]
        r20 = (close[-1] - close[-21]) / close[-21]
        mom = 0.5 * r10 + 0.5 * r20
        peak60 = float(np.max(close[-61:-1]))
        dd_from_peak = close[-1] / peak60 - 1.0 if peak60 > 0 else 0.0
        dr = np.diff(close[-21:]) / close[-21:-1]
        vol20 = float(np.std(dr) * np.sqrt(252))

        # 后向收益 (多窗口)
        pos = ci.get(td)
        fwd = {}
        for hz in HORIZONS:
            if pos is not None and pos + hz < len(dates):
                fut = dates[pos + hz]
                frow = df[df["trade_date"] == fut]
                if not frow.empty:
                    fwd[hz] = float(frow.iloc[0]["close"]) / close[-1] - 1.0

        # 恢复时间: 首次回到被甩价 (收盘) 的交易日数
        recover = None
        if pos is not None:
            for k in range(1, min(31, len(dates) - pos)):
                fut = dates[pos + k]
                frow = df[df["trade_date"] == fut]
                if not frow.empty and float(frow.iloc[0]["close"]) >= close[-1]:
                    recover = k
                    break

        # 机会成本: 换入品种 target 后5/10日收益
        to_fwd = {}
        if target in data:
            tdf = data[target]
            tmask = tdf["trade_date"] <= td
            if tmask.sum() > 0:
                t_idx = tmask.sum() - 1
                t_close = float(tdf["close"].values[t_idx])
                for hz in (5, 10):
                    if pos is not None and pos + hz < len(dates):
                        fut = dates[pos + hz]
                        trow = tdf[tdf["trade_date"] == fut]
                        if not trow.empty:
                            to_fwd[hz] = float(trow.iloc[0]["close"]) / t_close - 1.0

        # 市场环境: 同池其他品种平均动量 (被甩日)
        others = []
        for c in rq.ETF_POOL:
            if c == holding or c not in data:
                continue
            cm = data[c]
            cmask = cm["trade_date"] <= td
            if cmask.sum() > 30:
                cclose = cm["close"].values[: cmask.sum()].astype(float)
                cr10 = (cclose[-1] - cclose[-11]) / cclose[-11]
                cr20 = (cclose[-1] - cclose[-21]) / cclose[-21]
                others.append(0.5 * cr10 + 0.5 * cr20)
        market_mom = float(np.mean(others)) if others else 0.0

        events.append(
            {
                "date": str(td),
                "code": holding,
                "name": rq.ETF_POOL[holding],
                "to": target,
                "day_drop": round(float(day_drop), 4),
                "ret60": round(float(ret60), 4),
                "mom": round(float(mom), 4),
                "dd_peak": round(float(dd_from_peak), 4),
                "vol20": round(float(vol20), 3),
                "market_mom": round(float(market_mom), 4),
                "fwd": {str(k): round(v, 4) for k, v in fwd.items()},
                "recover_days": recover,
                "to_fwd": {str(k): round(v, 4) for k, v in to_fwd.items()},
            }
        )
    return events


def bucket_report(events: list[dict], key: str, buckets: list[tuple], label: str) -> None:
    """按桶输出误杀率 (5日口径)."""
    print(f"  【{label}】")
    for lo, hi, name in buckets:
        grp = [e for e in events if lo <= e[key] < hi]
        if len(grp) < 2:
            continue
        f5 = [e["fwd"]["5"] for e in grp if "5" in e["fwd"]]
        if not f5:
            continue
        fake = sum(1 for x in f5 if x > 0)
        avg = float(np.mean(f5))
        print(
            f"    {name:<22} {len(grp):>2}次 误杀 {fake}/{len(f5)} = {fake / len(f5) * 100:>3.0f}% "
            f"后5日均 {avg:+.2%}"
        )


def main() -> None:
    print("=" * 80)
    print("  被甩事件全景分析 | 38次 DROP_FILTER 甩出持仓的正反数据")
    print("=" * 80)

    data = rq.load_data()
    events = collect_events(data)
    total = len(events)
    fake5 = [e for e in events if "5" in e["fwd"] and e["fwd"]["5"] > 0]
    real5 = [e for e in events if "5" in e["fwd"] and e["fwd"]["5"] <= 0]
    print(f"\n  被甩事件: {total}次 | 5日误杀 {len(fake5)} / 真跌 {len(real5)}")

    # === 1. 资产 × 正反 ===
    print("\n" + "=" * 80)
    print("  ① 资产维度: 哪些资产的被甩是误杀, 哪些是真跌")
    for code in rq.ETF_POOL:
        grp = [e for e in events if e["code"] == code]
        if not grp:
            continue
        f5 = [e["fwd"]["5"] for e in grp if "5" in e["fwd"]]
        if not f5:
            continue
        fake = sum(1 for x in f5 if x > 0)
        print(
            f"    {rq.ETF_POOL[code]:<8} {len(grp):>2}次 误杀 {fake}/{len(f5)} "
            f"({fake / len(f5) * 100:.0f}%) 后5日均 {np.mean(f5):+.2%} "
            f"后20日均 {np.mean([e['fwd'].get('20', np.nan) for e in grp if '20' in e['fwd']] or [np.nan]):+.2%}"
        )

    # === 2. 年度 × 正反 ===
    print("\n  【年度维度】")
    for y in sorted(set(e["date"][:4] for e in events)):
        grp = [e for e in events if e["date"][:4] == y]
        f5 = [e["fwd"]["5"] for e in grp if "5" in e["fwd"]]
        if not f5:
            continue
        fake = sum(1 for x in f5 if x > 0)
        print(
            f"    {y}: {len(grp)}次 误杀 {fake}/{len(f5)} ({fake / len(f5) * 100:.0f}%) "
            f"后5日均 {np.mean(f5):+.2%}"
        )

    # === 3. 特征分桶 ===
    print("\n  【特征分桶 (误杀率)】")
    bucket_report(
        events,
        "day_drop",
        [(-0.15, -0.05, "单日跌5%~15%"), (-0.049, -0.03, "单日跌3%~5%")],
        "暴跌幅度",
    )
    bucket_report(
        events,
        "ret60",
        [(-1, 0.1, "ret60<10% (平淡)"), (0.1, 0.3, "ret60 10%~30%"), (0.3, 10, "ret60>30% (大涨)")],
        "趋势位置",
    )
    bucket_report(
        events,
        "dd_peak",
        [(-1, -0.1, "距60日峰值回撤>10%"), (-0.099, 0, "距峰值回撤<10%")],
        "回撤深度",
    )
    bucket_report(
        events,
        "vol20",
        [(0, 0.35, "低波动<35%"), (0.35, 0.6, "中波动35~60%"), (0.6, 10, "高波动>60%")],
        "20日波动率",
    )
    bucket_report(
        events, "mom", [(-1, 0, "动量<0"), (0, 0.1, "动量0~10%"), (0.1, 10, "动量>10%")], "生产动量"
    )

    # === 4. 机会成本 ===
    print("\n  【机会成本: 换入品种 vs 原品种 (5日)】")
    opp = []
    for e in events:
        if "5" in e["fwd"] and "5" in e["to_fwd"]:
            opp.append((e, e["to_fwd"]["5"] - e["fwd"]["5"]))
    if opp:
        arr = np.array([x[1] for x in opp])
        print(
            f"    换仓净差 (换入-原品种): 平均 {arr.mean():+.2%} | "
            f"误杀组: {np.mean([x[1] for x in opp if x[0]['fwd']['5'] > 0]):+.2%} | "
            f"真跌组: {np.mean([x[1] for x in opp if x[0]['fwd']['5'] <= 0]):+.2%}"
        )
        better = sum(1 for x in opp if x[1] > 0)
        print(f"    换仓比持有原品种好: {better}/{len(opp)} = {better / len(opp) * 100:.0f}%")

    # === 5. 恢复时间 ===
    print("\n  【误杀组的恢复时间 (回到被甩价)】")
    rec = [e["recover_days"] for e in fake5 if e["recover_days"]]
    nrec = [e for e in fake5 if not e["recover_days"]]
    if rec:
        print(
            f"    恢复: {len(rec)}/{len(fake5)} 次, 中位数 {int(np.median(rec))} 交易日, "
            f"平均 {np.mean(rec):.1f} 交易日"
        )
    print(f"    30日内未恢复: {len(nrec)} 次")

    # === 6. 市场环境 ===
    print("\n  【市场环境: 同池其他品种平均动量】")
    bucket_report(
        events,
        "market_mom",
        [
            (-1, -0.05, "同池弱 (均值<-5%)"),
            (-0.05, 0.05, "同池中性"),
            (0.05, 10, "同池强 (均值>5%)"),
        ],
        "市场强弱",
    )

    out = {"n_events": total, "n_fake5": len(fake5), "n_real5": len(real5), "events": events}
    path = OUTPUT_DIR / "drop_analysis.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n  ✓ 结果已保存: {path}")


if __name__ == "__main__":
    main()
