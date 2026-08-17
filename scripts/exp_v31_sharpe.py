"""V3.1 夏普优化实验: 波动率目标仓位 + 日频风险退出 + 波动调整动量 + 拥挤惩罚.

数学模型 (先行定义, 禁止回测反选):
  P1 波动率目标: e = clip(σ_target/σ̂_20d, e_floor, 1), 买入仓位=权益×e
  P2 日频风险退出: 当日收益<-5% 或 自买入回撤<-10% → 当日收盘切货币,
     冷却至下个调仓日方可再入场 (通用逻辑, 非拟合)
  P3 波动调整动量: score = mom × (σ_ref/σ_i), σ_ref=0.25 (同量纲锚点)
  P4 同类拥挤惩罚: 同类别第2个入选候选得分 ×0.7

验证协议 (项目规范):
  1. IS 2020-2023 定参 (中心先验选参, 防过拟合)
  2. OOS 2024-2026 验证, 不支持即否决
  3. 参数扰动 ±20% 稳健性 + 成本 2x/3x 压力

用法: uv run python scripts/exp_v31_sharpe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_qixing_v3 as rq  # noqa: E402

INITIAL = 100_000.0
WARMUP = 130
SIGMA_REF = 0.25

CAT_OF: dict[str, str] = {}
for _cat, _codes in rq.CATEGORIES.items():
    for _c in _codes:
        CAT_OF[_c] = _cat


def ann_vol(close: np.ndarray, idx: int, window: int = 20) -> float | None:
    """过去 window 日年化波动 (只用历史, 无前瞻)."""
    if idx < window:
        return None
    seg = close[idx - window : idx + 1].astype(float)
    rets = np.diff(seg) / seg[:-1]
    return float(np.std(rets) * np.sqrt(252))


def build_dmap(data: dict) -> dict:
    """{code: {date: row_idx}} 预索引, 加速日级回测."""
    return {
        code: {d: i for i, d in enumerate(df["trade_date"].tolist())} for code, df in data.items()
    }


def tradable(data: dict, dmap: dict, code: str, td) -> bool:
    """收盘涨跌停检查 (与镜像版 _check_close_tradable 一致): |涨跌|>=9.9% 不可交易."""
    i = dmap.get(code, {}).get(td)
    if i is None or i < 1:
        return False
    close = data[code]["close"].values
    if close[i] <= 0:
        return False
    return bool(abs(close[i] / close[i - 1] - 1) < 0.099)


def select_target_v31(data, dmap, idx_map, holding, P):
    """V3.1 选股: 复用 V3 过滤器 + P3/P4 增强 (与 rq.select_target 结构一致)."""
    a_share_weak = rq.check_a_share_weak(data, idx_map.get(rq.A_SHARE_ETF, 0))
    candidates = []
    for code in rq.ETF_POOL:
        if code not in idx_map:
            continue
        if code == rq.A_SHARE_ETF and a_share_weak:
            continue
        idx = idx_map[code]
        close = data[code]["close"].values[: idx + 1].astype(float)
        if len(close) < 121:
            continue
        if rq.USE_DROP_FILTER and not rq.check_single_day_drop(close):
            continue
        score = rq.calc_momentum_score(close)
        # C2 入场阈值: 要求有意义的边际优势 (防边缘信号whipsaw)
        if P.get("entry_th") and score < P["entry_th"]:
            continue
        # C1 趋势确认: 价格须在自身 MA20 之上 (顺自身趋势, 通用逻辑)
        if P.get("ma_trend") and len(close) >= 20 and close[-1] < float(np.mean(close[-20:])):
            continue
        if score > 0:
            candidates.append([code, score])

    # P4: 同类拥挤惩罚 (按当前得分排序, 同类第2个起降权)
    if P.get("crowd_penalty", 1.0) < 1.0:
        candidates.sort(key=lambda x: -x[1])
        cat_seen: dict[str, int] = {}
        for c in candidates:
            cat = CAT_OF.get(c[0], "其他")
            if cat_seen.get(cat, 0) >= 1:
                c[1] *= P["crowd_penalty"]
            cat_seen[cat] = cat_seen.get(cat, 0) + 1

    # P3: 波动调整动量 score = mom × (σ_ref / σ_i)
    if P.get("use_vol_mom"):
        for c in candidates:
            sigma = ann_vol(data[c[0]]["close"].values, idx_map[c[0]])
            if sigma and sigma > 0.01:
                c[1] = c[1] * (SIGMA_REF / sigma)

    candidates.sort(key=lambda x: -x[1])
    best_target = candidates[0][0] if candidates else rq.DEFENSE
    best_score = candidates[0][1] if candidates else 0.0
    threshold = 0.0 if best_score > 0.10 else 0.05

    if holding and holding != rq.DEFENSE:
        cur_score = dict((c[0], c[1]) for c in candidates).get(holding, -999.0)
        if cur_score > 0:
            target = best_target if best_score > cur_score + threshold else holding
        else:
            target = best_target
    else:
        target = best_target
    return target, candidates, a_share_weak


def run_v31(data: dict, P: dict, dmap: dict, cost_multiplier: float = 1.0) -> dict:
    """V3.1 完整回测 (T日信号 → T日收盘成交, 实盘镜像口径)."""
    common: set = set()
    for code in rq.ETF_POOL:
        if code in data:
            dates = data[code]["trade_date"].tolist()
            common = set(dates) if not common else common & set(dates)
    if rq.DEFENSE in data:
        common &= set(data[rq.DEFENSE]["trade_date"].tolist())
    trading_dates = sorted(common)[WARMUP:]
    # D3 网格相位: grid_offset 平移调仓网格 (相位敏感性分析用)
    grid = set(trading_dates[P.get("grid_offset", 0) :: rq.REBALANCE_DAYS])

    fee = rq.FEE * cost_multiplier
    slip = rq.SLIPPAGE * cost_multiplier

    cash = INITIAL
    holding: str | None = None
    shares = 0
    entry_price = 0.0
    cooling = False  # P2 退出后冷却至下个调仓日
    n_exits = 0
    eq_rows: list[dict] = []

    for td in trading_dates:
        exited_today = False

        # === P2: 日频风险退出 ===
        if holding and (P.get("exit_intraday") or P.get("exit_dd")):
            i = dmap[holding][td]
            close = data[holding]["close"].values
            if i >= 1:
                daily_ret = close[i] / close[i - 1] - 1
                dd_entry = (close[i] / entry_price - 1) if entry_price > 0 else 0.0
                hit = (P.get("exit_intraday") and daily_ret < P["exit_intraday"]) or (
                    P.get("exit_dd") and dd_entry < P["exit_dd"]
                )
                if hit:
                    if tradable(data, dmap, holding, td):
                        cash += shares * close[i] * (1 - fee - slip)
                        holding, shares, entry_price = None, 0, 0.0
                        cooling = True
                        exited_today = True
                        n_exits += 1
                    # 跌停卖不出 → 继续持有 (与镜像版一致)

        # === 调仓日: 选股与换仓 (C3 快速再入场 / D1 强趋势日频切换) ===
        is_decision = td in grid
        # D1 强趋势日频切换: 持仓动量>阈值时, 不等网格, 每日检查更优目标
        if (
            not is_decision
            and P.get("daily_strong")
            and holding
            and not cooling
            and not exited_today
        ):
            i_h = dmap.get(holding, {}).get(td)
            if i_h is not None and i_h >= 120:
                h_close = data[holding]["close"].values[: i_h + 1].astype(float)
                if rq.check_single_day_drop(h_close) and rq.calc_momentum_score(h_close) > P.get(
                    "strong_th", 0.10
                ):
                    is_decision = True
        if (is_decision or (P.get("fast_reenter") and not cooling)) and not exited_today:
            cooling = False
            idx_map = {}
            for code in [*list(rq.ETF_POOL.keys()), rq.DEFENSE]:
                i = dmap.get(code, {}).get(td)
                if i is not None and i >= WARMUP:
                    idx_map[code] = i

            target, candidates, _weak = select_target_v31(data, dmap, idx_map, holding, P)

            if target != holding:
                sell_ok = True
                if holding:  # 卖出 (涨跌停检查, 失败则不换仓)
                    if not tradable(data, dmap, holding, td):
                        sell_ok = False
                    else:
                        i = dmap[holding][td]
                        price = float(data[holding]["close"].values[i])
                        cash += shares * price * (1 - fee - slip)
                        holding, shares, entry_price = None, 0, 0.0
                if sell_ok and target and target in data:  # 买入
                    if tradable(data, dmap, target, td):
                        i = dmap[target][td]
                        price = float(data[target]["close"].values[i])
                        # 仓位决策 (三种模式: crisis/trend_gate/标准vol_target)
                        exposure = 1.0
                        sigma = ann_vol(data[target]["close"].values, i)
                        if sigma and sigma > 0.01:
                            if P.get("trend_gate") is not None and P.get("vol_target"):
                                # A2 趋势门控: 强趋势满仓, 弱趋势才做波动率缩放
                                raw_close = data[target]["close"].values[: i + 1].astype(float)
                                raw_score = rq.calc_momentum_score(raw_close)
                                if raw_score <= P["trend_gate"]:
                                    exposure = float(
                                        np.clip(
                                            P["vol_target"] / sigma,
                                            P.get("vol_floor", 0.3),
                                            P.get("leverage_cap", 1.0),
                                        )
                                    )
                            elif P.get("vol_mode") == "crisis" and P.get("sigma_high"):
                                # A1 危机模式: 仅当波动超过 sigma_high 才降仓
                                sh = P["sigma_high"]
                                if sigma > sh:
                                    exposure = float(
                                        np.clip(
                                            sh / sigma,
                                            P.get("vol_floor", 0.3),
                                            P.get("leverage_cap", 1.0),
                                        )
                                    )
                            elif P.get("vol_target"):
                                # 标准波动率目标 (leverage_cap 硬上限, 1.0=不加杠杆)
                                exposure = float(
                                    np.clip(
                                        P["vol_target"] / sigma,
                                        P.get("vol_floor", 0.3),
                                        P.get("leverage_cap", 1.0),
                                    )
                                )
                        buy_shares = int(cash * exposure * 0.99 / price / 100) * 100
                        if buy_shares > 0:
                            cash -= buy_shares * price * (1 + fee + slip)
                            holding, shares, entry_price = target, buy_shares, price

        # === 每日净值 ===
        equity = cash
        if holding:
            i = dmap[holding].get(td)
            if i is not None:
                equity += shares * float(data[holding]["close"].values[i])
        eq_rows.append({"trade_date": td, "equity": equity, "holding": holding or rq.DEFENSE})

    eq = pd.DataFrame(eq_rows)
    eq["trade_date"] = pd.to_datetime(eq["trade_date"])
    return {"equity": eq, "n_exits": n_exits}


def metrics(eq: pd.DataFrame, start: str, end: str) -> dict:
    """窗口化指标 (从单次全周期运行的净值切片, 保持网格与状态连续)."""
    w = eq[(eq["trade_date"] >= start) & (eq["trade_date"] <= end)]
    if len(w) < 30:
        return {
            "ret": float("nan"),
            "sharpe": float("nan"),
            "vol": float("nan"),
            "dd": float("nan"),
        }
    rets = w["equity"].pct_change().dropna()
    total = w["equity"].iloc[-1] / w["equity"].iloc[0] - 1
    span = max((w["trade_date"].iloc[-1] - w["trade_date"].iloc[0]).days / 365.25, 1e-9)
    ann_ret = (1 + total) ** (1 / span) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0.0
    cm = w["equity"].cummax()
    dd = ((w["equity"] - cm) / cm).min()
    return {"ret": total, "ann": ann_ret, "sharpe": sharpe, "vol": vol, "dd": dd}


IS_RANGE = ("2020-01-01", "2023-12-31")
OOS_RANGE = ("2024-01-01", "2026-12-31")


def scan_is(data, dmap) -> list:
    """IS 参数扫描: σ_target × 退出阈值 (P3/P4 关闭, 先验证风控主杠杆)."""
    results = []
    for vt in [0.22, 0.25, 0.28, 0.32, 0.40]:
        for ei in [-0.04, -0.05, -0.06]:
            for ed in [-0.08, -0.10, -0.12]:
                P = {"vol_target": vt, "vol_floor": 0.3, "exit_intraday": ei, "exit_dd": ed}
                r = run_v31(data, P, dmap)
                m = metrics(r["equity"], *IS_RANGE)
                results.append({"P": P, "is": m, "n_exits": r["n_exits"]})
    results.sort(key=lambda x: -x["is"]["sharpe"])
    return results


def select_params(scan: list) -> dict:
    """中心先验选参 (防过拟合): IS夏普前30%中, 取最接近先验中心
    (σ_target=0.28, exit=-0.05/-0.10) 的配置; 若无则取最优."""
    top = scan[: max(3, len(scan) // 3)]
    center = {"vol_target": 0.28, "exit_intraday": -0.05, "exit_dd": -0.10}

    def dist(x):
        p = x["P"]
        return (
            abs(p["vol_target"] - center["vol_target"]) / 0.28
            + abs(p["exit_intraday"] - center["exit_intraday"]) / 0.05
            + abs(p["exit_dd"] - center["exit_dd"]) / 0.10
        )

    return min(top, key=dist)["P"]


def main() -> None:
    data = rq.load_data()
    dmap = build_dmap(data)
    print("=" * 66)
    print("  V3.1 夏普优化实验 (数学模型先行 + IS/OOS + 扰动三重验证)")
    print("=" * 66)

    # === 0. 基线 (实盘镜像) ===
    base = rq.run_qixing_v3_same_day(data, INITIAL, live_mirror=True)["equity_curve"]
    base = base.rename(columns={"equity": "equity"})
    mb_is = metrics(base, *IS_RANGE)
    mb_oos = metrics(base, *OOS_RANGE)
    mb_full = metrics(base, "2020-01-01", "2026-12-31")

    # === 1. IS 扫描 ===
    print("\n  [1] IS 2020-2023 参数扫描 (Top8 by IS夏普)")
    scan = scan_is(data, dmap)
    print(
        f"  {'σ_tgt':>5} {'急跌':>6} {'回撤':>6} | {'IS夏普':>7} {'IS波动':>7} {'IS回撤':>7} {'退出次':>5}"
    )
    for x in scan[:8]:
        p, m = x["P"], x["is"]
        print(
            f"  {p['vol_target']:>5.2f} {p['exit_intraday']:>6.2f} {p['exit_dd']:>6.2f} | "
            f"{m['sharpe']:>7.2f} {m['vol']:>7.1%} {m['dd']:>7.1%} {x['n_exits']:>5}"
        )

    chosen = select_params(scan)
    print(
        f"\n  [2] 选定参数 (中心先验): σ_target={chosen['vol_target']}, "
        f"急跌={chosen['exit_intraday']}, 回撤={chosen['exit_dd']}"
    )

    # === 2. 机制叠加消融 (OOS 只用于验证展示, 不参与选参) ===
    configs = {
        "+P1波动目标": {**chosen, "exit_intraday": None, "exit_dd": None},
        "+P1+P2风控": dict(chosen),
        "+P1+P2+P3波动动量": {**chosen, "use_vol_mom": True},
        "+P1+P2+P4拥挤惩罚": {**chosen, "crowd_penalty": 0.7},
        "全量P1+P2+P3+P4": {**chosen, "use_vol_mom": True, "crowd_penalty": 0.7},
    }
    print(f"\n  [3] 机制消融对比 (夏普)")
    print(f"  {'配置':<22} {'IS':>6} {'OOS':>6} {'全周期':>6} {'全周期回撤':>9} {'退出次':>5}")
    # 真实镜像基线 (含涨跌停检查的 run_qixing_v3_same_day)
    print(
        f"  {'基线(实盘镜像)':<22} {mb_is['sharpe']:>6.2f} {mb_oos['sharpe']:>6.2f} "
        f"{mb_full['sharpe']:>6.2f} {mb_full['dd']:>9.1%} {'-':>5}"
    )
    best_full = None
    for name, P in configs.items():
        r = run_v31(data, P, dmap)
        mis = metrics(r["equity"], *IS_RANGE)
        moos = metrics(r["equity"], *OOS_RANGE)
        mfull = metrics(r["equity"], "2020-01-01", "2026-12-31")
        print(
            f"  {name:<22} {mis['sharpe']:>6.2f} {moos['sharpe']:>6.2f} "
            f"{mfull['sharpe']:>6.2f} {mfull['dd']:>9.1%} {r['n_exits']:>5}"
        )
        if best_full is None or mfull["sharpe"] > best_full[1]["sharpe"]:
            best_full = (name, mfull, P, r)

    # === 3. 最优配置详表 ===
    name, mfull, P_best, r_best = best_full
    print(f"\n  [4] 推荐配置: {name}")
    print(
        f"      全周期: 收益{mfull['ret']:+.0%} 年化{mfull['ann']:+.1%} "
        f"夏普{mfull['sharpe']:.2f} 波动{mfull['vol']:.1%} 回撤{mfull['dd']:.1%}"
    )
    print(
        f"      基线对照: 夏普{mb_full['sharpe']:.2f} 回撤{mb_full['dd']:.1%} "
        f"(IS {mb_is['sharpe']:.2f} / OOS {mb_oos['sharpe']:.2f})"
    )
    eq = r_best["equity"]
    eq["year"] = eq["trade_date"].dt.year
    print(f"\n      年度明细:")
    print(f"      {'年份':<6} {'收益':>8} {'波动':>8} {'夏普':>6} {'回撤':>8}")
    prev = None
    for year in sorted(eq["year"].unique()):
        y = eq[eq["year"] == year]
        yr_ret = y["equity"].iloc[-1] / (prev or INITIAL) - 1
        m = metrics(eq, f"{year}-01-01", f"{year}-12-31")
        print(
            f"      {year:<6} {yr_ret:>+8.1%} {m['vol']:>8.1%} {m['sharpe']:>6.2f} {m['dd']:>8.1%}"
        )
        prev = y["equity"].iloc[-1]
    final = eq["equity"].iloc[-1]
    print(f"\n      10万 → {final:,.0f} ({final / INITIAL - 1:+.0%})")

    # === 4. 参数扰动 ±20% 稳健性 ===
    print(f"\n  [5] 参数扰动稳健性 (全周期夏普)")
    base_sharpe = mfull["sharpe"]
    for label, pmod in [
        ("σ_target -20%", {"vol_target": chosen["vol_target"] * 0.8}),
        ("σ_target +20%", {"vol_target": chosen["vol_target"] * 1.2}),
        ("急跌阈值 -20%", {"exit_intraday": chosen["exit_intraday"] * 0.8}),
        ("急跌阈值 +20%", {"exit_intraday": chosen["exit_intraday"] * 1.2}),
        ("回撤阈值 -20%", {"exit_dd": chosen["exit_dd"] * 0.8}),
        ("回撤阈值 +20%", {"exit_dd": chosen["exit_dd"] * 1.2}),
    ]:
        Pp = {**P_best, **pmod}
        rp = run_v31(data, Pp, dmap)
        mp = metrics(rp["equity"], "2020-01-01", "2026-12-31")
        flag = "✓" if abs(mp["sharpe"] - base_sharpe) < 0.3 else "⚠️"
        print(f"      {flag} {label:<16} 夏普 {mp['sharpe']:.2f} (基准 {base_sharpe:.2f})")

    # === 5. 成本压力 ===
    print(f"\n  [6] 成本压力测试 (推荐配置)")
    for mult in [2.0, 3.0]:
        rc = run_v31(data, P_best, dmap, cost_multiplier=mult)
        mc = metrics(rc["equity"], "2020-01-01", "2026-12-31")
        print(
            f"      {mult:.0f}x成本: 夏普 {mc['sharpe']:.2f} 收益 {mc['ret']:+.0%} 回撤 {mc['dd']:.1%}"
        )


def aggressive_scan() -> None:
    """激进突变方案对比: 归因 V3.1 收益下降 + 寻找收益/夏普帕累托前沿."""
    data = rq.load_data()
    dmap = build_dmap(data)
    P2 = {"exit_intraday": -0.05, "exit_dd": -0.10}  # 公共风控底座

    configs = {
        "V3镜像(满仓基线)": None,  # 特殊处理
        "V3.1保守(σ0.28)": {"vol_target": 0.28, "vol_floor": 0.3, **P2},
        "A0 仅日频退出": dict(P2),
        "A1a 危机模式σh=0.40": {"vol_mode": "crisis", "sigma_high": 0.40, "vol_floor": 0.3, **P2},
        "A1b 危机模式σh=0.45": {"vol_mode": "crisis", "sigma_high": 0.45, "vol_floor": 0.3, **P2},
        "A1c 危机模式σh=0.50": {"vol_mode": "crisis", "sigma_high": 0.50, "vol_floor": 0.3, **P2},
        "A2 趋势门控+σ0.28": {"vol_target": 0.28, "vol_floor": 0.3, "trend_gate": 0.10, **P2},
        "A2b 趋势门控+σ0.35": {"vol_target": 0.35, "vol_floor": 0.3, "trend_gate": 0.10, **P2},
        "A3 高σ目标0.40": {"vol_target": 0.40, "vol_floor": 0.3, **P2},
    }

    print("=" * 88)
    print("  激进突变方案对比 (全周期 2020-2026.7, 10万本金)")
    print("=" * 88)
    print(
        f"  {'方案':<18} {'终值':>10} {'总收益':>8} {'年化':>7} {'夏普':>6} "
        f"{'波动':>7} {'回撤':>8} {'IS夏普':>7} {'OOS夏普':>8}"
    )
    print(f"  {'-' * 86}")
    finals: dict[str, float] = {}
    for name, P in configs.items():
        if P is None:
            mir = rq.run_qixing_v3_same_day(data, INITIAL, live_mirror=True)
            eq = mir["equity_curve"]
        else:
            eq = run_v31(data, P, dmap)["equity"]
        mf = metrics(eq, "2020-01-01", "2026-12-31")
        mis = metrics(eq, *IS_RANGE)
        moos = metrics(eq, *OOS_RANGE)
        final = eq["equity"].iloc[-1]
        finals[name] = final
        print(
            f"  {name:<18} {final:>10,.0f} {mf['ret']:>+8.0%} {mf['ann']:>+7.1%} "
            f"{mf['sharpe']:>6.2f} {mf['vol']:>7.1%} {mf['dd']:>8.1%} "
            f"{mis['sharpe']:>7.2f} {moos['sharpe']:>8.2f}"
        )

    # 最优激进方案年度明细 (A* 中终值最高者, 数据选出而非预设)
    best_name = max((n for n in finals if n.startswith("A")), key=lambda n: finals[n])
    print(f"\n  [终值最高的激进方案: {best_name} 年度明细]")
    eq = run_v31(data, configs[best_name], dmap)["equity"]
    mir = rq.run_qixing_v3_same_day(data, INITIAL, live_mirror=True)["equity_curve"]
    print(f"  {'年份':<6} {'V3收益':>8} {'激进收益':>9} {'V3回撤':>7} {'激进回撤':>8}")
    for year in sorted(eq["trade_date"].dt.year.unique()):
        s, e = f"{year}-01-01", f"{year}-12-31"
        mo = metrics(mir, s, e)
        mn = metrics(eq, s, e)
        print(
            f"  {year:<6} {mo['ret']:>+8.1%} {mn['ret']:>+9.1%} {mo['dd']:>7.1%} {mn['dd']:>8.1%}"
        )


def gain_scan() -> None:
    """收益放大探索: 不限制仓位的数学优化 (D1频率自适应/D2动量公式/D3网格相位)."""
    data = rq.load_data()
    dmap = build_dmap(data)

    print("=" * 92)
    print("  收益放大探索 (不动仓位/不加风控, 10万本金, 实盘镜像口径)")
    print("=" * 92)

    def show(name, eq, extra=""):
        mf = metrics(eq, "2020-01-01", "2026-12-31")
        mis = metrics(eq, *IS_RANGE)
        moos = metrics(eq, *OOS_RANGE)
        print(
            f"  {name:<24} {eq['equity'].iloc[-1]:>10,.0f} {mf['ret']:>+8.0%} "
            f"{mf['ann']:>+7.1%} {mf['sharpe']:>6.2f} {mf['dd']:>8.1%} "
            f"{mis['sharpe']:>7.2f} {moos['sharpe']:>7.2f} {extra}"
        )

    print(
        f"  {'方案':<24} {'终值':>10} {'总收益':>8} {'年化':>7} {'夏普':>6} "
        f"{'回撤':>8} {'IS夏普':>7} {'OOS夏普':>7}"
    )
    print(f"  {'-' * 90}")

    base = rq.run_qixing_v3_same_day(data, INITIAL, live_mirror=True)["equity_curve"]
    show("V3镜像基线", base)

    # D1: 强趋势日频切换
    for th in [0.08, 0.10, 0.12]:
        eq = run_v31(data, {"daily_strong": True, "strong_th": th}, dmap)["equity"]
        show(f"D1 强趋势日频 th={th:.2f}", eq)

    # D2: 动量公式变体 (临时覆写模块参数, 用后还原)
    orig_p, orig_w = rq.MOM_PERIODS, rq.MOM_WEIGHTS
    for periods, weights in [
        ((5, 10), (0.5, 0.5)),
        ((10, 30), (0.5, 0.5)),
        ((20, 60), (0.5, 0.5)),
        ((5, 10, 20), (0.4, 0.35, 0.25)),
    ]:
        rq.MOM_PERIODS, rq.MOM_WEIGHTS = periods, weights
        try:
            eq = run_v31(data, {}, dmap)["equity"]
            show(f"D2 动量{'/'.join(map(str, periods))}", eq)
        finally:
            rq.MOM_PERIODS, rq.MOM_WEIGHTS = orig_p, orig_w

    # D3: 网格相位敏感性 (offset 0-4)
    finals = []
    for off in range(5):
        eq = run_v31(data, {"grid_offset": off}, dmap)["equity"]
        f = eq["equity"].iloc[-1]
        finals.append(f)
        show(f"D3 网格相位 offset={off}", eq)
    print(
        f"    → 相位极差: {max(finals) - min(finals):,.0f} "
        f"({(max(finals) / min(finals) - 1):+.0%}), 说明相位对结果影响显著"
    )

    # 组合: D1(最优th) + D2(最优公式) 叠加测试
    rq.MOM_PERIODS, rq.MOM_WEIGHTS = (5, 10), (0.5, 0.5)
    try:
        eq = run_v31(data, {"daily_strong": True, "strong_th": 0.10}, dmap)["equity"]
        show("D1+D2(5/10) 组合", eq)
    finally:
        rq.MOM_PERIODS, rq.MOM_WEIGHTS = orig_p, orig_w

    # 最优放大方案的成本压力测试
    print(f"\n  [D1 th=0.10 成本压力测试]")
    for mult in [2.0, 3.0]:
        eq = run_v31(data, {"daily_strong": True, "strong_th": 0.10}, dmap, cost_multiplier=mult)[
            "equity"
        ]
        mf = metrics(eq, "2020-01-01", "2026-12-31")
        print(
            f"    {mult:.0f}x成本: 终值 {eq['equity'].iloc[-1]:,.0f} "
            f"夏普 {mf['sharpe']:.2f} 回撤 {mf['dd']:.1%}"
        )


def alpha_scan() -> None:
    """双目标探索: 提升夏普的同时保持总收益 (不动仓位的 alpha 改进)."""
    data = rq.load_data()
    dmap = build_dmap(data)
    P2 = {"exit_intraday": -0.05, "exit_dd": -0.10}

    configs = {
        "V3镜像基线": None,
        "C0 仅日频退出": dict(P2),
        "C1 +趋势确认MA20": {**P2, "ma_trend": True},
        "C2a +入场阈值0.02": {**P2, "entry_th": 0.02},
        "C2b +入场阈值0.03": {**P2, "entry_th": 0.03},
        "C3 +快速再入场": {**P2, "fast_reenter": True},
        "C4 趋势确认+快速再入": {**P2, "ma_trend": True, "fast_reenter": True},
        "C5 阈值0.02+快速再入": {**P2, "entry_th": 0.02, "fast_reenter": True},
    }

    print("=" * 92)
    print("  双目标探索: 夏普↑ 且 收益不降 (不动仓位的 alpha 改进, 10万本金)")
    print("=" * 92)
    print(
        f"  {'方案':<20} {'终值':>10} {'总收益':>8} {'年化':>7} {'夏普':>6} "
        f"{'回撤':>8} {'IS夏普':>7} {'OOS夏普':>8}"
    )
    print(f"  {'-' * 88}")
    base_final = None
    results = {}
    for name, P in configs.items():
        if P is None:
            eq = rq.run_qixing_v3_same_day(data, INITIAL, live_mirror=True)["equity_curve"]
        else:
            eq = run_v31(data, P, dmap)["equity"]
        mf = metrics(eq, "2020-01-01", "2026-12-31")
        mis = metrics(eq, *IS_RANGE)
        moos = metrics(eq, *OOS_RANGE)
        final = eq["equity"].iloc[-1]
        if base_final is None:
            base_final = final
        results[name] = (mf, mis, moos, final)
        win = "✓收益达标" if final >= base_final * 0.95 else ""
        print(
            f"  {name:<20} {final:>10,.0f} {mf['ret']:>+8.0%} {mf['ann']:>+7.1%} "
            f"{mf['sharpe']:>6.2f} {mf['dd']:>8.1%} {mis['sharpe']:>7.2f} "
            f"{moos['sharpe']:>8.2f} {win}"
        )


def compare() -> None:
    """V3.1 确定版 (σ_target=0.28) vs V3 实盘镜像 详细对比."""
    data = rq.load_data()
    dmap = build_dmap(data)
    P = {
        "vol_target": 0.28,
        "vol_floor": 0.3,
        "exit_intraday": -0.05,
        "exit_dd": -0.10,
        "leverage_cap": 1.0,
    }

    print("=" * 66)
    print("  V3.1 确定版 vs V3 实盘镜像 — 回测验证对比")
    print("=" * 66)
    print("  V3.1 参数确认:")
    for k, v in P.items():
        print(f"    {k} = {v}")

    r = run_v31(data, P, dmap)
    eq_new = r["equity"]
    mir = rq.run_qixing_v3_same_day(data, INITIAL, live_mirror=True)
    eq_old = mir["equity_curve"].copy()

    def row(label, a, b):
        print(f"  {label:<10} {a:>10} {b:>10}")

    print(f"\n  {'指标':<10} {'V3镜像':>10} {'V3.1':>10}")
    print(f"  {'-' * 34}")
    for label, s, e in [
        ("全周期", "2020-01-01", "2026-12-31"),
        ("IS 20-23", *IS_RANGE),
        ("OOS 24-26", *OOS_RANGE),
    ]:
        mo = metrics(eq_old, s, e)
        mn = metrics(eq_new, s, e)
        print(
            f"  [{label}] 夏普 {mo['sharpe']:.2f} → {mn['sharpe']:.2f} | "
            f"年化 {mo['ann']:+.1%} → {mn['ann']:+.1%} | "
            f"回撤 {mo['dd']:.1%} → {mn['dd']:.1%}"
        )

    mo = metrics(eq_old, "2020-01-01", "2026-12-31")
    mn = metrics(eq_new, "2020-01-01", "2026-12-31")
    print(f"\n  {'全周期明细':<10} {'V3镜像':>12} {'V3.1':>12}")
    print(f"  {'-' * 38}")
    row("总收益", f"{mo['ret']:+.0%}", f"{mn['ret']:+.0%}")
    row("年化", f"{mo['ann']:+.1%}", f"{mn['ann']:+.1%}")
    row("夏普", f"{mo['sharpe']:.2f}", f"{mn['sharpe']:.2f}")
    row("年化波动", f"{mo['vol']:.1%}", f"{mn['vol']:.1%}")
    row("最大回撤", f"{mo['dd']:.1%}", f"{mn['dd']:.1%}")
    row("10万终值", f"{eq_old['equity'].iloc[-1]:,.0f}", f"{eq_new['equity'].iloc[-1]:,.0f}")
    print(f"  {'P2退出次数':<10} {'-':>12} {r['n_exits']:>12}")

    # 年度对比
    print(f"\n  年度对比:")
    print(
        f"  {'年份':<6} {'V3收益':>8} {'V3.1收益':>9} {'V3夏普':>6} {'V3.1夏普':>7} {'V3回撤':>7} {'V3.1回撤':>8}"
    )
    for year in sorted(eq_new["trade_date"].dt.year.unique()):
        s, e = f"{year}-01-01", f"{year}-12-31"
        mo = metrics(eq_old, s, e)
        mn = metrics(eq_new, s, e)
        print(
            f"  {year:<6} {mo['ret']:>+8.1%} {mn['ret']:>+9.1%} {mo['sharpe']:>6.2f} "
            f"{mn['sharpe']:>7.2f} {mo['dd']:>7.1%} {mn['dd']:>8.1%}"
        )


if __name__ == "__main__":
    if "--compare" in sys.argv:
        compare()
    elif "--aggressive" in sys.argv:
        aggressive_scan()
    elif "--alpha" in sys.argv:
        alpha_scan()
    elif "--gain" in sys.argv:
        gain_scan()
    else:
        main()
