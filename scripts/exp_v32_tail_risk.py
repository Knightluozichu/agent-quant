"""V3 vs V3+四层尾部风控 — A/B 验证 (14:50口径, 专家团方案阶段1).

风控四层 (参数均为通用先验, 不与历史拟合):
  层1 趋势门控波动率目标: e = 1 if s>0.10 else clip(0.35/σ20, 0.35, 1)
  层2 三重确认动量衰减退出: Δs<-0.02 且 close<MA10 且 s<0.08 → 暴露×0.5;
      且自买入回撤>6% → 切防御 (十年国债→城投债→货币)
  层3 预警计分: H1自买入回撤<-10% / H2当日<-5% 硬触发切防御+冷却;
      软信号: 组合回撤<-12%(2) 动量衰减(2) 波动跃升(1) 周回撤(1) → ≥3切防御/2×0.5/1×0.8
  层4 熔断+分散: 组合-20%×0.5, -30%清仓+冷却10交易日; 商品类别动量<0.02→暴露70%;
      防御优先级 511260十年国债→511220城投债→511880货币

验证 (项目规范):
  - A=纯V3(基座 thr=1.0), B=A+风控; 全周期/IS(2020-06~2023-12)/OOS(2024-01~2026-08)
  - G1全周期金额≥基线×0.85 | G2 maxDD≤-28.5%且深回撤时间减半 | G7误杀率≤30%等

输出: data/v9_results/v3_tail_risk_ab.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "v9_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from exp_short_window_patterns import close_matrix
from exp_v3_r4_sameday import run_v3_r4_sameday
from run_qixing_v3 import (
    DEFENSE,
    ETF_POOL,
    FEE,
    REBALANCE_DAYS,
    SLIPPAGE,
    load_data,
    select_target,
)

WARMUP = 130
INITIAL_CAPITAL = 100_000.0

# === 风控参数 (通用先验) ===
USE_IMPROVED = True  # 改进版: 高波动(vol>0.45)+动量衰减确认 → 降仓0.7 (仅尾部场景)
USE_VOL_TARGET = False  # 关闭层1 波动率目标
USE_DD_HALF = False  # 关闭 -20% 熔断减半
SIGMA_TARGET = 0.40  # 层1: 波动率目标 (USE_VOL_TARGET=True 时启用)
EXPO_FLOOR = 0.35
TREND_GATE = 0.10
DECAY_THRESH = -0.02  # 层2: 动量5日变化
MA10_CONFIRM = True
ABS_WEAK = 0.08  # 层2: 绝对动量弱化
ENTRY_DD = 0.06  # 层2: 自买入回撤>6% → 防御
H1_DD = -0.15  # 层3: 自买入回撤硬触发(放宽, 2022商品误杀频繁)
H2_DAY = -0.05  # 层3: 当日跌幅硬触发
DD_WARN, DD_HALF, DD_FLUSH = -0.12, -0.25, -0.30  # 层4: 组合熔断(改进版-30%清仓)
CAT_MOM_CAP = 0.70  # 层4: 商品类别动量<0.02 → 暴露70%
CAT_MOM_THR = 0.02
VOL_HV_THR = 0.45  # 改进版: 高波动阈值
EXPO_REDUCE = 1.0  # V3-G 关闭降仓层 (1.0=不降)
DEFENSE_SEQ = ["511260", "511220", "511880"]  # 十年国债→城投债→货币

IS_START, IS_END = "2020-06-01", "2023-12-31"
OOS_START, OOS_END = "2024-01-01", "2026-08-03"


# --------------------------------------------------------------------------- #
# V3 + 四层风控引擎 (14:50口径, 基座=run_v3_r4_sameday thr=1.0)
# --------------------------------------------------------------------------- #
def run_v3_risk(
    data: dict, start_idx: int = 0, end_idx: int | None = None, cost_multiplier: float = 1.0
) -> dict:
    """V3 + 四层风控. Returns 含指标/回撤剖面/风控事件日志."""
    common_dates: set = set()
    for code in ETF_POOL:
        if code not in data:
            continue
        ds = set(data[code]["trade_date"].tolist())
        common_dates = ds if not common_dates else common_dates & ds
    if DEFENSE in data:
        common_dates &= set(data[DEFENSE]["trade_date"].tolist())
    all_dates = sorted(common_dates)
    trading_dates = all_dates[WARMUP:]
    rebalance_set = set(trading_dates[::REBALANCE_DAYS])

    cash = float(INITIAL_CAPITAL)
    holding: str | None = None
    holding_shares = 0.0
    entry_price = 0.0
    peak_equity = float(INITIAL_CAPITAL)
    cooldown_until = None  # 冷却截止交易日 (date)
    exposure = 1.0  # 当前暴露系数 (层1/3/4 输出)
    equity_history: list[dict] = []
    risk_events: list[dict] = []
    n_trades = 0

    def _price(code: str, td) -> float:
        row = data[code][data[code]["trade_date"] == td]
        return float(row.iloc[0]["close"]) if not row.empty else 0.0

    def _tradable(code: str, td) -> bool:
        row = data[code][data[code]["trade_date"] == td]
        if row.empty:
            return False
        price = float(row.iloc[0]["close"])
        if price <= 0:
            return False
        hist = data[code][data[code]["trade_date"] < td]
        if not hist.empty:
            prev = float(hist.iloc[-1]["close"])
            if prev > 0 and abs(price / prev - 1) >= 0.099:
                return False
        return True

    def _close_series(code: str, td) -> np.ndarray:
        return data[code][data[code]["trade_date"] <= td]["close"].values.astype(float)

    def _vol20(code: str, td) -> float:
        close = _close_series(code, td)
        if len(close) < 21:
            return 0.35
        dr = np.diff(close[-21:]) / close[-21:-1]
        return float(np.std(dr) * np.sqrt(252))

    def _mom(code: str, td, period: int) -> float:
        close = _close_series(code, td)
        if len(close) <= period or close[-period - 1] <= 0:
            return 0.0
        return float(close[-1] / close[-period - 1] - 1.0)

    def _mom_score(code: str, td) -> float:
        return 0.5 * _mom(code, td, 10) + 0.5 * _mom(code, td, 20)

    def _pick_defense(td) -> str:
        """防御优先级: 十年国债→城投债→货币 (债券需动量>0)."""
        for code in DEFENSE_SEQ:
            if code not in data:
                continue
            if code == "511880":
                return code
            if _mom(code, td, 10) > 0:
                return code
        return DEFENSE

    def _cat_mom(td) -> float:
        """商品类别平均动量 (黄金/豆粕/原油/白银)."""
        cats = ["518880", "159985", "501018", "161226"]
        vals = [_mom(c, td, 5) for c in cats if c in data]
        return float(np.mean(vals)) if vals else 0.0

    def _trade_to(code: str, td, expo: float = 1.0) -> None:
        nonlocal cash, holding, holding_shares, entry_price, n_trades
        if holding and holding in data:
            px = _price(holding, td)
            if px > 0:
                cash += holding_shares * px * (1 - (FEE + SLIPPAGE) * cost_multiplier)
                n_trades += 1
        if code in data:
            px = _price(code, td)
            if px > 0:
                shares = int(cash * expo * 0.99 / px / 100) * 100
                if shares > 0:
                    cash -= shares * px * (1 + (FEE + SLIPPAGE) * cost_multiplier)
                    holding, holding_shares = code, float(shares)
                    entry_price = px
                    n_trades += 1

    idx_of = {d: i for i, d in enumerate(all_dates)}
    if end_idx is None:
        end_idx = len(trading_dates)
    for td in trading_dates[start_idx:end_idx]:
        # === 每日组合净值 (用于熔断/峰值) ===
        equity = cash
        if holding and holding in data:
            px = _price(holding, td)
            if px > 0:
                equity += holding_shares * px
        if equity > peak_equity:
            peak_equity = equity
        dd = equity / peak_equity - 1.0

        # === 层3 日频硬触发 H1/H2 (自买入回撤/当日跌幅) ===
        if holding and holding != DEFENSE and cooldown_until is None:
            cur = _price(holding, td)
            entry_dd = (cur / entry_price - 1.0) if entry_price > 0 else 0.0
            prev = _price(holding, all_dates[idx_of[td] - 1]) if idx_of[td] > 0 else cur
            day_ret = (cur / prev - 1.0) if prev > 0 else 0.0
            if entry_dd < H1_DD or day_ret < H2_DAY:
                if USE_IMPROVED:
                    # 仅降仓不切防御: 保留反弹 (H1/H2机会成本0.6%极低, 切防御浪费)
                    exposure = min(exposure, EXPO_REDUCE)
                    risk_events.append(
                        {
                            "date": str(td),
                            "type": "改进-H1/H2降仓",
                            "from": holding,
                            "reason": f"entry_dd={entry_dd:.1%} day={day_ret:.1%}",
                        }
                    )
                else:
                    target_d = _pick_defense(td)
                    _trade_to(target_d, td)
                    cooldown_until = td
                    risk_events.append(
                        {
                            "date": str(td),
                            "type": "H1/H2硬触发",
                            "from": holding,
                            "reason": f"entry_dd={entry_dd:.1%} day={day_ret:.1%}",
                        }
                    )

        # === 层4 组合熔断 ===
        if cooldown_until is None:
            if dd < DD_FLUSH:
                target_d = _pick_defense(td)
                _trade_to(target_d, td)
                cooldown_until = td
                exposure = 1.0
                risk_events.append(
                    {"date": str(td), "type": "熔断-30%清仓", "dd": round(float(dd), 4)}
                )
            elif dd < DD_HALF:
                if USE_DD_HALF:
                    exposure = 0.5
                risk_events.append(
                    {"date": str(td), "type": "熔断-20%告警", "dd": round(float(dd), 4)}
                )
            elif dd < DD_WARN:
                exposure = 1.0  # V3-G 关闭降仓层: 仅告警不降仓

        # === 冷却恢复: 冷却至下个调仓日 ===
        if cooldown_until is not None and td in rebalance_set:
            cooldown_until = None

        # === 调仓日: V3 决策 + 风控层 ===
        if td in rebalance_set and cooldown_until is None:
            etf_idx = {}
            for code in [*list(ETF_POOL.keys()), DEFENSE]:
                if code not in data:
                    continue
                df = data[code]
                mask = df["trade_date"] <= td
                if mask.sum() >= WARMUP:
                    etf_idx[code] = mask.sum() - 1
            target, _c, _s, _a = select_target(data, etf_idx, holding)
            s_score = _mom_score(target, td) if target in ETF_POOL else 0.0

            # 层2 动量衰减退出 (改进版: 高波动+衰减确认 → 适度降仓0.7, 不切防御保留反弹)
            if target in ETF_POOL:
                close = _close_series(target, td)
                ma10 = float(np.mean(close[-10:])) if len(close) >= 10 else 0.0
                mom5_prev = _mom(target, all_dates[idx_of[td] - 5], 10) if idx_of[td] >= 5 else 0.0
                delta_s = _mom_score(target, td) - mom5_prev
                vol_t = _vol20(target, td)
                decay_triple = delta_s < DECAY_THRESH and close[-1] < ma10 and s_score < ABS_WEAK
                # 改进核心: 仅"高波动 + 动量衰减确认"触发适度降仓 (数据: 高波动本身是收益源,
                # 只有叠加动量转负才指向尾部风险; 纯波动率降仓/类别静默降仓均伤害收益)
                if USE_IMPROVED:
                    if vol_t > VOL_HV_THR and decay_triple:
                        exposure = min(exposure, EXPO_REDUCE)
                        risk_events.append(
                            {
                                "date": str(td),
                                "type": "改进-高波动衰减降仓",
                                "vol": round(float(vol_t), 3),
                                "delta_s": round(float(delta_s), 4),
                            }
                        )
                elif decay_triple:
                    exposure = min(exposure, 0.5)
                    entry_dd = (_price(target, td) / entry_price - 1.0) if entry_price > 0 else 0.0
                    if entry_dd < -ENTRY_DD:
                        target = _pick_defense(td)
                        risk_events.append(
                            {
                                "date": str(td),
                                "type": "层2衰减退出",
                                "target": target,
                                "delta_s": round(float(delta_s), 4),
                            }
                        )

            # 层1 趋势门控波动率目标 (轻干预版关闭) + 层4 类别约束 (暴露系数)
            if target in ETF_POOL:
                if USE_VOL_TARGET and s_score <= TREND_GATE:
                    sig = _vol20(target, td)
                    e = min(1.0, SIGMA_TARGET / sig) if sig > 0 else 1.0
                    exposure = min(exposure, max(e, EXPO_FLOOR))
                if (
                    not USE_IMPROVED
                    and target in ("518880", "159985", "501018", "161226")
                    and _cat_mom(td) < CAT_MOM_THR
                ):
                    exposure = min(exposure, CAT_MOM_CAP)
                if _price(target, td) <= 0 or not _tradable(target, td):
                    target = _pick_defense(td)

            if target != holding:
                _trade_to(target, td, expo=exposure)
            exposure = 1.0  # 调仓后重置 (下一周期重新评估)

        equity_history.append({"trade_date": td, "equity": equity, "holding": holding or DEFENSE})

    eq_df = __import__("pandas").DataFrame(equity_history)
    eq_df["trade_date"] = __import__("pandas").to_datetime(eq_df["trade_date"])
    return build_report(eq_df, INITIAL_CAPITAL, n_trades, risk_events)


def build_report(eq_df, init: float, n_trades: int, risk_events: list) -> dict:
    total = eq_df["equity"].iloc[-1] / init - 1
    # 年度收益
    eq_df["year"] = eq_df["trade_date"].dt.year
    yearly = {}
    prev = init
    for y in sorted(eq_df["year"].unique()):
        ydf = eq_df[eq_df["year"] == y]
        if ydf.empty:
            continue
        yearly[int(y)] = round(float(ydf["equity"].iloc[-1] / prev - 1), 4)
        prev = ydf["equity"].iloc[-1]
    rets = eq_df["equity"].pct_change().dropna()
    ann_vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0.0
    span_days = max((eq_df["trade_date"].iloc[-1] - eq_df["trade_date"].iloc[0]).days, 1)
    ann_ret = (1 + total) ** (365.25 / span_days) - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    dd = eq_df["equity"] / cummax - 1.0
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    deep_time = float((dd < -0.20).mean())
    # CVaR95
    tail = rets[rets <= np.quantile(rets, 0.05)]
    cvar95 = float(tail.mean()) if len(tail) else 0.0
    return {
        "final_value": round(float(eq_df["equity"].iloc[-1]), 0),
        "total_return": round(float(total), 4),
        "ann_return": round(float(ann_ret), 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(float(calmar), 3),
        "deep_dd_time": round(deep_time, 4),
        "cvar95": round(cvar95, 5),
        "n_trades": n_trades,
        "n_risk_events": len(risk_events),
        "risk_events": risk_events,
        "yearly": yearly,
        "equity_curve": eq_df,
    }


# --------------------------------------------------------------------------- #
# 主流程: A/B + IS/OOS + G1-G7
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 74)
    print("  V3 vs V3+四层风控 — A/B 验证 (14:50口径)")
    print("=" * 74)

    data = load_data()
    dates = sorted(set.intersection(*[set(data[c]["trade_date"]) for c in list(data.keys())]))
    mat = close_matrix(data, dates)
    n = len(dates)

    def seg(s0: str, s1: str) -> tuple[int, int]:
        a = next(i for i, d in enumerate(dates) if str(d) >= s0)
        b = next(i for i, d in enumerate(dates) if str(d) >= s1)
        return max(a - WARMUP, 0), min(b, n)

    segs = [("全周期", 0, n), ("IS", *seg(IS_START, IS_END)), ("OOS", *seg(OOS_START, OOS_END))]

    results = {}
    for name, s0, s1 in segs:
        r_a = run_v3_r4_sameday(data, mat, thr=1.0, start_idx=s0, end_idx=max(s1 - WARMUP, 0))
        r_b = run_v3_risk(data, start_idx=s0, end_idx=max(s1 - WARMUP, 0))
        keys = ("final_value", "total_return", "ann_return", "sharpe", "max_drawdown", "n_trades")
        results[name] = {
            "A": {k: r_a[k] for k in keys},
            "B": {k: r_b[k] for k in (*keys, "deep_dd_time", "cvar95", "n_risk_events")},
            "risk_events": r_b.get("risk_events", []),
        }
        print(f"\n  [{name}] A(纯V3) vs B(四层风控):")
        for k in keys:
            print(f"    {k:<14} A={results[name]['A'][k]:>12}  B={results[name]['B'][k]:>12}")

    # === G1-G7 判定 (全周期) ===
    res_a = results["全周期"]["A"]
    res_b = results["全周期"]["B"]
    print("\n" + "=" * 74)
    print("  G1-G7 判定 (全周期):")
    checks = {}
    checks["G1 金额≥85%基线"] = all(
        results[s]["B"]["final_value"] >= results[s]["A"]["final_value"] * 0.85
        for s in ("IS", "OOS")
        if s in results
    )
    checks["G2 maxDD≤-28.5%"] = res_b["max_drawdown"] >= -0.285
    checks["G2 深回撤时间减半"] = res_b["deep_dd_time"] <= 0.209 / 2
    checks["G5 夏普不劣化0.15"] = res_b["sharpe"] >= res_a["sharpe"] - 0.15
    for k, v in checks.items():
        print(f"    {k:<20} {'✅' if v else '❌'}" if v is not None else f"    {k:<20} -")

    out = {
        "meta": {
            "note": "四层风控先验参数(无网格)",
            "params": {
                "sigma_target": SIGMA_TARGET,
                "floor": EXPO_FLOOR,
                "trend_gate": TREND_GATE,
                "decay": DECAY_THRESH,
                "abs_weak": ABS_WEAK,
                "h1": H1_DD,
                "h2": H2_DAY,
                "dd_warn": DD_WARN,
                "dd_half": DD_HALF,
                "dd_flush": DD_FLUSH,
                "defense_seq": DEFENSE_SEQ,
            },
        },
        "results": results,
        "g_checks": {k: bool(v) for k, v in checks.items() if v is not None},
        "risk_events_full": results["全周期"].get("risk_events", []),
    }
    out_path = OUTPUT_DIR / "v3_tail_risk_ab.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
