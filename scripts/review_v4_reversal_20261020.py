"""H6: frozen, causal reversal-veto research. No broker or production writes.

Run: uv run --no-sync python -m scripts.review_v4_reversal_20261020
Rules and verdict are preregistered in tasks/EXPERIMENTS.md before results.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from scripts import live_signal as live
from scripts import run_qixing_v3 as rq
from scripts.review_oil_proxy_20260916 import prepare_bars
from scripts.review_rotation_20260916 import DefensiveOverride, replay_next_open

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/etf_momentum_20261020"
OUT = ROOT / "artifacts/v4_reversal_20261020"
CUTOFF = date(2026, 10, 19)


@dataclass(frozen=True)
class Rule:
    name: str
    return_days: int
    ma_days: int
    gate_entries: bool = False


RULES = (
    Rule("F-exit", 3, 10),
    Rule("F-gate", 3, 10, True),
    Rule("S-exit", 5, 20),
    Rule("S-gate", 5, 20, True),
)


def is_weak(close: NDArray[np.float64], rule: Rule) -> bool:
    if len(close) < max(rule.return_days + 1, rule.ma_days):
        raise ValueError("Insufficient signal history")
    if not np.isfinite(close).all() or np.any(close <= 0):
        raise ValueError("Invalid signal price")
    return bool(
        close[-1] < close[-rule.return_days - 1] and close[-1] < np.mean(close[-rule.ma_days :])
    )


class ReversalFilter:
    """Only the <=T close prefix is queried; never changes a canonical risk exit."""

    def __init__(self, data: dict[str, pd.DataFrame], rule: Rule) -> None:
        self.rule = rule
        self.closes = {c: f.close.to_numpy(dtype=float) for c, f in data.items()}
        self.index = {
            c: dict(zip(f.trade_date, range(len(f)), strict=True)) for c, f in data.items()
        }

    def weak(self, code: str, td: date) -> bool:
        if code not in self.index or td not in self.index[code]:
            raise ValueError(f"Missing signal observation for {code} on {td}")
        return is_weak(self.closes[code][: self.index[code][td] + 1], self.rule)

    def __call__(
        self, td: date, holding: str | None, target: str, allowed: bool
    ) -> DefensiveOverride:
        existing_exit = allowed and target != holding
        if holding and holding != live.DEFENSE and not existing_exit and self.weak(holding, td):
            return DefensiveOverride(live.DEFENSE, True, "reversal_exit")
        if (
            self.rule.gate_entries
            and allowed
            and target != holding
            and target != live.DEFENSE
            and self.weak(target, td)
        ):
            return DefensiveOverride(live.DEFENSE, False, "weak_entry_blocked")
        return DefensiveOverride(target)


def assert_evaluation_window(start: date, end: date) -> None:
    if start > end:
        raise ValueError("Reversed evaluation window")
    config = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())
    partition = config["protocol"]["time_partition"]
    lo = pd.Timestamp(partition["locked_test_start"]).date()
    hi = pd.Timestamp(partition["locked_test_end"]).date()
    if start <= hi and end >= lo:
        raise ValueError("Evaluation window overlaps locked test")


def load_frozen_data(directory: Path = INPUT) -> dict[str, pd.DataFrame]:
    manifest = json.loads((directory / "data_manifest.json").read_text())
    if manifest["requested_cutoff"] != str(CUTOFF):
        raise ValueError("Unexpected frozen cutoff")
    # Check every declared file before reading any untrusted parquet bytes.
    for code, item in manifest["assets"].items():
        path = directory / "bars" / f"{code}.parquet"
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {code}")
    codes = [*rq.ETF_POOL, rq.DEFENSE]
    if set(manifest["assets"]) != set(codes):
        raise ValueError("Frozen input pool mismatch")
    return {
        c: prepare_bars(pd.read_parquet(directory / "bars" / f"{c}.parquet"), c, CUTOFF)
        for c in codes
    }


@contextmanager
def research_pool(pool: dict[str, str]) -> Iterator[None]:
    """Process-local research scope, restored even on failure. No files are written."""
    old_rq, old_live = rq.ETF_POOL, live.ETF_POOL
    try:
        rq.ETF_POOL = dict(pool)
        live.ETF_POOL = dict(pool)
        yield
    finally:
        rq.ETF_POOL, live.ETF_POOL = old_rq, old_live


def audit_ledger(
    result: dict[str, Any], data: dict[str, pd.DataFrame], *, cost_multiplier: float
) -> dict[str, float]:
    """Independently reconstruct every fill and daily cash+market-value balance."""
    cash, shares, holding, total_cost, error = 100000.0, 0, None, 0.0, 0.0
    by_date: dict[str, list[dict[str, Any]]] = {}
    for trade in result["trades"]:
        by_date.setdefault(trade["date"], []).append(trade)
    frames = {c: f.set_index("trade_date") for c, f in data.items()}
    calendar = sorted(set.intersection(*(set(f.trade_date) for f in data.values())))
    positions = {td: i for i, td in enumerate(calendar)}
    for row in result["curve"].itertuples():
        td = row.trade_date
        for trade in by_date.get(str(td), []):
            signal = date.fromisoformat(trade["signal_date"])
            if positions[td] != positions[signal] + 1:
                raise ValueError("Fill is not next observed open")
            code, price = trade["code"], float(trade["price"])
            if price != float(frames[code].loc[td, "open"]):
                raise ValueError("Fill is not recorded open")
            if trade["action"] == "buy":
                if holding is not None:
                    raise ValueError("Overlapping ledger holdings")
                shares = int(trade["shares"])
                if shares <= 0 or shares % 100:
                    raise ValueError("Invalid lot")
                holding = code
                cost = shares * price * (rq.FEE + rq.SLIPPAGE) * cost_multiplier
                cash -= shares * price + cost
            else:
                if code != holding or shares <= 0:
                    raise ValueError("Sale without ledger holding")
                cost = shares * price * (rq.FEE + rq.SLIPPAGE) * cost_multiplier
                cash += shares * price - cost
                holding, shares = None, 0
            total_cost += cost
        equity = cash + (shares * float(frames[holding].loc[td, "close"]) if holding else 0)
        error = max(error, abs(equity - row.equity), abs(cash - row.cash))
        if (
            error > 1e-7
            or cash < -1e-7
            or holding != (None if pd.isna(row.holding) else row.holding)
        ):
            raise ValueError("Ledger accounting mismatch")
    return {"total_cost": total_cost, "max_accounting_error": error}


def summarize(
    result: dict[str, Any], data: dict[str, pd.DataFrame], cost: float, end: date
) -> dict[str, Any]:
    curve = result["curve"]
    equity = np.r_[100000.0, curve.equity.to_numpy(dtype=float)]
    daily = np.diff(equity) / equity[:-1]
    drawdown = equity / np.maximum.accumulate(equity) - 1
    longest, current = 0, 0
    for dd in drawdown[1:]:
        current = current + 1 if dd < -1e-12 else 0
        longest = max(longest, current)
    event_keys = {
        (e["date"], e["holding"])
        for e in result["research_events"]
        if e["reason"] == "reversal_exit"
    }
    rebounds = []
    for trade in result["trades"]:
        if trade["action"] != "sell" or (trade["signal_date"], trade["code"]) not in event_keys:
            continue
        f = data[trade["code"]]
        f = f.loc[f.trade_date <= end].reset_index(drop=True)
        idx = f.index[f.trade_date == date.fromisoformat(trade["date"])][0]
        if idx + 5 < len(f):
            rebounds.append(float(f.iloc[idx + 5].open / trade["price"] - 1))
    trough = int(np.argmin(drawdown))
    peak = int(np.argmax(equity[: trough + 1]))
    recovered = np.flatnonzero(equity[trough + 1 :] >= equity[peak])
    stats = dict(result["metrics"])
    stats.update(
        max_drawdown=float(drawdown.min()),
        worst_day=float(daily.min()),
        worst_dd_peak=str(curve.iloc[peak - 1].trade_date) if peak else "initial_cash",
        worst_dd_trough=str(curve.iloc[trough - 1].trade_date) if trough else "initial_cash",
        recovered_on=(
            str(curve.iloc[trough + int(recovered[0])].trade_date) if len(recovered) else None
        ),
        longest_underwater_days=longest,
        trade_legs=len(result["trades"]),
        closed_risk_positions=sum(
            t["action"] == "sell" and t["code"] != live.DEFENSE for t in result["trades"]
        ),
        risk_day_fraction=float((curve.holding.notna() & (curve.holding != live.DEFENSE)).mean()),
        protective_exit_signals=sum(
            e["reason"] == "reversal_exit" for e in result["research_events"]
        ),
        weak_entry_blocks=sum(
            e["reason"] == "weak_entry_blocked" for e in result["research_events"]
        ),
        exit_rebound_5d_sample=len(rebounds),
        exit_rebound_5d_positive=sum(r > 0 for r in rebounds),
        exit_rebound_5d_mean=float(np.mean(rebounds)) if rebounds else None,
        unfilled_last_signal=bool(result["unfilled_next_open"]),
        **audit_ledger(result, data, cost_multiplier=cost),
    )
    return stats


def verdict(table: pd.DataFrame) -> dict[str, Any]:
    outputs = {}
    subset = table.loc[table.window.isin(["2025", "2026"]) & table.cost.isin([1, 2])]
    for rule in RULES:
        pairs = subset.loc[subset.strategy == rule.name].merge(
            subset.loc[subset.strategy == "V4"],
            on=["pool", "window", "cost"],
            suffixes=("_candidate", "_base"),
            validate="one_to_one",
        )
        dd_gain = pairs.max_drawdown_candidate - pairs.max_drawdown_base
        ret_delta = pairs.total_return_candidate - pairs.total_return_base
        complete = pairs.calendar_status_candidate.eq("complete")
        qualifies = (dd_gain >= 0.03) & (ret_delta >= -0.05) & complete
        passed = len(pairs) == 8 and bool(qualifies.all())
        failed_available = int((complete & ~qualifies).sum())
        outputs[rule.name] = {
            "status": (
                "forward_shadow_candidate_only"
                if passed
                else "not_passed_available_cells"
                if failed_available
                else "blocked_incomplete_calendar"
                if not bool(complete.all())
                else "not_passed"
            ),
            "passing_cells": int(qualifies.sum()),
            "required_cells": 8,
            "complete_cells": int(complete.sum()),
            "failed_complete_cells": failed_available,
            "incomplete_cells": int((~complete).sum()),
            "minimum_dd_improvement_complete": float(dd_gain[complete].min()),
            "minimum_return_difference_complete": float(ret_delta[complete].min()),
        }
    return outputs


def markdown_table(frame: pd.DataFrame) -> str:
    rows = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    rows.extend(
        "| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def write_report(table: pd.DataFrame, curves: pd.DataFrame, output: dict[str, Any]) -> None:
    """Render only complete windows in the primary chart; disclose all diagnostics."""
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"V4": "#1d3557", "F-exit": "#e76f51", "F-gate": "#2a9d8f", "S-gate": "#a277b5"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col")
    for col, window in enumerate(("2025", "last30")):
        for name, color in colors.items():
            f = curves.loc[
                (curves.pool == "with_oil")
                & (curves.window == window)
                & (curves.cost == 1)
                & (curves.strategy == name)
            ]
            values = f.equity.to_numpy(dtype=float)
            dates = pd.to_datetime(f.trade_date)
            axes[0, col].plot(
                dates,
                (values / 100000 - 1) * 100,
                label=name,
                color=color,
                linewidth=2 if name == "V4" else 1.4,
            )
            dd = values / np.maximum.accumulate(np.r_[100000, values])[1:] - 1
            axes[1, col].plot(dates, dd * 100, color=color, linewidth=1.4)
        axes[0, col].set_title("2025 全年" if window == "2025" else "最近30交易日（现金重启）")
        axes[0, col].set_ylabel("累计收益 %")
        axes[1, col].set_ylabel("回撤 %")
        axes[0, col].legend(fontsize=9)
        for ax in axes[:, col]:
            ax.grid(alpha=0.2)
            ax.tick_params(axis="x", rotation=20)
    fig.suptitle("V4反转保护验证｜含油研究池｜收盘信号→次日开盘｜单边费滑0.15%", fontsize=13)
    fig.text(
        0.5,
        0.01,
        "历史研究，不是实盘收益；两窗各10万元现金重启；S-exit与S-gate在图示区间重合。",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)

    sections = [
        "# H6：V4反转保护离线验证（2026-10-20）",
        "## 结论",
        "**4个固定候选均未通过已完成的2025完整日历检验，不建议替换V4或进入影子候选。** "
        "快速退出多次避开回撤，也多次卖出随后反弹的资产；更快不等于更高收益或更低整体回撤。"
        "2026及2021–22另有缺行阻塞，不能补充为完整可执行性证据。未部署、未下单、未修改账户。",
        "## 输入、规则、成交与裁决",
        "- 输入：冻结8只日线，截止2026-10-19，逐文件SHA256核对；"
        "详见input_manifest.json/recipe.json。"
        "含油研究池与删油运行池分开；含油结果不代表已核实产品准入/溢价。",
        "- V4原版已有快慢共识提前轮动、2次确认和3日锁。本轮不改核心参数。"
        "F=3日收益负且跌破MA10；S=5日收益负且跌破MA20。"
        "exit只在原V4本日不退出时强制转511880；gate另拦截同样弱势的可执行买入。"
        "不改变原V4主动退出；再入遵循原网格及提前轮动，无额外冷却。",
        "- 各窗10万元现金重启，网格全历史固定；T收盘信号→下一公共观察日open，"
        "100份整数、99%现金买入，单边费滑1x=0.15%、2x=0.30%、3x=0.45%。"
        "非同日收盘成交。未模拟限价成交概率、开盘封板、停牌申报拒单、最小佣金、盘口冲击或税费细分。",
        "- 预登记8格：两池×2025/2026×1x/2x，每格回撤改善≥3pp且净收益较V4不差超过5pp。"
        "4格2025均未通过；4格2026日历不完整不计验收。阈值未按结果修改。",
        "- 2023–24封存期不回放/评分，仅作后续信号预热；此前历史已被查看，非独立样本外。"
        "last30为2026-08-31至10-19，只有30日不参与裁决。多候选与嵌套窗口不视作独立重复，未计算PBO或统计显著性。",
        "## 完整窗口：2025与最近30日（1x）",
    ]
    for window in ("2025", "last30"):
        frame = table.loc[
            (table.window == window) & (table.cost == 1),
            [
                "pool",
                "strategy",
                "total_return",
                "max_drawdown",
                "trade_legs",
                "closed_risk_positions",
                "protective_exit_signals",
                "weak_entry_blocks",
            ],
        ].copy()
        for column in ("total_return", "max_drawdown"):
            frame[column] = frame[column].map(lambda x: f"{x:.2%}")
        sections.extend(["### " + window, markdown_table(frame)])
    sections.extend(
        [
            "## 成本压力：2025完整日历",
            markdown_table(
                table.loc[
                    table.window == "2025",
                    [
                        "pool",
                        "cost",
                        "strategy",
                        "total_return",
                        "max_drawdown",
                        "trade_legs",
                        "total_cost",
                    ],
                ].round(6)
            ),
            "## 反弹/持仓诊断：2025，1x",
            "退出后5个观察日open到open涨幅仅作事后诊断，不是完整归因，不参与信号；"
            "正收益不能单独证明该退出错误（期间可能还有下跌），末端缺5日样本不计。",
            markdown_table(
                table.loc[
                    (table.window == "2025") & (table.cost == 1),
                    [
                        "pool",
                        "strategy",
                        "risk_day_fraction",
                        "longest_underwater_days",
                        "exit_rebound_5d_sample",
                        "exit_rebound_5d_positive",
                        "exit_rebound_5d_mean",
                    ],
                ].round(6)
            ),
            "2025含油V4完整风险持仓18笔，快速gate22笔，均不足30笔，不能声称统计有效。"
            "快速gate13个保护退出信号中12个有完整5日后续，10个随后开盘到开盘上涨，均值约+2.54%。"
            "这与反复下车/错失回升相容，但不是严格因果归因。",
            "## 缺行窗口：只供机制诊断，不能认证下一真实交易日成交或完整回撤",
            "2021-02-08缺159915，2022-01-13缺513100；2026-01-22缺161226，"
            "01-30缺501018/161226，03-24缺501018。原因未区分停牌/源漏行；"
            "保留原始数据不填造，不因交集看起来收益更好而解除阻塞。详见data_audit.json。",
            markdown_table(
                table.loc[
                    table.calendar_status != "complete",
                    [
                        "pool",
                        "window",
                        "cost",
                        "strategy",
                        "total_return",
                        "max_drawdown",
                        "calendar_status",
                    ],
                ].round(6)
            ),
            "## 结果裁决",
            "```json",
            json.dumps(output["verdict"], ensure_ascii=False, indent=2),
            "```",
            "## 工程与研究自审",
            "- 无效果钩子逐点复现原next-open基线；独立重建所有成交/现金/市值，"
            "最大金额误差小于1e-7元；"
            "每笔成交核对信号日后的下一公共观察日open，不在最终信号日偷成交；最后未成交意图单列。",
            "- 测试覆盖未来扰动前缀不变、整手、成本、反转双条件、弱势入场、只允许防御覆盖、"
            "锁从实际成交日起算、异常价格/哈希/窗口与封存边界。测试与静态检查范围见tests.log/ruff.log/mypy.log。",
            "- 旧报告same-close高收益不能证明14:50已知最终收盘；本轮不据此宣称必须同日交易，"
            "也不推断次日开盘口径等于用户实际持仓收益。真实执行还需分钟快照与成交记录。",
            "- 不追求30日事后最优路径，不用未来收益选基金，未训练预测/强化学习模型，"
            "未按结果扫描阈值。不能由此推断所有反转保护都无效。",
            "## 复跑与产物",
            "```bash",
            "uv run --no-sync python -m scripts.review_v4_reversal_20261020",
            "```",
            "输出目录：artifacts/v4_reversal_20261020。metrics.csv为全量汇总，curves.parquet为逐日净值，"
            "results.json保留逐笔成交/保护事件/末日未成交意图；recipe.json记录规则及源码哈希。",
            "![完整窗口对比](../artifacts/v4_reversal_20261020/comparison.png)",
        ]
    )
    report = ROOT / "reports/2026-10-20_V4反转保护验证.md"
    report.write_text("\n\n".join(sections) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_frozen_data()
    original = dict(rq.ETF_POOL)
    pools = {
        "with_oil": original,
        "without_oil": {c: n for c, n in original.items() if c != "501018"},
    }
    common = sorted(set.intersection(*(set(f.trade_date) for f in data.values())))
    grid = set(common[130 :: rq.REBALANCE_DAYS])
    windows = {
        "2021-2022": (date(2021, 1, 1), date(2022, 12, 31)),
        "2025": (date(2025, 1, 1), date(2025, 12, 31)),
        "2026": (date(2026, 1, 1), CUTOFF),
        "last30": (common[-30], CUTOFF),
    }
    sources = [
        Path(__file__),
        ROOT / "scripts/review_rotation_20260916.py",
        ROOT / "scripts/review_oil_proxy_20260916.py",
        ROOT / "scripts/live_signal.py",
        ROOT / "scripts/run_qixing_v3.py",
        ROOT / "scripts/qixing_v4.py",
        ROOT / "scripts/risk_overrides.py",
        ROOT / "configs/protocol.yaml",
        ROOT / "tasks/EXPERIMENTS.md",
    ]
    metadata = {
        "hypothesis": "H6",
        "cutoff": str(CUTOFF),
        "rules": [asdict(r) for r in RULES],
        "windows": windows,
        "grid_anchor": str(common[130]),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "v4_params": asdict(live.v4.V4_PARAMS),
        "single_side_cost": rq.FEE + rq.SLIPPAGE,
        "execution": "T close -> next observed day open; no same-close evaluation",
        "research_only": True,
        "all_history_previously_seen": True,
    }
    (OUT / "input_manifest.json").write_bytes((INPUT / "data_manifest.json").read_bytes())
    # Freeze recipe/source identities BEFORE the first evaluation result is calculated.
    (OUT / "recipe.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str)
    )
    rows, curves, runs, audits = [], [], [], []
    for pool_name, pool in pools.items():
        codes = [*pool, rq.DEFENSE]
        selected = {c: data[c] for c in codes}
        with research_pool(pool):
            for window, (start, end) in windows.items():
                assert_evaluation_window(start, end)
                visible = [d for d in common if start <= d <= end]
                # Do not let intersection silently hide missing bars during evaluation.
                union = sorted(set.union(*(set(f.trade_date) for f in selected.values())))
                missing = [d for d in union if start <= d <= end and d not in visible]
                calendar_status = "incomplete_diagnostic" if missing else "complete"
                audits.append(
                    {
                        "pool": pool_name,
                        "window": window,
                        "status": calendar_status,
                        "missing": [
                            {
                                "date": str(d),
                                "codes": [
                                    c for c, f in selected.items() if d not in set(f.trade_date)
                                ],
                            }
                            for d in missing
                        ],
                    }
                )
                (OUT / "data_audit.json").write_text(
                    json.dumps(audits, ensure_ascii=False, indent=2)
                )
                for cost in (1, 2, 3):
                    for rule in (None, *RULES):
                        strategy = rule.name if rule else "V4"
                        identity = {
                            "pool": pool_name,
                            "window": window,
                            "cost": cost,
                            "strategy": strategy,
                        }
                        replay = replay_next_open(
                            selected,
                            start=start,
                            end=end,
                            calendar=common,
                            grid=grid,
                            cost_multiplier=cost,
                            target_filter=ReversalFilter(selected, rule) if rule else None,
                        )
                        stats = summarize(replay, selected, cost, end)
                        stats["calendar_status"] = calendar_status
                        rows.append({**identity, **stats})
                        curves.append(replay["curve"].assign(**identity))
                        runs.append(
                            {
                                **identity,
                                "metrics": stats,
                                "trades": replay["trades"],
                                "research_events": replay["research_events"],
                                "unfilled_next_open": replay["unfilled_next_open"],
                                "canonical_snapshot_count": len(replay["decisions"]),
                            }
                        )
                print(f"completed {pool_name} {window}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "metrics.csv", index=False)
    all_curves = pd.concat(curves, ignore_index=True)
    all_curves.to_parquet(OUT / "curves.parquet", index=False)
    output = {"recipe": metadata, "verdict": verdict(table), "runs": runs}
    (OUT / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    write_report(table, all_curves, output)
    print(json.dumps(output["verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
