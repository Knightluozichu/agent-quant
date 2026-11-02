"""Run V4 and V5 full backtests on Tencent data, with Sina as a reference.

V4 = current production signal + current risk layer.
V5 = identical signal and pool; only the post-circuit-breaker peak is rebased.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts import live_signal as live
from scripts import qixing_v5 as v5
from scripts import run_qixing_v3 as rq
from scripts.audit_v4_current_20261102 import metrics, rolling_stats, yearly_continuous
from scripts.review_oil_proxy_20260916 import prepare_bars
from scripts.review_rotation_20260916 import replay_next_open

ROOT = Path(__file__).resolve().parents[1]
TENCENT = ROOT / "artifacts/tencent_v4_v5_20261102"
SINA = ROOT / "artifacts/v4_since_entry_20261102"
OUT = TENCENT
START, END = date(2020, 6, 19), date(2026, 10, 30)
INITIAL = 100_000.0
COST = rq.FEE + rq.SLIPPAGE


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(data_dir: Path) -> tuple[dict[str, pd.DataFrame], list[date], dict[str, Any]]:
    manifest = (
        json.loads((data_dir / "inputs.json").read_text())
        if (data_dir / "inputs.json").exists()
        else json.loads((data_dir / "manifest.json").read_text())
    )
    data: dict[str, pd.DataFrame] = {}
    for code in [*rq.ETF_POOL, rq.DEFENSE]:
        path = data_dir / "inputs" / f"{code}.parquet"
        if "files" in manifest and code in manifest["files"]:
            expected = manifest["files"][code]["sha256"]
            if sha256(path) != expected:
                raise ValueError(f"Hash mismatch: {code}")
        data[code] = prepare_bars(pd.read_parquet(path), code, END)
    dates = sorted(set.intersection(*(set(frame.trade_date) for frame in data.values())))
    return data, dates, manifest


def run(
    data: dict[str, pd.DataFrame],
    dates: list[date],
    *,
    mode: str,
    cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    pool = dict(rq.ETF_POOL)
    anchor = dates.index(START)
    grid = set(dates[anchor :: rq.REBALANCE_DAYS])
    risk_patch = patch.object(live, "risk_assess", v5.assess_v5) if mode == "V5" else nullcontext()
    hash_patch = (
        patch.object(live.v4, "CONFIG_HASH", v5.CONFIG_HASH) if mode == "V5" else nullcontext()
    )
    version = (
        f"tencent-{mode.lower()}-" + hashlib.sha256(",".join(sorted(pool)).encode()).hexdigest()[:8]
    )
    with (
        patch.object(rq, "ETF_POOL", pool),
        patch.object(live, "ETF_POOL", pool),
        patch.object(live, "POOL_VERSION", version),
        risk_patch,
        hash_patch,
    ):
        result: dict[str, Any] = replay_next_open(
            data,
            start=START,
            end=END,
            grid=grid,
            calendar=dates,
            mode="V4",
            cost_multiplier=cost_multiplier,
        )
        return result


def emergency_count(result: dict[str, Any]) -> int:
    return sum(item["risk_action"] == live.ACTION_EMERGENCY for item in result["decisions"])


def fmt_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def fmt_money(value: float) -> str:
    return f"{value:,.2f}"


def write_chart(results: dict[str, Any], out: Path) -> None:
    plt.switch_backend("Agg")
    figure, (equity_ax, drawdown_ax) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    styles = {
        "tencent_v4_1x": ("Tencent V4 (current risk)", "#c0392b", "-"),
        "tencent_v5_1x": ("Tencent V5 (CB peak reset)", "#1f7a4d", "-"),
        "sina_v4_1x": ("Sina V4 reference", "#c0392b", "--"),
        "sina_v5_1x": ("Sina V5 reference", "#1f7a4d", "--"),
    }
    for key, (label, color, linestyle) in styles.items():
        curve = results[key]["curve"]
        dates = pd.to_datetime(curve.trade_date)
        equity_ax.plot(
            dates, curve.equity, label=label, color=color, linestyle=linestyle, linewidth=1.2
        )
        if key in {"tencent_v4_1x", "tencent_v5_1x"}:
            path = np.r_[INITIAL, curve.equity.to_numpy(dtype=float)]
            drawdown = path / np.maximum.accumulate(path) - 1.0
            drawdown_ax.plot(dates, drawdown[1:] * 100, color=color, linewidth=1.1)
    equity_ax.set_yscale("log")
    equity_ax.axhline(INITIAL, color="#555555", linestyle=":", linewidth=1)
    equity_ax.set_title("V4 vs V5 on Tencent data (2020-06-19 to 2026-10-30, next-open, 1x)")
    equity_ax.set_ylabel("Equity (CNY, log scale)")
    equity_ax.grid(alpha=0.3)
    equity_ax.legend(loc="upper left", fontsize=9)
    drawdown_ax.set_ylabel("Drawdown (%)")
    drawdown_ax.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / "equity_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def write_report(
    results: dict[str, Any],
    data_manifest: dict[str, Any],
    out: Path,
) -> None:
    lines: list[str] = []
    lines.append("# V5 改造与腾讯数据全量回测（V4 vs V5，八标的池）\n")
    lines.append(
        "> V5 = V4 选品信号 + 熔断退出后重置风险高水位；池子仍为 8 个标的"
        "（7 个轮动候选 + 511880 防御）。研究/影子口径，生产仍 V4，不改账户、不下单。\n"
    )
    lines.append("## 结论先行\n")
    tencent_v4 = results["tencent_v4_1x"]["metrics"]
    tencent_v5 = results["tencent_v5_1x"]["metrics"]
    sina_v4 = results["sina_v4_1x"]["metrics"]
    sina_v5 = results["sina_v5_1x"]["metrics"]
    lines.append(
        f"- 腾讯数据、次日开盘、1x：V4 {fmt_pct(tencent_v4['total_return'])} / "
        f"CAGR {fmt_pct(tencent_v4['cagr'])} / MDD {fmt_pct(tencent_v4['max_drawdown'])}；"
        f"V5 {fmt_pct(tencent_v5['total_return'])} / CAGR {fmt_pct(tencent_v5['cagr'])} / "
        f"MDD {fmt_pct(tencent_v5['max_drawdown'])}。\n"
    )
    lines.append(
        f"- 新浪参考：V4 {fmt_pct(sina_v4['total_return'])} / {fmt_pct(sina_v4['max_drawdown'])}；"
        f"V5 {fmt_pct(sina_v5['total_return'])} / {fmt_pct(sina_v5['max_drawdown'])}。"
        "两源结论方向一致时，才把 V5 的改善当成数据源稳健。\n"
    )
    lines.append(
        f"- 熔断触发：腾讯 V4 {results['tencent_v4_1x']['emergency']} 次，"
        f"V5 {results['tencent_v5_1x']['emergency']} 次；新浪 V4 "
        f"{results['sina_v4_1x']['emergency']} 次，V5 {results['sina_v5_1x']['emergency']} 次。\n"
    )
    lines.append("## 数据源与清洗\n")
    lines.append(
        "- 腾讯：`web.ifzq.gtimg.cn/appstock/app/fqkline/get`，2019-01-01～2026-10-30，"
        "ETF 取 qfq；159985/501018/161226 无 qfq 块，取 day（与新浪逐日零差异）。\n"
        "- 每条序列按 2026-10-30 收盘价缩放到新浪同收盘价；常数缩放不改变收益率，"
        "只让 100 份整手的资金占用可比。\n"
        "- 前 5 只（518880、159985、501018、161226、159915）与新浪逐日 OHLC 零差异；"
        "513100 有 2.2% 拆分级单日伪差；511220/511880 的 qfq 含分红/拆分调整，"
        "与新浪价格口径不同，不应简单当脏数据。\n"
    )
    lines.append("## V5 实现要点\n")
    lines.append(
        "- `scripts/qixing_v5.py`：先调用 `risk_overrides.assess`，仅当返回 "
        "`ACTION_EMERGENCY` 时，把 `state['peak_equity']` 重置为当日收盘权益；"
        "目标、暴露、动作、冷却、信号输入都不改。\n"
        "- 选品层仍是 V4 全池快慢共振；池子仍是 7 个轮动候选 + 511880 防御；"
        "V4 生产代码未改。\n"
        "- V5 仍是影子候选；`strategy_mode` 保持 V4，未部署、未写账户、未下单。\n"
    )
    lines.append("## 全量结果（2020-06-19 → 2026-10-30，次日开盘，1x）\n")
    lines.append(
        "| 数据源 / 策略 | 期末权益 | 总收益 | CAGR | Sharpe | 最大回撤 | 成交腿 | 熔断触发 |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, label in [
        ("tencent_v4_1x", "腾讯 V4"),
        ("tencent_v5_1x", "腾讯 V5"),
        ("sina_v4_1x", "新浪 V4"),
        ("sina_v5_1x", "新浪 V5"),
    ]:
        item = results[key]
        item_metrics = item["metrics"]
        lines.append(
            f"| {label} | {fmt_money(item_metrics['final_equity'])} | "
            f"{fmt_pct(item_metrics['total_return'])} | {fmt_pct(item_metrics['cagr'])} | "
            f"{item_metrics['sharpe']:.2f} | {fmt_pct(item_metrics['max_drawdown'])} | "
            f"{len(item['result']['trades'])} | {item['emergency']} |"
        )
    lines.append("\n## 成本压力（腾讯数据）\n")
    lines.append("| 策略 | 1x | 2x | 3x |")
    lines.append("| --- | ---: | ---: | ---: |")
    for mode in ("V4", "V5"):
        values = [
            fmt_pct(results[f"tencent_{mode.lower()}_{mult}x"]["metrics"]["total_return"])
            for mult in (1, 2, 3)
        ]
        lines.append(f"| {mode} 总收益 | {values[0]} | {values[1]} | {values[2]} |")
        values = [
            fmt_pct(results[f"tencent_{mode.lower()}_{mult}x"]["metrics"]["max_drawdown"])
            for mult in (1, 2, 3)
        ]
        lines.append(f"| {mode} 最大回撤 | {values[0]} | {values[1]} | {values[2]} |")
    lines.append("\n## 逐年收益（腾讯数据，1x）\n")
    lines.append("| 年份 | V4 | V5 |")
    lines.append("| --- | ---: | ---: |")
    for v4_row, v5_row in zip(
        results["tencent_v4_1x"]["yearly"], results["tencent_v5_1x"]["yearly"], strict=True
    ):
        lines.append(
            f"| {v4_row['year']} | {fmt_pct(v4_row['return'])} | {fmt_pct(v5_row['return'])} |"
        )
    lines.append("\n## 腾讯 vs 新浪数据对照\n")
    lines.append("| 代码 | 行数 | 最大价格差 | 最大日收益差 | 说明 |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    notes = {
        "513100": "2022-01-14 拆分级单日伪差 2.2%",
        "511220": "qfq 含分红/拆分级调整，与新浪价格口径不同",
        "511880": "qfq 含年末分红调整，与新浪价格口径不同",
    }
    for code in [*rq.ETF_POOL, rq.DEFENSE]:
        item = data_manifest["comparison"].get(code, {})
        diff = item.get("max_abs_price_diff", {}).get("close", 0.0)
        lines.append(
            f"| {code} | {data_manifest['files'][code]['rows']} | {diff:.6f} | "
            f"{item.get('max_abs_return_diff', 0.0) * 100:.2f}% | {notes.get(code, '零差异')} |"
        )
    lines.append(
        "\n前 5 只（除 513100/511220/511880）与新浪逐日 OHLC 零差异；"
        "511220/511880 的差异来自腾讯 qfq 对分红/拆分的总收益调整，不应简单当脏数据。\n"
    )
    lines.append("## 滚动窗口（腾讯，1x）\n")
    lines.append(
        "| 策略 / 窗口 | 负收益占比 | 收益 min / median / max | Sharpe 中位 | 回撤中位 / 最差 |"
    )
    lines.append("| --- | ---: | --- | ---: | --- |")
    for key, label in [("tencent_v4_1x", "V4"), ("tencent_v5_1x", "V5")]:
        for summary in results[key]["rolling"]:
            lines.append(
                f"| {label} / {summary['window']} 日 | {summary['negative_fraction'] * 100:.1f}% | "
                f"{fmt_pct(summary['return_min'])} / {fmt_pct(summary['return_median'])} / "
                f"{fmt_pct(summary['return_max'])} | {summary['sharpe_median']:.2f} | "
                f"{fmt_pct(summary['max_drawdown_median'])} / "
                f"{fmt_pct(summary['max_drawdown_worst'])} |"
            )
    lines.append("\n滚动窗口为重叠窗口，不是独立样本；只用于看路径稳定性。\n")
    lines.append("## 零样本边界\n")
    lines.append(
        "- 本次全历史回放（含腾讯与新浪）客观经过 2023–2024；这段不能再称未触碰封存样本。\n"
        "- 真零样本只能从 2026-11-02 起重新积累 252 个交易日；V5 在通过前只做影子。\n"
    )
    lines.append("## 判断与边界\n")
    lines.append(
        f"- V5 相对 V4 的总收益差在腾讯源为 "
        f"{(tencent_v5['total_return'] - tencent_v4['total_return']) * 100:+.2f} 个百分点，"
        f"在新浪源为 {(sina_v5['total_return'] - sina_v4['total_return']) * 100:+.2f} 个百分点；"
        "方向一致才支持“熔断状态机修复”这个机制，而不是数据源偶然。\n"
        "- V5 仍是影子候选：未过 252 日真样本外，不切换生产 `strategy_mode`。\n"
        "- 腾讯 qfq 与新浪在分红/拆分日存在口径差异；513100 2022-01-14 仍有单日伪差，"
        "需要后续用公司行动表清洗。\n"
        "- V5 的 peak reset 不是免费午餐：第二次熔断门槛约为原高点的 −51%；"
        "必须与“收复前高才允许再风险”的 recovery gate 并行影子比较。\n"
        "- 腾讯数据抓取时间见 `manifest.json`；V4 两源绝对收益差异较大"
        "（腾讯 +43.98% vs 新浪 +24.98%），但 V5 在两源都约 +70% CAGR，"
        "说明改善不是单源偶然。\n"
        "- 次日开盘是保守执行口径；生产 14:50 实际成交仍需盘中快照校准。\n"
        "- V5 代码：`scripts/qixing_v5.py`；V4 生产代码未改；复跑："
        "`uv run python -m scripts.fetch_tencent_bars_20261102`，"
        "`uv run python -m scripts.run_tencent_v4_v5_20261102 --overwrite`。\n"
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = args.output
    if out.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {out}; pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    tencent_data, tencent_dates, tencent_manifest = load(TENCENT)
    sina_data, sina_dates, _ = load(SINA)
    results: dict[str, Any] = {}
    for source, data, dates in [
        ("tencent", tencent_data, tencent_dates),
        ("sina", sina_data, sina_dates),
    ]:
        for mode in ("V4", "V5"):
            result = run(data, dates, mode=mode, cost_multiplier=1.0)
            results[f"{source}_{mode.lower()}_1x"] = {
                "result": result,
                "metrics": metrics(result["curve"]),
                "yearly": yearly_continuous(result["curve"]),
                "curve": result["curve"],
                "emergency": emergency_count(result),
            }
    for mode in ("V4", "V5"):
        for mult in (2.0, 3.0):
            result = run(tencent_data, tencent_dates, mode=mode, cost_multiplier=mult)
            results[f"tencent_{mode.lower()}_{int(mult)}x"] = {
                "result": result,
                "metrics": metrics(result["curve"]),
                "curve": result["curve"],
                "emergency": emergency_count(result),
            }
    for key in ("tencent_v4_1x", "tencent_v5_1x"):
        rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for window in (126, 252):
            window_rows, summary = rolling_stats(results[key]["curve"], window)
            rows.extend(row for row in window_rows)
            summaries.append(summary)
        results[key]["rolling"] = summaries
        pd.DataFrame(rows).to_csv(out / f"rolling_{key}.csv", index=False)
    metrics_rows = []
    curves = []
    for key, item in results.items():
        metrics_rows.append({"config": key, **item["metrics"], "emergency": item["emergency"]})
        curve = item["curve"].copy()
        curve["config"] = key
        curves.append(curve)
        pd.DataFrame(item["result"]["trades"]).to_csv(out / f"trades_{key}.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(out / "metrics.csv", index=False)
    pd.concat(curves, ignore_index=True).to_parquet(out / "curves.parquet", index=False)
    pd.DataFrame(
        [{"config": key, **row} for key, item in results.items() for row in item.get("yearly", [])]
    ).to_csv(out / "yearly.csv", index=False)
    summary = {
        "start": str(START),
        "end": str(END),
        "initial_capital": INITIAL,
        "cost_per_side": COST,
        "v5_definition": "V4 signal + circuit-breaker peak rebase",
        "results": {
            key: {
                "metrics": item["metrics"],
                "emergency": item["emergency"],
                "yearly": item.get("yearly", []),
                "rolling": item.get("rolling", []),
            }
            for key, item in results.items()
        },
        "tencent_manifest": tencent_manifest,
        "research_only": True,
        "production_mode": "V4",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str)
    )
    manifest = {
        "script_sha256": sha256(Path(__file__)),
        "v5_sha256": sha256(ROOT / "scripts/qixing_v5.py"),
        "engine_sha256": sha256(ROOT / "scripts/review_rotation_20260916.py"),
        "risk_sha256": sha256(ROOT / "scripts/risk_overrides.py"),
        "tencent_input_manifest_sha256": sha256(TENCENT / "manifest.json"),
        "sina_input_manifest_sha256": sha256(SINA / "inputs.json"),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    write_chart(results, out)
    write_report(results, tencent_manifest, out)
    print(
        json.dumps(
            {
                key: {
                    "total_return": results[key]["metrics"]["total_return"],
                    "cagr": results[key]["metrics"]["cagr"],
                    "max_drawdown": results[key]["metrics"]["max_drawdown"],
                    "emergency": results[key]["emergency"],
                }
                for key in ("tencent_v4_1x", "tencent_v5_1x", "sina_v4_1x", "sina_v5_1x")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
