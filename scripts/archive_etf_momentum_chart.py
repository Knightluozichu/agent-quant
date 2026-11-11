"""Read-only 30-session ETF momentum / daily-return chart.

Example: uv run python scripts/archive_etf_momentum_chart.py --data-dir <snapshot/bars>
    --cutoff 2026-10-19 --output-dir <report-directory>
No quotes, accounts, strategies or production caches are modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import run_qixing_v3 as rq

NAMES = {**rq.ETF_POOL, rq.DEFENSE: "货币ETF（防御）"}


def build_view(bars: pd.DataFrame, *, cutoff: pd.Timestamp, days: int = 30) -> pd.DataFrame:
    """Return a long, aligned table using only closes at or before the cutoff.

    The union calendar catches per-asset gaps; it is not an independent exchange
    calendar. Each displayed return and momentum uses the same price series.
    """
    if days < 1 or bars.empty:
        raise ValueError("days must be positive and history nonempty")
    frame = bars[["code", "trade_date", "close"]].copy()
    frame["trade_date"] = pd.to_datetime(frame.trade_date)
    frame = frame[frame.trade_date <= cutoff].sort_values(["code", "trade_date"])
    if frame.empty or frame.trade_date.max() != cutoff:
        raise ValueError("cutoff must be an observed completed trading date")
    if frame.duplicated(["code", "trade_date"]).any():
        raise ValueError("duplicate code/date in input")
    frame["close"] = pd.to_numeric(frame.close, errors="raise")
    if not np.isfinite(frame.close.to_numpy()).all() or (frame.close <= 0).any():
        raise ValueError("prices must be positive finite values")
    dates = pd.DatetimeIndex(frame.trade_date.unique()).sort_values()
    warmup = max(rq.MOM_PERIODS)
    if len(dates) < days + warmup:
        raise ValueError("insufficient history for displayed days plus momentum warmup")
    required = dates[-(days + warmup) :]
    result = []
    for _, group in frame.groupby("code", sort=False):
        if not required.isin(group.trade_date).all():
            raise ValueError("missing trading date in asset history")
        group = group.copy()
        group["daily_return"] = group.close.pct_change(fill_method=None)
        group["momentum"] = sum(
            weight * (group.close / group.close.shift(period) - 1)
            for period, weight in zip(rq.MOM_PERIODS, rq.MOM_WEIGHTS, strict=True)
        )
        result.append(group.tail(days))
    return pd.concat(result, ignore_index=True)


def render(view: pd.DataFrame, output: Path) -> None:
    """Draw two shared-date heatmaps; never label contemporaneous returns as P&L."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Rectangle

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    codes = [code for code in NAMES if code in set(view.code)]
    dates = pd.DatetimeIndex(sorted(view.trade_date.unique()))
    fig, axes = plt.subplots(2, 1, figsize=(23, 10.8), layout="constrained")
    fig.set_facecolor("#fafbfd")
    cmap = LinearSegmentedColormap.from_list("china_returns", ["#13836d", "#ffffff", "#d3464c"])
    fields = [
        ("momentum", "01   收盘动量｜50% × 10日收益 + 50% × 20日收益", 1),
        ("daily_return", "02   对应同一天的ETF涨跌｜当日收盘 / 前一交易日收盘 − 1", 2),
    ]
    momentum = view.pivot(index="code", columns="trade_date", values="momentum")
    risk_codes = [code for code in rq.ETF_POOL if code in codes]
    leaders = [
        risk_codes[int(index)]
        for index in np.argmax(momentum.reindex(index=risk_codes, columns=dates).to_numpy(), axis=0)
    ]
    for ax, (field, title, precision) in zip(axes, fields, strict=True):
        table = (
            view.pivot(index="code", columns="trade_date", values=field).reindex(
                index=codes,
                columns=dates,
            )
            * 100
        )
        values = table.to_numpy()
        bound = max(float(np.abs(values).max()), 0.01)
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
        image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
        for row, code in enumerate(codes):
            for col in range(len(dates)):
                value = values[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:+.{precision}f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > bound * 0.62 else "#273344",
                )
                if code == leaders[col]:
                    ax.add_patch(
                        Rectangle(
                            (col - 0.47, row - 0.46),
                            0.94,
                            0.92,
                            fill=False,
                            edgecolor="#1b335a",
                            linewidth=1.7,
                        )
                    )
        ax.set_yticks(range(len(codes)), [f"{NAMES[code]}  {code}" for code in codes], fontsize=10)
        ax.set_xticks(
            range(len(dates)),
            [day.strftime("%m/%d") for day in dates],
            rotation=55,
            ha="right",
            fontsize=9,
        )
        ax.set_xticks(np.arange(-0.5, len(dates), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(codes), 1), minor=True)
        ax.grid(which="minor", color="#e5e9ef", linewidth=0.6)
        ax.tick_params(which="both", length=0)
        ax.set_title(title + "（%）", loc="left", fontsize=14, pad=16, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(image, ax=ax, fraction=0.014, pad=0.012, label="%")
    fig.suptitle(
        f"ETF动量 × 每日涨跌  |  最近{len(dates)}个完整交易日\n"
        f"{dates[0]:%Y-%m-%d} — {dates[-1]:%Y-%m-%d}  ·  7只轮动资产（含原油）+ 1只防御资产",
        fontsize=20,
        fontweight="bold",
        color="#172842",
    )
    fig.supxlabel(
        "红涨绿跌；深色边框＝当天7只轮动资产中原始动量最高者（未应用买入门禁）\n"
        "上下为同日对照：动量包含当天收盘信息，不是盘前信号、预期收益或策略实盘收益；未计费用。",
        fontsize=10,
        color="#526176",
    )
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=pd.Timestamp, required=True)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    for code in NAMES:
        frame = pd.read_parquet(args.data_dir / f"{code}.parquet")
        frame["code"] = code
        frames.append(frame)
    view = build_view(pd.concat(frames, ignore_index=True), cutoff=args.cutoff, days=args.days)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    view.to_csv(args.output_dir / "daily_momentum_returns.csv", index=False, float_format="%.10f")
    render(view, args.output_dir / "momentum_daily_returns.png")
    print(f"{view.trade_date.min():%Y-%m-%d} — {view.trade_date.max():%Y-%m-%d}; {len(view)} rows")
    print(args.output_dir.resolve() / "momentum_daily_returns.png")


if __name__ == "__main__":
    main()
