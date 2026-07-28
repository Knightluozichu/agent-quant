"""绘制七星V3策略全景图: 净值曲线 + 回撤 + 逐年收益 + 持仓时间线.

用法: uv run python scripts/plot_strategy.py
输出: data/qixing_results/strategy_overview.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# 中文字体 (macOS / Linux 兼容)
matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Noto Sans CJK SC",
]
matplotlib.rcParams["axes.unicode_minus"] = False

import run_qixing_v3 as rq  # noqa: E402

UP = "#e5484d"      # 红涨
DOWN = "#18a058"    # 绿跌

NAMES = {**rq.ETF_POOL, rq.DEFENSE: "货币基金"}
COLORS = {
    "518880": "#E6B800",  # 黄金
    "159985": "#A0522D",  # 豆粕
    "501018": "#2F4F4F",  # 原油
    "161226": "#9099A0",  # 白银
    "513100": "#3B6FE0",  # 纳指
    "159915": "#FF6347",  # 创业板
    "511220": "#2E9E5B",  # 城投债
    "511880": "#C8CDD3",  # 货币
}


def main() -> None:
    data = rq.load_data()
    result = rq.run_qixing_v3(data)
    eq = result["equity_curve"].copy()
    eq["trade_date"] = pd.to_datetime(eq["trade_date"])

    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1.2, 1.6, 1.8], hspace=0.32)

    # ---- Panel 1: 净值曲线 (对数) ----
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(eq["trade_date"], eq["equity"] / 10000, color=UP, linewidth=1.6)
    ax1.set_yscale("log")
    ax1.set_title(
        f"七星V3 净值曲线  |  累计 {result['total_return']*100:+.0f}%  "
        f"年化 {result['ann_return']*100:+.1f}%  夏普 {result['sharpe']:.2f}",
        fontsize=13, fontweight="bold",
    )
    ax1.set_ylabel("账户净值 (万元, 对数)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(10, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    # ---- Panel 2: 回撤 ----
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    cummax = eq["equity"].cummax()
    dd = (eq["equity"] - cummax) / cummax * 100
    ax2.fill_between(eq["trade_date"], dd, 0, color=DOWN, alpha=0.35)
    ax2.plot(eq["trade_date"], dd, color=DOWN, linewidth=0.7)
    ax2.set_ylabel("回撤 %")
    ax2.set_title(f"水下曲线 (最大回撤 {result['max_drawdown']*100:.1f}%)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: 逐年收益 ----
    ax3 = fig.add_subplot(gs[2])
    yearly = result["yearly"]
    years = sorted(yearly)
    rets = [yearly[y]["return"] * 100 for y in years]
    bar_colors = [UP if r >= 0 else DOWN for r in rets]
    bars = ax3.bar([str(y) for y in years], rets, color=bar_colors)
    ax3.set_title("逐年收益率", fontsize=11)
    ax3.set_ylabel("收益 %")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.grid(True, axis="y", alpha=0.3)
    for b, r in zip(bars, rets):
        ax3.text(b.get_x() + b.get_width() / 2, r + (3 if r >= 0 else -6),
                 f"{r:+.0f}%", ha="center", fontsize=9)

    # ---- Panel 4: 持仓时间线 ----
    ax4 = fig.add_subplot(gs[3])
    codes = list(NAMES.keys())
    code_to_y = {c: i for i, c in enumerate(codes)}
    for c in codes:
        mask = eq["holding"] == c
        if mask.any():
            ax4.scatter(eq.loc[mask, "trade_date"], [code_to_y[c]] * int(mask.sum()),
                        color=COLORS.get(c, "gray"), s=26, marker="s",
                        label=NAMES[c], edgecolors="none")
    ax4.set_yticks(range(len(codes)))
    ax4.set_yticklabels([f"{NAMES[c]} ({c})" for c in codes], fontsize=9)
    ax4.set_title("持仓时间线 (每点=一次调仓日持有的ETF)", fontsize=11)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax4.set_xlim(eq["trade_date"].min(), eq["trade_date"].max())
    ax4.grid(True, axis="x", alpha=0.3)
    ax4.legend(loc="upper left", ncol=4, fontsize=8, framealpha=0.8)

    span = (eq["trade_date"].max() - eq["trade_date"].min()).days / 365.25
    fig.suptitle(
        f"七星ETF轮动超级增强V3 — 策略全景 ({eq['trade_date'].min():%Y-%m} ~ "
        f"{eq['trade_date'].max():%Y-%m}, {span:.1f}年)",
        fontsize=16, fontweight="bold", y=0.995,
    )

    out = Path("data/qixing_results/strategy_overview.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  ✓ 已保存: {out.resolve()}")


if __name__ == "__main__":
    main()
