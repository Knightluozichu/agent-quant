"""实验: 结构交易 (隔日/缠论) 全量回测 — 预注册研究, 不触达生产.

预注册假设 (2026-10-02, 改参数即作废重注册):
  H1 (隔日): 日线可观测的"突破跟随/强势回踩"两类买点, T 收盘确认、T+1 开盘进、
      T+2 了结 (封板延续除外), 在 ETF 池全历史上, 扣除 1x 成本后期望为正;
      证伪标准: 1x 成本利润因子 ≤ 1 或样本 < 30 笔.
  H2 (缠论): 以日线为操作级别、日线笔为次级别段, 一买/二买/三买结构入场,
      退出只由结构死亡或卖点代理触发, 扣 1x 成本后期望为正;
      证伪标准: 同上. 3x 成本下不失控 (无路径悬崖).

数据与代理限制 (如实披露):
  - 无个股、无分时、无板块热点数据: 隔日模式只能代理验证"手法骨架",
    选股充分条件 (热点最强一只) 无法验证, 结论不能外推到个股超短.
  - 隔日买点 3 (二次放量) 需要开盘量, 日线数据无法构造, 本实验不测;
    买点 4 (飞刀) 按 skill 默认关闭.
  - 缠论操作级别取日线, 次级别段取日线笔 (实战通用简化, 未构线段);
    力度 = 段幅度 + MACD(12,26,9) 柱面积, 双确认才算背驰;
    区间套收不到日线以下, 一买精度限于日线笔.
  - 结构死亡的日线代理: 三买 close<ZG / 一二买 close<一买低点.
  - 因果处理: 包含合并在线进行; 分型在中间 K 右侧第二根 K 追加时才确认
    (右邻先定型), 确认前结构不可见; 信号 T 收盘确认、T+1 开盘成交.
  - 夏普按成交笔权益步长计算, 非逐日, 仅作近似.

执行: uv run python scripts/exp_structure_trade.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_qixing_v3 as rq

PROJECT_ROOT = Path(__file__).parent.parent
RESULT_DIR = PROJECT_ROOT / "data" / "qixing_results"
REPORT_DIR = PROJECT_ROOT / "reports"

COST_SIDE = rq.FEE + rq.SLIPPAGE  # 0.0015 单边
MIN_BI_GAP = 4  # 新笔规则: 顶底极值 K 之间 (不含两端) ≥3 根 → 索引差 ≥4
GERI_PLATFORM = 20  # 突破平台回看
GERI_VOL_MULT = 1.5  # 突破量能倍数
GERI_NO_CHASE = 1.03  # T+1 开盘超 T 收盘 3% 不追
GERI_PULLBACK_STRENGTH = 0.15  # 20 日强势阈值
GERI_PULLBACK_VOL = 0.7  # 缩量阈值 (对 5 日均量)
GERI_LIMIT = {"159915": 0.198}  # 创业板 ETF 20% 板, 其余 10%
DEFAULT_LIMIT = 0.098
LIMIT_HOLD_VOL = 1.2  # 封板延续量能阈值 (对 5 日均量)


# --------------------------------------------------------------------------- #
# 几何基元
# --------------------------------------------------------------------------- #
@dataclass
class Bar:
    idx: int
    open: float
    high: float
    low: float
    close: float
    i0: int = -1  # 原始 K 线跨度 (含包含合并)
    i1: int = -1

    def __post_init__(self):
        if self.i0 < 0:
            self.i0 = self.i1 = self.idx


@dataclass
class Fractal:
    kind: str  # "top" | "bottom"
    idx: int  # 合并后 K 线索引 (分型中间那根)
    price: float  # 极值


@dataclass
class Bi:
    direction: str  # "up" | "down"
    start_idx: int  # 合并 K 线索引
    end_idx: int
    start_price: float
    end_price: float
    start_orig: int = -1
    end_orig: int = -1

    @property
    def lo(self) -> float:
        return min(self.start_price, self.end_price)

    @property
    def hi(self) -> float:
        return max(self.start_price, self.end_price)

    @property
    def amplitude(self) -> float:
        return abs(self.end_price / self.start_price - 1)


@dataclass
class Zhongshu:
    zg: float
    zd: float
    gg: float
    dd: float
    start_bi: int
    end_bi: int


def merge_inclusion(bars: list[Bar]) -> list[Bar]:
    """包含关系合并 (reference.md: 方向由前一根非包含 K 线决定). 在线算法."""
    merged: list[Bar] = []
    for b in bars:
        merged_append(merged, Bar(0, b.open, b.high, b.low, b.close, b.i0, b.i1))
    for i, m in enumerate(merged):
        m.idx = i
    return merged


def merged_append(merged: list[Bar], nb: Bar) -> bool:
    """把一根原始 K 喂进合并序列; 返回是否追加 (False = 被末根吸收)."""
    if not merged:
        nb.idx = 0
        merged.append(nb)
        return True
    last = merged[-1]
    included = (last.high >= nb.high and last.low <= nb.low) or (
        last.high <= nb.high and last.low >= nb.low
    )
    if included:
        up = last.high >= merged[-2].high if len(merged) >= 2 else nb.close >= last.open
        if up:
            last.high = max(last.high, nb.high)
            last.low = max(last.low, nb.low)
        else:
            last.high = min(last.high, nb.high)
            last.low = min(last.low, nb.low)
        last.close = nb.close
        last.i1 = nb.i1
        return False
    nb.idx = len(merged)
    merged.append(nb)
    return True


def find_fractals(merged: list[Bar]) -> list[Fractal]:
    """标准三分型 (批量, 用于测试与合成校验; 实盘引擎见 ChanEngine)."""
    out: list[Fractal] = []
    for i in range(1, len(merged) - 1):
        a, b, c = merged[i - 1], merged[i], merged[i + 1]
        if b.high > a.high and b.high > c.high and b.low > a.low and b.low > c.low:
            out.append(Fractal("top", i, b.high))
        elif b.low < a.low and b.low < c.low and b.high < a.high and b.high < c.high:
            out.append(Fractal("bottom", i, b.low))
    return out


def _fractal_at(a: Bar, b: Bar, c: Bar) -> Fractal | None:
    if b.high > a.high and b.high > c.high and b.low > a.low and b.low > c.low:
        return Fractal("top", b.idx, b.high)
    if b.low < a.low and b.low < c.low and b.high < a.high and b.high < c.high:
        return Fractal("bottom", b.idx, b.low)
    return None


def build_bi(merged: list[Bar], fractals: list[Fractal]) -> list[Bi]:
    """交替分型连笔; 极值 K 间 (不含两端) 至少 3 根 → 索引差 ≥ MIN_BI_GAP."""
    if not fractals:
        return []
    kept: list[Fractal] = [fractals[0]]
    for f in fractals[1:]:
        last = kept[-1]
        if f.kind == last.kind:
            if (f.kind == "top" and f.price > last.price) or (
                f.kind == "bottom" and f.price < last.price
            ):
                kept[-1] = f
        elif f.idx - last.idx >= MIN_BI_GAP:
            kept.append(f)
        # 反向分型但距离不足: 忽略, 等更极端的同向分型替换锚点
    bis: list[Bi] = []
    for a, b in pairwise(kept):
        direction = "up" if a.kind == "bottom" else "down"
        bis.append(
            Bi(direction, a.idx, b.idx, a.price, b.price,
               merged[a.idx].i0, merged[b.idx].i1)
        )
    return bis


def build_zhongshu(bis: list[Bi]) -> list[Zhongshu]:
    """中枢 = 连续三笔价格重叠; ZG=三高最低, ZD=三低最高, ZD<ZG 才成立."""
    out: list[Zhongshu] = []
    i = 0
    while i + 2 < len(bis):
        trio = bis[i : i + 3]
        zg = min(b.hi for b in trio)
        zd = max(b.lo for b in trio)
        if zd < zg:
            j = i + 2
            gg = max(b.hi for b in trio)
            dd = min(b.lo for b in trio)
            while j + 1 < len(bis):
                nb = bis[j + 1]
                if nb.lo < zg and nb.hi > zd:  # 仍重叠 → 延伸
                    j += 1
                    gg = max(gg, nb.hi)
                    dd = min(dd, nb.lo)
                else:
                    break
            out.append(Zhongshu(zg, zd, gg, dd, i, j))
            i = j + 1
        else:
            i += 1
    return out


# --------------------------------------------------------------------------- #
# 缠论流式引擎 (因果: 右邻定型后才确认分型, 确认笔列表只增不改)
# --------------------------------------------------------------------------- #
def macd_hist(close: pd.Series) -> pd.Series:
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    return (dif - dea) * 2


class ChanEngine:
    """逐 K 喂入. 新 K 追加 (未被吸收) 时, 倒数第 2 根定型, 确认倒数第 3 根处分型."""

    def __init__(self) -> None:
        self.merged: list[Bar] = []
        self._raw_count = 0
        self._anchor: Fractal | None = None
        self.bis: list[Bi] = []
        self.zhongshu: list[Zhongshu] = []

    def feed(self, o: float, h: float, l: float, c: float) -> None:
        nb = Bar(0, o, h, l, c, self._raw_count, self._raw_count)
        self._raw_count += 1
        appended = merged_append(self.merged, nb)
        if not appended:
            return  # 末根仍可变, 不确认任何分型
        m = self.merged
        p = len(m) - 3  # p+1 刚定型 → p 处分型可确认
        if p < 1:
            return
        f = _fractal_at(m[p - 1], m[p], m[p + 1])
        if f is not None:
            self._on_fractal(f)

    def _on_fractal(self, f: Fractal) -> None:
        a = self._anchor
        if a is None:
            self._anchor = f
            return
        if f.kind == a.kind:
            if (f.kind == "top" and f.price > a.price) or (
                f.kind == "bottom" and f.price < a.price
            ):
                self._anchor = f
            return
        if f.idx - a.idx < MIN_BI_GAP:
            return  # 距离不足, 不成笔, 锚点不动
        self.bis.append(
            Bi("up" if a.kind == "bottom" else "down", a.idx, f.idx, a.price, f.price,
               self.merged[a.idx].i0, self.merged[f.idx].i1)
        )
        self._anchor = f
        self.zhongshu = build_zhongshu(self.bis)


def _bi_area(hist: np.ndarray, bi: Bi, sign: int) -> float:
    """笔跨度内 MACD 柱面积 (sign=+1 红柱, -1 绿柱, 取绝对值)."""
    seg = hist[bi.start_orig : bi.end_orig + 1]
    seg = seg[np.sign(seg) == sign]
    return float(np.abs(seg).sum())


# --------------------------------------------------------------------------- #
# 缠论回测 (单标的状态机)
# --------------------------------------------------------------------------- #
def run_chan_on_series(df: pd.DataFrame, cost: float = COST_SIDE) -> list[dict]:
    df = df.sort_values("trade_date").reset_index(drop=True)
    hist = macd_hist(df["close"]).to_numpy()
    eng = ChanEngine()
    pos: dict | None = None
    trades: list[dict] = []
    buy1: dict | None = None  # 最近一买点 {"low": float, "bi": int}
    last_up: Bi | None = None
    seen_third_buy_zs = -1

    for i in range(len(df)):
        r = df.iloc[i]
        eng.feed(float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]))
        td = str(r["trade_date"])[:10]
        nxt = i + 1
        if nxt >= len(df):
            break

        # ---- 持仓退出检查 (只用 T 收盘已确认事实) ----
        if pos is not None:
            exit_reason = ""
            if pos["buy_type"] == "third_buy" and float(r["close"]) < pos["zg"]:
                exit_reason = "三买死亡: 收回中枢上沿之下"
            elif pos["buy_type"] in ("first_buy", "second_buy") and float(r["close"]) < pos["buy1_low"]:
                exit_reason = "买点结构死亡: 跌破一买低点"
            if not exit_reason and len(eng.bis) > pos["bi_count_at_entry"]:
                latest = eng.bis[-1]
                if latest.direction == "up":
                    weaker_high = last_up is not None and latest.end_price < last_up.end_price
                    weaker_area = (
                        last_up is not None
                        and _bi_area(hist, latest, +1) < _bi_area(hist, last_up, +1)
                    )
                    if weaker_high or weaker_area:
                        exit_reason = "走完: 次级别向上不创新高或盘整背驰"
                    else:
                        last_up = latest
            if exit_reason:
                ep = float(df["open"].iloc[nxt]) * (1 - cost)
                trades.append({
                    **pos,
                    "exit_date": str(df["trade_date"].iloc[nxt])[:10],
                    "exit_price": ep,
                    "exit_reason": exit_reason,
                    "ret": ep / pos["entry_price"] - 1,
                })
                pos = None
            continue

        # ---- 空仓入场检查 ----
        bis, zss = eng.bis, eng.zhongshu
        if not bis:
            continue
        latest = bis[-1]
        signal: dict | None = None

        # 一买: 两个不重叠下跌中枢 (后 GG < 前 DD) + 离开段 c 背驰连接段 b
        if latest.direction == "down" and len(zss) >= 2:
            za, zb = zss[-2], zss[-1]
            if zb.gg < za.dd and latest.end_price < zb.zd and latest.start_idx >= zb.start_bi:
                conn = [b for b in bis[za.end_bi : zb.start_bi + 1] if b.direction == "down"]
                if conn:
                    b_seg = max(conn, key=lambda b: b.amplitude)
                    if latest.amplitude < b_seg.amplitude and _bi_area(
                        hist, latest, -1
                    ) < _bi_area(hist, b_seg, -1):
                        signal = {"buy_type": "first_buy", "buy1_low": latest.end_price}
                        buy1 = {"low": latest.end_price, "bi": len(bis) - 1}

        # 二买: 一买后向上笔完成, 回抽笔低点不破一买
        if (
            signal is None
            and buy1 is not None
            and latest.direction == "down"
            and len(bis) - 1 > buy1["bi"] + 1
            and latest.end_price > buy1["low"]
            and any(b.direction == "up" for b in bis[buy1["bi"] + 1 :])
        ):
            signal = {"buy_type": "second_buy", "buy1_low": buy1["low"]}

        # 三买: 中枢上最后一根端点突破 ZG 的向上笔 + 回抽笔低点不跌破 ZG (每中枢只做一次)
        if signal is None and zss and latest.direction == "down":
            zb = zss[-1]
            leave = [
                b for b in bis[zb.start_bi : len(bis) - 1]
                if b.direction == "up" and b.end_price > zb.zg and b.start_price <= zb.zg
            ]
            if (
                leave
                and latest.start_idx >= leave[-1].end_idx
                and latest.end_price > zb.zg
                and seen_third_buy_zs != zb.start_bi
            ):
                signal = {"buy_type": "third_buy", "zg": zb.zg, "zd": zb.zd}
                seen_third_buy_zs = zb.start_bi

        if signal is not None:
            ep = float(df["open"].iloc[nxt]) * (1 + cost)
            pos = {
                **signal,
                "signal_date": td,
                "entry_date": str(df["trade_date"].iloc[nxt])[:10],
                "entry_price": ep,
                "bi_count_at_entry": len(bis),
            }
            last_up = next((b for b in reversed(bis) if b.direction == "up"), None)

    if pos is not None:  # 期末未平: 按最后收盘记 (披露为强迫平仓)
        ep = float(df["close"].iloc[-1]) * (1 - cost)
        trades.append({
            **pos,
            "exit_date": str(df["trade_date"].iloc[-1])[:10],
            "exit_price": ep,
            "exit_reason": "期末强平 (研究口径)",
            "ret": ep / pos["entry_price"] - 1,
        })
    return trades


# --------------------------------------------------------------------------- #
# 隔日回测 (日线代理)
# --------------------------------------------------------------------------- #
def run_geri_on_series(
    df: pd.DataFrame,
    cost: float = COST_SIDE,
    env_df: pd.DataFrame | None = None,
    code: str = "",
) -> list[dict]:
    df = df.sort_values("trade_date").reset_index(drop=True).copy()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["vma5"] = df["volume"].rolling(5).mean()
    df["vma20"] = df["volume"].rolling(20).mean()
    df["plat_high"] = df["high"].shift(1).rolling(GERI_PLATFORM).max()
    df["prev_close"] = df["close"].shift(1)
    df["close_20ago"] = df["close"].shift(20)
    df["max_close_20"] = df["close"].shift(1).rolling(20).max()

    if env_df is not None and not env_df.empty:
        e = env_df.sort_values("trade_date")[["trade_date", "close"]].copy()
        e["env_ma60"] = e["close"].rolling(60).mean()
        e["env_ret5"] = e["close"].pct_change(5)
        df = df.merge(e[["trade_date", "env_ma60", "env_ret5"]], on="trade_date", how="left")
        env_bear = df["env_ma60"].notna() & (df["close"] < df["env_ma60"])
        env_crash = df["env_ret5"] <= -0.07
    else:
        env_bear = pd.Series(False, index=df.index)
        env_crash = pd.Series(False, index=df.index)

    limit = GERI_LIMIT.get(code, DEFAULT_LIMIT)
    pos: dict | None = None
    trades: list[dict] = []

    for i in range(GERI_PLATFORM + 1, len(df)):
        r = df.iloc[i]
        td = str(r["trade_date"])[:10]
        nxt = i + 1

        # ---- 持仓管理: T+1 收盘判"证明买错", T+2 了结, 封板延续 ----
        if pos is not None:
            d = i - pos["entry_i"]
            if d <= 0:
                continue
            if d == 1:  # T+1 收盘评估预期
                wrong = False
                if pos["buy_type"] == "breakout" and float(r["close"]) < pos["plat_high"]:
                    wrong = True  # 假突破: 收回平台
                if float(r["high"]) > pos["sig_high"] * 1.02 and float(r["volume"]) < pos["sig_vol"] * 0.7:
                    wrong = True  # 冲高无量
                pos["wrong"] = wrong
                continue
            if pos.get("wrong"):
                if nxt < len(df):  # 次日开盘走
                    ep = float(df["open"].iloc[nxt]) * (1 - cost)
                    trades.append({**pos, "exit_date": str(df["trade_date"].iloc[nxt])[:10],
                                   "exit_price": ep, "exit_reason": "次日开盘走: 买错已证明",
                                   "ret": ep / pos["entry_price"] - 1})
                    pos = None
                continue
            strong_seal = (
                float(r["close"]) >= float(r["prev_close"]) * (1 + limit - 0.002)
                and float(r["volume"]) > float(r["vma5"]) * LIMIT_HOLD_VOL
            )
            if strong_seal:
                continue  # 快速大单封死涨停代理: 暂留
            ep = float(r["close"]) * (1 - cost)
            reason = "次日了结: 预期兑现或预期未现" if d == 2 else "封板失败离场"
            trades.append({**pos, "exit_date": td, "exit_price": ep,
                           "exit_reason": reason, "ret": ep / pos["entry_price"] - 1})
            pos = None
            continue

        # ---- 入场: T 收盘确认, T+1 开盘成交 ----
        if nxt >= len(df) or bool(env_crash.iloc[i]):
            continue
        sig = ""
        if (
            not bool(env_bear.iloc[i])
            and float(r["close"]) > float(r["plat_high"])
            and float(r["volume"]) > float(r["vma20"]) * GERI_VOL_MULT
            and float(r["close"]) >= float(r["open"])
        ):
            sig = "breakout"
        elif (
            float(r["max_close_20"]) >= float(r["close_20ago"]) * (1 + GERI_PULLBACK_STRENGTH)
            and float(r["low"]) <= max(float(r["ma5"]), float(r["ma10"]))
            and float(r["close"]) >= min(float(r["ma5"]), float(r["ma10"]))
            and float(r["volume"]) < float(r["vma5"]) * GERI_PULLBACK_VOL
            and float(r["close"]) >= float(r["low"]) + 0.5 * (float(r["high"]) - float(r["low"]))
        ):
            sig = "pullback"
        if not sig:
            continue
        open_nxt = float(df["open"].iloc[nxt])
        if open_nxt > float(r["close"]) * GERI_NO_CHASE:
            continue  # 高位不追
        pos = {
            "buy_type": sig,
            "signal_date": td,
            "entry_date": str(df["trade_date"].iloc[nxt])[:10],
            "entry_i": nxt,
            "entry_price": open_nxt * (1 + cost),
            "plat_high": float(r["plat_high"]),
            "sig_high": float(r["high"]),
            "sig_vol": float(r["volume"]),
        }

    if pos is not None:
        ep = float(df["close"].iloc[-1]) * (1 - cost)
        trades.append({**pos, "exit_date": str(df["trade_date"].iloc[-1])[:10],
                       "exit_price": ep, "exit_reason": "期末强平 (研究口径)",
                       "ret": ep / pos["entry_price"] - 1})
    for t in trades:
        t.pop("entry_i", None)
        t.pop("wrong", None)
    return trades


# --------------------------------------------------------------------------- #
# 合成数据 (测试用)
# --------------------------------------------------------------------------- #
def _path_to_df(points: list[float], per_leg: int = 6, start: str = "2026-01-05") -> pd.DataFrame:
    rows: list[dict] = []
    dates = pd.bdate_range(start, periods=(len(points) - 1) * per_leg + 1)
    price = points[0]
    for a, b in pairwise(points):
        for j in range(per_leg):
            nxt = a + (b - a) * (j + 1) / per_leg
            rows.append({
                "trade_date": dates[len(rows)],
                "open": price,
                "high": max(price, nxt) + 0.05,
                "low": min(price, nxt) - 0.05,
                "close": nxt,
                "volume": 1000.0,
            })
            price = nxt
    # 转折点严格极值化 (避免相邻 K 极值相等破坏分型)
    for k in range(per_leg - 1, len(rows) - 1, per_leg):
        prev_c, cur_c, next_c = rows[k - 1]["close"], rows[k]["close"], rows[k + 1]["close"]
        if cur_c < prev_c and next_c > cur_c:  # 底
            rows[k]["low"] = min(rows[k - 1]["low"], rows[k + 1]["low"]) - 0.1
        elif cur_c > prev_c and next_c < cur_c:  # 顶
            rows[k]["high"] = max(rows[k - 1]["high"], rows[k + 1]["high"]) + 0.1
    return pd.DataFrame(rows)


def make_synthetic_third_buy_series() -> pd.DataFrame:
    # 下跌 → 三笔重叠中枢 → 离开至 11.5 → 回抽 10.4 不破 ZG → 新高 12.5 → 不创新高 12.3
    pts = [11.0, 8.0, 10.0, 9.0, 10.2, 9.2, 11.5, 10.4, 12.5, 11.8, 12.3, 11.5]
    return _path_to_df(pts)


def make_synthetic_breakout_series(gap_open: float = 1.0, bear_env: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=30)
    rows = [
        {"trade_date": d, "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.0,
         "volume": 1000.0}
        for d in dates[:25]
    ]
    rows.append({"trade_date": dates[25], "open": 10.1, "high": 10.8,
                 "low": 10.05, "close": 10.7, "volume": 2500.0})  # T: 放量突破
    o1 = 10.7 * gap_open
    rows.append({"trade_date": dates[26], "open": o1, "high": max(o1, 10.9),
                 "low": min(o1, 10.5), "close": 10.65, "volume": 1800.0})  # T+1
    rows.append({"trade_date": dates[27], "open": 10.6, "high": 10.7,
                 "low": 10.3, "close": 10.4, "volume": 1500.0})  # T+2: 不封板 → 了结
    rows.append({"trade_date": dates[28], "open": 10.4, "high": 10.5,
                 "low": 10.2, "close": 10.3, "volume": 1200.0})
    rows.append({"trade_date": dates[29], "open": 10.3, "high": 10.4,
                 "low": 10.1, "close": 10.2, "volume": 1100.0})
    df = pd.DataFrame(rows)
    if bear_env:
        df.attrs["bear_env"] = True
    return df


def make_synthetic_pullback_series() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=26)
    rows: list[dict] = []
    price = 10.0
    for i in range(21):  # 21 天上涨, 回踩日落在回测循环起点 (idx 21) 处
        nxt = price + 0.13
        rows.append({"trade_date": dates[i], "open": price, "high": nxt + 0.05,
                     "low": price - 0.05, "close": nxt, "volume": 1100.0})
        price = nxt
    # T (idx 21): 缩量回踩均线, 收在当日区间上半
    rows.append({"trade_date": dates[21], "open": price - 0.1, "high": price + 0.02,
                 "low": price - 0.45, "close": price - 0.15, "volume": 500.0})
    rows.append({"trade_date": dates[22], "open": price - 0.18, "high": price + 0.12,
                 "low": price - 0.2, "close": price + 0.02, "volume": 900.0})
    rows.append({"trade_date": dates[23], "open": price + 0.02, "high": price + 0.3,
                 "low": price, "close": price + 0.22, "volume": 1000.0})
    rows.append({"trade_date": dates[24], "open": price + 0.22, "high": price + 0.3,
                 "low": price + 0.05, "close": price + 0.15, "volume": 1000.0})
    rows.append({"trade_date": dates[25], "open": price + 0.15, "high": price + 0.2,
                 "low": price, "close": price + 0.1, "volume": 1000.0})
    return pd.DataFrame(rows)


def make_bear_env_df(n: int = 160) -> pd.DataFrame:
    # 缓慢阴跌: close < MA60 (熊市) 但 5 日跌幅不到 -7% (非暴跌), 覆盖合成序列日期
    dates = pd.bdate_range("2025-07-01", periods=n)
    close = np.linspace(12.0, 10.5, n)
    return pd.DataFrame({
        "trade_date": dates, "open": close, "high": close + 0.05,
        "low": close - 0.05, "close": close, "volume": 1000.0,
    })


# --------------------------------------------------------------------------- #
# 组合层: 单仓账户, 多标的信号先到先得
# --------------------------------------------------------------------------- #
def simulate_account(
    all_trades: dict[str, list[dict]], initial: float = 100_000.0
) -> tuple[list[dict], list[dict]]:
    flat = [{"code": c, **t} for c, ts in all_trades.items() for t in ts]
    flat.sort(key=lambda t: (t["entry_date"], t["signal_date"]))
    cash = initial
    taken: list[dict] = []
    curve: list[dict] = []
    busy_until = ""
    for t in flat:
        if t["entry_date"] <= busy_until:
            continue
        before = cash
        cash *= 1 + t["ret"]
        taken.append({**t, "cash_before": before, "cash_after": cash})
        busy_until = t["exit_date"]
        curve.append({"date": t["exit_date"], "equity": cash})
    return curve, taken


def metrics(curve: list[dict], taken: list[dict], initial: float = 100_000.0) -> dict:
    if not taken:
        return {"trades": 0}
    eq = pd.Series([initial, *[c["equity"] for c in curve]])
    rets = eq.pct_change().dropna()
    mdd = float((eq / eq.cummax() - 1).min())
    wins = [t for t in taken if t["ret"] > 0]
    gp = sum(t["ret"] for t in taken if t["ret"] > 0)
    gl = -sum(t["ret"] for t in taken if t["ret"] < 0)
    years = max(
        (pd.Timestamp(taken[-1]["exit_date"]) - pd.Timestamp(taken[0]["entry_date"])).days / 365.25,
        1e-9,
    )
    total = float(eq.iloc[-1] / initial - 1)
    return {
        "trades": len(taken),
        "win_rate": round(len(wins) / len(taken), 4),
        "avg_ret": round(float(np.mean([t["ret"] for t in taken])), 4),
        "profit_factor": round(gp / gl, 3) if gl > 0 else float("inf"),
        "total_return": round(total, 4),
        "cagr": round(float((1 + total) ** (1 / years) - 1), 4),
        "mdd": round(mdd, 4),
        "sharpe": round(float(rets.mean() / rets.std() * np.sqrt(252)), 3)
        if len(rets) > 1 and rets.std() > 0
        else 0.0,
        "avg_hold_days": round(
            float(np.mean([
                (pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days for t in taken
            ])),
            1,
        ),
    }


def by_key(taken: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for t in taken:
        out.setdefault(t[key], []).append(t)
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_all(cost_mult: float = 1.0) -> dict:
    data = rq.load_data()
    env = data.get("159915")
    cost = COST_SIDE * cost_mult
    chan_trades: dict[str, list[dict]] = {}
    geri_trades: dict[str, list[dict]] = {}
    for code, df in data.items():
        if code == rq.DEFENSE:
            continue
        chan_trades[code] = run_chan_on_series(df, cost=cost)
        geri_trades[code] = run_geri_on_series(df, cost=cost, env_df=env, code=code)
    result: dict = {"cost_mult": cost_mult, "modes": {}}
    for mode, trades_map in (("chan", chan_trades), ("geri", geri_trades)):
        curve, taken = simulate_account(trades_map)
        m = metrics(curve, taken)
        m["by_buy_type"] = {
            k: {
                "trades": len(v),
                "win_rate": round(len([t for t in v if t["ret"] > 0]) / len(v), 4),
                "avg_ret": round(float(np.mean([t["ret"] for t in v])), 4),
                "sum_ret": round(float(np.sum([t["ret"] for t in v])), 4),
            }
            for k, v in by_key(taken, "buy_type").items()
        }
        m["by_code"] = {
            k: {"trades": len(v), "sum_ret": round(float(np.sum([t["ret"] for t in v])), 4)}
            for k, v in by_key(taken, "code").items()
        }
        result["modes"][mode] = m
        result[f"{mode}_trades"] = taken
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-mult", type=float, default=0.0, help="只跑指定成本倍数; 默认全跑 1/2/3x")
    args = ap.parse_args()
    mults = [args.cost_mult] if args.cost_mult else [1.0, 2.0, 3.0]
    out = {}
    for m in mults:
        print(f"=== 成本 {m:.0f}x ===")
        res = run_all(m)
        out[f"{m}x"] = res
        for mode, mm in res["modes"].items():
            head = {k: v for k, v in mm.items() if not k.startswith("by_")}
            print(f"  [{mode}] {json.dumps(head, ensure_ascii=False)}")
            print(f"    by_type: {json.dumps(mm['by_buy_type'], ensure_ascii=False)}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fp = RESULT_DIR / "structure_trade_20261002.json"
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n结果已写入 {fp}")


if __name__ == "__main__":
    main()
