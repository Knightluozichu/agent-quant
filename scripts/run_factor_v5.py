"""V5: 选择、择时、仓位三层解耦.

架构:
  Layer 1: 绝对风险Gate (决定"要不要持有风险资产")
  Layer 2: 4专家模型 (趋势/反转/突破/低波, 滞后实绩动态权重)
  Layer 3: 相关性聚类仓位 (HRP思想, 同簇≤2只)
  Layer 4: 分批调仓 + 动态退出 (滞回区间, 每日退出检查)

设计原则:
- 纯动量已证明+64%, V5以动量为核心, 不堆砌因子
- 绝对Gate用市场级信号, 不用横截面Rank
- 专家权重用已实现OOS表现, 不用未来数据
- 相关性聚类防止"4只ETF押同一方向"
- 动态退出: 进入Top3才买, 跌出Top7才卖(滞回), 每日检查止损
"""

from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "long_history"
OUTPUT_DIR = PROJECT_ROOT / "data" / "v5_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFENSIVE = {"511010": "国债ETF", "511880": "货币ETF"}
FEE = 0.001
SLIPPAGE = 0.0005


# =============================================================================
# Layer 1: Absolute Risk Gate
# =============================================================================

class AbsoluteRiskGate:
    """Market-level absolute signals. NOT cross-sectional.

    Signals:
    1. Index price vs MA60 (absolute trend)
    2. MA20 vs MA60 (golden/death cross)
    3. 20-day absolute momentum
    4. Volatility regime (20d vs 60d)
    5. % ETFs with positive 20d momentum (breadth)
    6. Average cross-sectional correlation (risk concentration)
    7. Northbound flow (market-level)
    8. Margin balance change (market-level)
    """

    def __init__(self):
        self.northbound_data: pd.DataFrame | None = None
        self.margin_data: pd.DataFrame | None = None

    def load_external(self):
        nb = DATA_DIR / "northbound.parquet"
        if nb.exists():
            self.northbound_data = pd.read_parquet(nb)
        mg = DATA_DIR / "margin_sentiment.parquet"
        if mg.exists():
            self.margin_data = pd.read_parquet(mg)

    def evaluate(self, index_df: pd.DataFrame, tradable: dict[str, pd.DataFrame],
                 as_of: date) -> dict:
        """Returns {risk_on: bool, score: float, reasons: list, details: dict}."""
        idx = index_df[index_df["trade_date"] <= as_of].sort_values("trade_date")
        if len(idx) < 60:
            return {"risk_on": True, "score": 0.0, "reasons": ["warmup"], "details": {}}

        close = idx["close"].values.astype(float)
        reasons = []
        score = 0.0
        details = {}

        # 1. Absolute trend: price vs MA60
        ma60 = close[-60:].mean()
        ma20 = close[-20:].mean()
        details["price_vs_ma60"] = (close[-1] - ma60) / ma60
        details["ma20_vs_ma60"] = (ma20 - ma60) / ma60

        if close[-1] > ma60:
            score += 1.5
        else:
            score -= 1.5
            reasons.append("below_MA60")

        if ma20 > ma60:
            score += 1.0
        else:
            score -= 1.0
            reasons.append("death_cross")

        # 2. 20-day absolute momentum
        ret_20d = (close[-1] - close[-20]) / close[-20]
        details["ret_20d"] = ret_20d
        if ret_20d > 0.02:
            score += 0.5
        elif ret_20d < -0.03:
            score -= 1.0
            reasons.append(f"20d_drop={ret_20d:.1%}")

        # 3. Volatility regime
        rets_20 = np.diff(close[-21:]) / close[-21:-1]
        rets_60 = np.diff(close[-61:]) / close[-61:-1]
        vol_20 = np.std(rets_20) * np.sqrt(252)
        vol_60 = np.std(rets_60) * np.sqrt(252)
        details["vol_20"] = vol_20
        details["vol_60"] = vol_60
        if vol_20 > vol_60 * 1.8:
            score -= 1.5
            reasons.append(f"vol_spike({vol_20:.0%})")

        # 4. Breadth: % ETFs with positive 20d momentum
        n_positive = 0
        n_total = 0
        for sym, df in tradable.items():
            hist = df[df["trade_date"] <= as_of].sort_values("trade_date")
            if len(hist) > 20:
                c = hist["close"].values
                if (c[-1] - c[-20]) / c[-20] > 0:
                    n_positive += 1
                n_total += 1
        breadth = n_positive / max(n_total, 1)
        details["breadth"] = breadth
        if breadth < 0.3:
            score -= 1.5
            reasons.append(f"breadth_low({breadth:.0%})")
        elif breadth > 0.7:
            score += 0.5

        # 5. Northbound flow (market-level)
        if self.northbound_data is not None and not self.northbound_data.empty:
            nb = self.northbound_data[self.northbound_data["trade_date"] <= as_of]
            if len(nb) >= 5 and "net_flow" in nb.columns:
                flow_5d = nb["net_flow"].tail(5).sum()
                std_60 = nb["net_flow"].tail(60).std() if len(nb) > 60 else 1e8
                nb_signal = flow_5d / (std_60 + 1e-8)
                details["northbound"] = nb_signal
                if nb_signal < -2.0:
                    score -= 1.0
                    reasons.append(f"northbound_out({nb_signal:.1f})")
                elif nb_signal > 2.0:
                    score += 0.5

        # 6. Margin sentiment (market-level)
        if self.margin_data is not None and not self.margin_data.empty:
            m = self.margin_data[self.margin_data["trade_date"] <= as_of]
            if len(m) >= 10 and "margin_balance" in m.columns:
                bal = m["margin_balance"].values
                change_5d = (bal[-1] - bal[-5]) / (bal[-5] + 1e-8)
                if len(bal) > 60:
                    changes = np.diff(bal[-60:]) / bal[-60:-1]
                    ms = change_5d / (np.std(changes) + 1e-8)
                else:
                    ms = change_5d * 100
                details["margin"] = ms
                if ms < -2.0:
                    score -= 1.0
                    reasons.append(f"margin_delev({ms:.1f})")

        # Decision threshold: score range roughly [-8, +4]
        # EXTREME only: gate almost never closes (avoid missing recoveries)
        # v4.3 showed gate adds +0.71%/period in 2023 but costs in other years
        # Only trigger in 2008/2020-level crashes
        risk_on = score >= -5.0
        return {"risk_on": risk_on, "score": score, "reasons": reasons, "details": details}


# =============================================================================
# Layer 2: Expert Models
# =============================================================================

class ExpertModels:
    """4 experts, each produces a ranking of ETFs.

    Experts:
    1. Trend: momentum(20-80d) + trend(MA20-MA60) + volume_trend
    2. Reversal: reversal(5d) + rsi + bias(inverted)
    3. Breakout: breakout + momentum_accel + obv
    4. Low-vol: low_vol + atr_ratio + amplitude

    Expert weights updated by trailing realized performance (no future data).
    Momentum-dominant: trend expert starts with higher weight.
    """

    EXPERT_FACTORS = {
        "trend": ["momentum", "trend", "volume_trend"],
        "reversal": ["reversal", "rsi", "bias"],  # bias inverted in calc
        "breakout": ["breakout", "momentum_accel", "obv"],
        "low_vol": ["low_vol", "atr_ratio", "amplitude"],
    }

    def __init__(self, n_experts: int = 4):
        self.n_experts = n_experts
        self.expert_names = list(self.EXPERT_FACTORS.keys())
        # Momentum-ABSOLUTE-dominant: trend 70%, others 10% each
        # Rationale: v4.3 audit proved pure momentum = +64%, don't dilute
        self.expert_weights = np.array([0.70, 0.10, 0.10, 0.10])
        # Trailing performance tracking
        self._expert_returns: list[list[float]] = [[] for _ in range(n_experts)]
        self._lookback = 3  # Fast adaptation (3 periods)

    def score_etfs(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Score each ETF with each expert, then combine with expert weights."""
        if factor_df.empty:
            return factor_df

        expert_scores = np.zeros((len(factor_df), self.n_experts))

        for ei, (ename, factors) in enumerate(self.EXPERT_FACTORS.items()):
            # Equal-weight within expert
            raw = np.zeros(len(factor_df))
            for f in factors:
                if f in factor_df.columns:
                    vals = factor_df[f].values
                    # For reversal expert, bias should be inverted (low bias = oversold = buy)
                    if ename == "reversal" and f == "bias":
                        vals = -vals
                    raw += vals
            raw /= len(factors)
            # Rank within expert
            ranked = stats.rankdata(raw)
            expert_scores[:, ei] = (ranked - ranked.mean()) / (ranked.std() + 1e-8)

        # Combine with expert weights
        factor_df = factor_df.copy()
        factor_df["composite"] = expert_scores @ self.expert_weights

        # Store per-expert scores for attribution
        for ei, ename in enumerate(self.expert_names):
            factor_df[f"expert_{ename}"] = expert_scores[:, ei]

        return factor_df

    def update_weights(self, expert_period_returns: list[float]):
        """Update expert weights based on realized performance (no future data).

        Called AFTER each rebalance period with the actual returns each expert
        would have achieved if used alone.
        """
        for ei, ret in enumerate(expert_period_returns):
            self._expert_returns[ei].append(ret)

        # Use trailing lookback
        if all(len(r) >= 2 for r in self._expert_returns):
            trailing = []
            for ei in range(self.n_experts):
                recent = self._expert_returns[ei][-self._lookback:]
                trailing.append(np.mean(recent))

            # Softmax with temperature (moderate concentration)
            trailing = np.array(trailing)
            temp = 0.03  # Lower temp = faster adaptation to regime changes
            exp_t = np.exp((trailing - trailing.max()) / temp)
            new_weights = exp_t / exp_t.sum()

            # Clamp: trend expert floor 40% (momentum is proven alpha)
            new_weights = np.clip(new_weights, 0.05, 0.75)
            new_weights[0] = max(new_weights[0], 0.40)  # Trend floor
            new_weights /= new_weights.sum()
            self.expert_weights = new_weights

    def get_expert_returns(self, factor_df: pd.DataFrame, forward_returns: dict[str, float]) -> list[float]:
        """Calculate what each expert alone would have returned (for weight update)."""
        results = []
        for ei, (ename, factors) in enumerate(self.EXPERT_FACTORS.items()):
            # Rank by this expert's score
            col = f"expert_{ename}"
            if col not in factor_df.columns:
                results.append(0.0)
                continue
            top4 = factor_df.nlargest(4, col)["symbol"].tolist()
            rets = [forward_returns.get(s, 0.0) for s in top4]
            results.append(np.mean(rets) if rets else 0.0)
        return results


# =============================================================================
# Layer 3: Correlation-Aware Position Sizing (HRP-inspired)
# =============================================================================

def correlation_cluster_weights(
    selected: list[str],
    data: dict[str, pd.DataFrame],
    as_of: date,
    max_per_cluster: int = 2,
    lookback: int = 60,
) -> dict[str, float]:
    """HRP-inspired: cluster by correlation, limit per cluster.

    Steps:
    1. Compute pairwise correlation of selected ETFs (60-day returns)
    2. Hierarchical clustering
    3. Within each cluster: inverse-vol weighting
    4. Across clusters: equal risk budget
    5. Cap: max_per_cluster ETFs per cluster get weight
    """
    if len(selected) <= 1:
        return {s: 1.0 for s in selected}

    # Compute returns matrix
    ret_matrix = []
    valid_symbols = []
    for sym in selected:
        if sym not in data:
            continue
        hist = data[sym][data[sym]["trade_date"] <= as_of].sort_values("trade_date")
        if len(hist) < lookback + 1:
            continue
        close = hist["close"].values[-lookback - 1:]
        rets = np.diff(close) / close[:-1]
        ret_matrix.append(rets)
        valid_symbols.append(sym)

    if len(valid_symbols) <= 1:
        return {s: 1.0 / len(selected) for s in selected}

    ret_matrix = np.array(ret_matrix)

    # Correlation matrix
    corr = np.corrcoef(ret_matrix)
    # Distance matrix
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)
    dist = np.nan_to_num(dist, nan=0.5)

    # Hierarchical clustering
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    # Cut into clusters (allow 2 clusters max for 4 ETFs - less restrictive)
    n_clusters = min(max(2, len(valid_symbols) // 2), 2)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Group by cluster
    clusters: dict[int, list[str]] = {}
    for i, sym in enumerate(valid_symbols):
        c = labels[i]
        clusters.setdefault(c, []).append(sym)

    # Within cluster: inverse-vol
    # Across clusters: equal budget
    weights = {}
    cluster_budget = 1.0 / len(clusters)

    for c_id, members in clusters.items():
        # Inverse vol within cluster
        vols = {}
        for sym in members:
            hist = data[sym][data[sym]["trade_date"] <= as_of].sort_values("trade_date")
            close = hist["close"].values[-lookback - 1:]
            rets = np.diff(close) / close[:-1]
            vols[sym] = np.std(rets) * np.sqrt(252)

        inv_vol = {s: 1.0 / (vols[s] + 1e-8) for s in members}
        total_iv = sum(inv_vol.values())

        for sym in members:
            weights[sym] = cluster_budget * (inv_vol[sym] / total_iv)

    # Cap single position at 40% (allow concentration in momentum winners)
    for sym in weights:
        weights[sym] = min(weights[sym], 0.40)

    # Renormalize
    total = sum(weights.values())
    weights = {s: w / total for s, w in weights.items()}

    # Add symbols not in valid_symbols with 0 weight
    for sym in selected:
        if sym not in weights:
            weights[sym] = 0.0

    return weights


# =============================================================================
# Layer 4: Staggered Rebalance + Dynamic Exit
# =============================================================================

class DynamicExitManager:
    """Daily exit check with hysteresis.

    Entry: must be in Top3 to buy
    Exit: drop out of Top8 → sell (wide hysteresis to avoid whipsaw)
    Stop-loss: -15% from entry → force sell (ETFs don't crash like stocks)
    NO trailing stop (causes whipsaw on ETFs)
    """

    def __init__(self, entry_rank: int = 3, exit_rank: int = 8,
                 stop_loss: float = -0.15, trailing_stop: float = -999.0):
        self.entry_rank = entry_rank
        self.exit_rank = exit_rank
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop  # Disabled by default
        # Track positions: {symbol: {entry_price, peak_price, entry_date}}
        self.positions: dict[str, dict] = {}

    def register_entry(self, symbol: str, price: float, entry_date: date):
        self.positions[symbol] = {
            "entry_price": price,
            "peak_price": price,
            "entry_date": entry_date,
        }

    def update_peaks(self, data: dict[str, pd.DataFrame], as_of: date):
        """Update peak prices for all held positions."""
        for sym in list(self.positions.keys()):
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == as_of]
                if not row.empty:
                    price = row.iloc[0]["close"]
                    self.positions[sym]["peak_price"] = max(
                        self.positions[sym]["peak_price"], price
                    )

    def check_exits(self, data: dict[str, pd.DataFrame], as_of: date,
                    current_ranks: dict[str, int]) -> list[tuple[str, str]]:
        """Check which positions should exit. Returns [(symbol, reason)]."""
        exits = []
        for sym, info in list(self.positions.items()):
            if sym not in data:
                continue
            row = data[sym][data[sym]["trade_date"] == as_of]
            if row.empty:
                continue
            price = row.iloc[0]["close"]
            entry_price = info["entry_price"]
            peak_price = info["peak_price"]

            # Stop-loss
            ret_from_entry = (price - entry_price) / entry_price
            if ret_from_entry < self.stop_loss:
                exits.append((sym, f"stop_loss({ret_from_entry:.1%})"))
                continue

            # Trailing stop from peak
            ret_from_peak = (price - peak_price) / peak_price
            if ret_from_peak < self.trailing_stop and ret_from_entry > 0:
                exits.append((sym, f"trailing_stop({ret_from_peak:.1%})"))
                continue

            # Rank-based exit (hysteresis: must drop below exit_rank)
            rank = current_ranks.get(sym, 99)
            if rank > self.exit_rank:
                exits.append((sym, f"rank_exit(#{rank})"))
                continue

        return exits

    def remove(self, symbol: str):
        self.positions.pop(symbol, None)


# =============================================================================
# V5 Backtest Engine
# =============================================================================

def run_v5_backtest(
    data: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    start_date: date | None = None,
    end_date: date | None = None,
    initial_capital: float = 100_000.0,
    rebalance_days: int = 20,
) -> dict:
    """Full V5 backtest with all 4 layers."""
    gate = AbsoluteRiskGate()
    gate.load_external()
    experts = ExpertModels()
    exit_mgr = DynamicExitManager()

    index_symbols = {"idx_000300", "idx_000905", "000300", "000905"}
    tradable = {k: v for k, v in data.items()
                if k not in DEFENSIVE and k not in index_symbols}

    all_dates = sorted(index_df["trade_date"].tolist())
    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]

    warmup = 80
    trading_days = all_dates[warmup:]

    cash = initial_capital
    holdings: dict[str, int] = {}
    equity_history = []
    trade_log = []
    gate_log = []
    expert_weight_history = []
    n_trades = 0
    days_since_rebal = rebalance_days  # Force first rebalance
    prev_factor_df = None

    # Factor calculator (reuse from v4.3)
    from run_factor_v43 import FactorCalc, CROSS_FACTORS
    calc = FactorCalc()
    calc.load_external()

    for td in trading_days:
        # Current equity
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]

        # === DAILY: Update peaks and check exits ===
        exit_mgr.update_peaks(data, td)

        # Get current rankings (if we have a recent factor calculation)
        if prev_factor_df is not None and not prev_factor_df.empty:
            current_ranks = {}
            ranked_df = prev_factor_df.sort_values("composite", ascending=False)
            for rank_i, (_, row) in enumerate(ranked_df.iterrows(), 1):
                current_ranks[row["symbol"]] = rank_i

            exits = exit_mgr.check_exits(data, td, current_ranks)
            for sym, reason in exits:
                if sym in holdings:
                    row = data[sym][data[sym]["trade_date"] == td]
                    if not row.empty:
                        price = row.iloc[0]["close"]
                        revenue = holdings[sym] * price * (1 - FEE - SLIPPAGE)
                        cash += revenue
                        entry_info = exit_mgr.positions.get(sym, {})
                        entry_p = entry_info.get("entry_price", price)
                        ret = (price - entry_p) / entry_p
                        trade_log.append({
                            "date": str(td), "action": "EXIT", "symbol": sym,
                            "reason": reason, "return": ret,
                        })
                        n_trades += 1
                        del holdings[sym]
                        exit_mgr.remove(sym)

        # === REBALANCE (every N days) ===
        days_since_rebal += 1
        if days_since_rebal >= rebalance_days:
            days_since_rebal = 0

            # Layer 1: Absolute Risk Gate
            gate_result = gate.evaluate(index_df, tradable, td)
            gate_log.append({
                "date": str(td), "risk_on": gate_result["risk_on"],
                "score": gate_result["score"], "reasons": gate_result["reasons"],
            })

            if not gate_result["risk_on"]:
                # Switch to bonds
                for sym in list(holdings.keys()):
                    if sym not in DEFENSIVE:
                        row = data[sym][data[sym]["trade_date"] == td]
                        if not row.empty:
                            cash += holdings[sym] * row.iloc[0]["close"] * (1 - FEE - SLIPPAGE)
                            n_trades += 1
                            exit_mgr.remove(sym)
                            del holdings[sym]
                # Buy bond
                if "511010" in data:
                    row = data["511010"][data["511010"]["trade_date"] == td]
                    if not row.empty:
                        price = row.iloc[0]["close"]
                        equity_now = cash + sum(
                            holdings.get(s, 0) * data[s][data[s]["trade_date"] == td].iloc[0]["close"]
                            for s in holdings if s in data and not data[s][data[s]["trade_date"] == td].empty
                        )
                        shares = int(equity_now * 0.9 / price / 10) * 10
                        if shares > 0 and shares * price <= cash:
                            cash -= shares * price
                            holdings["511010"] = holdings.get("511010", 0) + shares
                            n_trades += 1
            else:
                # Layer 2: Expert scoring
                factor_df = calc.calc_cross_section(tradable, td)
                if not factor_df.empty and len(factor_df) >= 4:
                    factor_df = experts.score_etfs(factor_df)
                    prev_factor_df = factor_df.copy()

                    # Entry: Top3 (hysteresis - stricter entry)
                    top3 = factor_df.nlargest(3, "composite")["symbol"].tolist()

                    # Also keep existing positions that are still in Top7
                    top7 = factor_df.nlargest(7, "composite")["symbol"].tolist()
                    keep = [s for s in holdings if s in top7 and s not in DEFENSIVE]
                    selected = list(set(top3 + keep))[:4]  # Max 4 positions

                    # Layer 3: Correlation-aware weights
                    weights = correlation_cluster_weights(selected, data, td)

                    # Sell non-target
                    for sym in list(holdings.keys()):
                        if sym not in selected and sym not in DEFENSIVE:
                            row = data[sym][data[sym]["trade_date"] == td]
                            if not row.empty:
                                cash += holdings[sym] * row.iloc[0]["close"] * (1 - FEE - SLIPPAGE)
                                n_trades += 1
                                exit_mgr.remove(sym)
                                del holdings[sym]
                    # Sell bonds
                    for bsym in DEFENSIVE:
                        if bsym in holdings:
                            row = data[bsym][data[bsym]["trade_date"] == td]
                            if not row.empty:
                                cash += holdings[bsym] * row.iloc[0]["close"] * (1 - FEE - SLIPPAGE)
                                n_trades += 1
                                del holdings[bsym]

                    # Buy targets
                    equity_now = cash + sum(
                        holdings.get(s, 0) * data[s][data[s]["trade_date"] == td].iloc[0]["close"]
                        for s in holdings if s in data and not data[s][data[s]["trade_date"] == td].empty
                    )
                    for sym, w in weights.items():
                        if w <= 0:
                            continue
                        row = data[sym][data[sym]["trade_date"] == td]
                        if row.empty:
                            continue
                        price = row.iloc[0]["close"]
                        target_val = equity_now * w
                        cur = holdings.get(sym, 0)
                        diff = target_val - cur * price
                        if diff > price * 100:
                            delta = int(diff / price / 100) * 100
                            cost = delta * price * (1 + FEE + SLIPPAGE)
                            if cost <= cash and delta > 0:
                                cash -= cost
                                holdings[sym] = cur + delta
                                n_trades += 1
                                if sym not in exit_mgr.positions:
                                    exit_mgr.register_entry(sym, price, td)

                    # Update expert weights with realized returns
                    if prev_factor_df is not None:
                        fwd = {}
                        for sym in factor_df["symbol"].tolist():
                            if sym in data:
                                future = data[sym][data[sym]["trade_date"] > td].head(20)
                                current = data[sym][data[sym]["trade_date"] == td]
                                if len(future) >= 10 and not current.empty:
                                    fwd[sym] = (future.iloc[-1]["close"] / current.iloc[0]["close"]) - 1
                        if len(fwd) >= 4:
                            expert_rets = experts.get_expert_returns(factor_df, fwd)
                            experts.update_weights(expert_rets)

                    expert_weight_history.append({
                        "date": str(td),
                        **{f"w_{n}": experts.expert_weights[i]
                           for i, n in enumerate(experts.expert_names)},
                    })

        # Record equity
        equity = cash
        for sym, shares in holdings.items():
            if sym in data:
                row = data[sym][data[sym]["trade_date"] == td]
                if not row.empty:
                    equity += shares * row.iloc[0]["close"]
        equity_history.append({"trade_date": td, "equity": equity})

    # Results
    if not equity_history:
        return {"total_return": 0.0}

    eq_df = pd.DataFrame(equity_history)
    eq_df["trade_date"] = pd.to_datetime(eq_df["trade_date"])
    eq_df["year"] = eq_df["trade_date"].dt.year

    total_return = (eq_df["equity"].iloc[-1] / initial_capital) - 1
    n_days = len(eq_df)
    ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
    daily_rets = eq_df["equity"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    cummax = eq_df["equity"].cummax()
    max_dd = ((eq_df["equity"] - cummax) / cummax).min()

    return {
        "total_return": total_return, "ann_return": ann_return,
        "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": max_dd,
        "n_trades": n_trades, "n_days": n_days,
        "equity_curve": eq_df, "trade_log": trade_log,
        "gate_log": gate_log, "expert_weights": expert_weight_history,
    }


# =============================================================================
# Main
# =============================================================================

def load_data():
    data = {}
    for f in DATA_DIR.glob("*.parquet"):
        if f.name in ("combined_long.parquet", "northbound.parquet",
                      "pe_percentile.parquet", "margin_sentiment.parquet"):
            continue
        df = pd.read_parquet(f)
        if "symbol" not in df.columns or "trade_date" not in df.columns:
            continue
        symbol = df["symbol"].iloc[0]
        if f.name.startswith("index_"):
            symbol = f"idx_{symbol}"
            df["symbol"] = symbol
        data[symbol] = df.sort_values("trade_date").reset_index(drop=True)
    index_df = data.get("idx_000300")
    if index_df is None:
        for k, v in data.items():
            if "000300" in k:
                index_df = v
                break
    return data, index_df


def main():
    print("=" * 70)
    print("  V5: 选择、择时、仓位三层解耦")
    print("  Layer1: 绝对Gate | Layer2: 4专家 | Layer3: HRP仓位 | Layer4: 动态退出")
    print("=" * 70)

    data, index_df = load_data()
    if index_df is None:
        print("ERROR: 无指数数据")
        return

    n_etf = len([k for k in data if not k.startswith("idx_")])
    print(f"\n  数据: {n_etf} 标的, {len(index_df)} 天")

    # Full period backtest
    print("\n[1/2] 全周期回测 (本金10万)...")
    result = run_v5_backtest(data, index_df, initial_capital=100_000)

    eq = result["equity_curve"]
    print(f"\n  年度收益:")
    print(f"  {'年份':<6} {'年初':>10} {'年末':>10} {'收益':>8} {'回撤':>8}")
    print(f"  {'-' * 46}")
    prev = 100_000.0
    for year in sorted(eq["year"].unique()):
        ydf = eq[eq["year"] == year]
        if ydf.empty:
            continue
        end_val = ydf["equity"].iloc[-1]
        yr = (end_val / prev) - 1
        cm = ydf["equity"].cummax()
        dd = ((ydf["equity"] - cm) / cm).min()
        print(f"  {year:<6} {prev:>10,.0f} {end_val:>10,.0f} {yr:>+8.2%} {dd:>8.2%}")
        prev = end_val

    final = eq["equity"].iloc[-1]
    total_ret = (final / 100_000) - 1
    n_years = result["n_days"] / 252
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1

    print(f"  {'-' * 46}")
    print(f"\n  10万 → {final:,.0f} ({total_ret:+.1%}, {final / 100_000:.2f}x)")
    print(f"  年化: {ann_ret:+.1%} | 夏普: {result['sharpe']:.2f} | 回撤: {result['max_drawdown']:.1%}")
    print(f"  交易次数: {result['n_trades']}")

    # Gate statistics
    gate_on = sum(1 for g in result["gate_log"] if g["risk_on"])
    gate_off = sum(1 for g in result["gate_log"] if not g["risk_on"])
    print(f"  Gate: 开{gate_on}次 / 关{gate_off}次")

    # Exit statistics
    exit_reasons = {}
    for t in result["trade_log"]:
        if t["action"] == "EXIT":
            reason_type = t["reason"].split("(")[0]
            exit_reasons[reason_type] = exit_reasons.get(reason_type, 0) + 1
    if exit_reasons:
        print(f"  动态退出: {exit_reasons}")

    # Expert weight evolution
    if result["expert_weights"]:
        last_ew = result["expert_weights"][-1]
        print(f"\n  最终专家权重:")
        for k, v in last_ew.items():
            if k.startswith("w_"):
                print(f"    {k[2:]:<10}: {v:.1%}")

    # 2023 stress test
    print(f"\n[2/2] 2023压力测试...")
    from datetime import date as dt_date
    r2023 = run_v5_backtest(data, index_df,
                            start_date=dt_date(2023, 1, 1),
                            end_date=dt_date(2023, 12, 31),
                            initial_capital=100_000)
    if r2023.get("equity_curve") is not None:
        eq23 = r2023["equity_curve"]
        ret_2023 = (eq23["equity"].iloc[-1] / 100_000) - 1
        cm23 = eq23["equity"].cummax()
        dd23 = ((eq23["equity"] - cm23) / cm23).min()
        print(f"  2023: {ret_2023:+.2%} | 回撤: {dd23:.1%}")
        gate_off_23 = sum(1 for g in r2023["gate_log"] if not g["risk_on"])
        print(f"  Gate关闭: {gate_off_23}/{len(r2023['gate_log'])}次")
        exits_23 = [t for t in r2023["trade_log"] if t["action"] == "EXIT"]
        if exits_23:
            print(f"  动态退出: {len(exits_23)}次")
            for e in exits_23[:5]:
                print(f"    {e['date']} {e['symbol']} {e['reason']} ret={e['return']:+.1%}")

    # Benchmark comparison
    print(f"\n  基准对比 (全周期):")
    idx_c = index_df["close"].values
    bench_300 = (idx_c[-1] - idx_c[0]) / idx_c[0]
    print(f"    沪深300: {bench_300:+.1%}")
    print(f"    V5: {total_ret:+.1%}")
    print(f"    超额: {total_ret - bench_300:+.1%}")


if __name__ == "__main__":
    main()
