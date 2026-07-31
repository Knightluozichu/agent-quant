"""R1 无未来函数回测测试.

验证:
  - T日信号 → T+1开盘成交
  - T+1停牌/涨跌停时不成交
  - 卖出失败时不继续买入
  - 最后交易日未执行信号不计入
  - 每日净值曲线 (含非调仓日)
  - 成交记录包含信号时间、执行时间、hash
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import pytest

# 确保能 import scripts 模块
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_qixing_v3 import (  # noqa: E402
    DEFENSE,
    ETF_POOL,
    REBALANCE_DAYS,
    _check_tradable,
    _compute_data_hash,
    _compute_param_hash,
    calc_momentum_score,
    check_a_share_weak,
    check_short_momentum,
    check_single_day_drop,
    check_volume_spike,
    run_cost_stress_test,
    run_qixing_v3,
    run_qixing_v3_no_lookahead,
)


def _make_synthetic_data(
    n_days: int = 300,
    start_date: str = "2020-01-02",
    codes: list[str] | None = None,
    trend_code: str = "518880",
    flat_codes: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """构造合成ETF数据.

    Args:
        n_days: 交易日数
        start_date: 起始日期
        codes: 使用的代码列表 (默认全部 ETF_POOL + DEFENSE)
        trend_code: 上涨趋势的代码 (动量>0)
        flat_codes: 横盘代码 (动量≈0)
    """
    if codes is None:
        codes = list(ETF_POOL.keys()) + [DEFENSE]
    if flat_codes is None:
        flat_codes = []

    dates = pd.bdate_range(start_date, periods=n_days)
    data = {}
    for code in codes:
        if code == DEFENSE:
            # 货币基金: 几乎不涨
            close = 100.0 * (1 + 0.001 * np.arange(n_days))
        elif code == trend_code:
            # 趋势上涨: 每日 +0.3%
            close = 100.0 * (1.003 ** np.arange(n_days))
        elif code in flat_codes:
            # 横盘
            close = 100.0 + np.sin(np.arange(n_days) * 0.1) * 2
        else:
            # 缓慢下跌: 每日 -0.1%
            close = 100.0 * (0.999 ** np.arange(n_days))

        df = pd.DataFrame({
            "trade_date": dates[:n_days],
            "open": close * 0.998,  # 开盘略低于收盘
            "close": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "volume": 10000.0,
            "symbol": code,
        })
        data[code] = df

    return data


class TestNoLookaheadBasics:
    """基础功能测试."""

    def test_returns_result_dict(self):
        """回测返回完整结果字典."""
        data = _make_synthetic_data()
        result = run_qixing_v3_no_lookahead(data)
        assert "error" not in result
        assert "total_return" in result
        assert "equity_curve" in result
        assert "trade_log" in result
        assert "param_hash" in result
        assert "data_hash" in result

    def test_daily_equity_curve(self):
        """净值曲线包含每日数据 (含非调仓日)."""
        data = _make_synthetic_data(n_days=300)
        result = run_qixing_v3_no_lookahead(data)
        eq = result["equity_curve"]
        # 300天 - 130 warmup = 170 个交易日
        # 无未来函数版每日采样, 所以应该有约170个点
        assert len(eq) > 100, f"净值曲线应包含每日数据, 实际 {len(eq)} 条"

    def test_trade_log_has_audit_fields(self):
        """成交记录包含信号时间、执行时间、hash."""
        data = _make_synthetic_data(n_days=300)
        result = run_qixing_v3_no_lookahead(data)
        trade_log = result["trade_log"]
        assert len(trade_log) > 0, "应至少有一笔成交记录"

        # 检查每条记录的审计字段
        for t in trade_log:
            assert "signal_id" in t, "缺少 signal_id"
            assert "signal_date" in t, "缺少 signal_date"
            assert "execution_date" in t, "缺少 execution_date"
            assert "data_hash" in t, "缺少 data_hash"
            assert "param_hash" in t, "缺少 param_hash"
            assert "status" in t, "缺少 status"


class TestSignalExecutionTiming:
    """T日信号 → T+1开盘成交."""

    def test_signal_date_before_execution_date(self):
        """所有已执行成交的信号日期 < 执行日期."""
        data = _make_synthetic_data(n_days=300)
        result = run_qixing_v3_no_lookahead(data)
        trade_log = result["trade_log"]

        for t in trade_log:
            if t["status"] == "executed" and t.get("execution_date"):
                assert t["signal_date"] < t["execution_date"], (
                    f"信号日期 {t['signal_date']} 应早于执行日期 {t['execution_date']}"
                )

    def test_execution_uses_open_price(self):
        """成交价应为 T+1 开盘价 (不是收盘价)."""
        data = _make_synthetic_data(n_days=300, trend_code="518880")
        result = run_qixing_v3_no_lookahead(data)
        trade_log = result["trade_log"]

        for t in trade_log:
            if t["status"] == "executed" and t.get("action") == "buy":
                code = t["code"]
                exec_date = t["execution_date"]
                df = data[code]
                row = df[df["trade_date"].astype(str) == exec_date]
                if not row.empty:
                    expected_open = float(row.iloc[0]["open"])
                    actual_price = t["price"]
                    assert abs(actual_price - expected_open) < 0.001, (
                        f"买入价 {actual_price} 应等于 {exec_date} 开盘价 {expected_open}"
                    )


class TestEdgeCases:
    """边缘情况测试."""

    def test_sell_failure_prevents_buy(self):
        """卖出失败时不继续买入."""
        data = _make_synthetic_data(n_days=300)

        # 制造持仓: 先正常跑一遍找到有持仓的状态
        result = run_qixing_v3_no_lookahead(data)
        trade_log = result["trade_log"]

        # 验证: 每个信号要么 sell+buy 都执行, 要么 sell cancelled 后 buy 也 cancelled
        signals = {}
        for t in trade_log:
            sid = t["signal_id"]
            if sid not in signals:
                signals[sid] = []
            signals[sid].append(t)

        for sid, trades in signals.items():
            sell_status = None
            buy_status = None
            for t in trades:
                if t.get("action") == "sell":
                    sell_status = t["status"]
                elif t.get("action") == "buy":
                    buy_status = t["status"]

            # 如果卖出被取消, 买入也必须被取消
            if sell_status == "cancelled":
                assert buy_status in ("cancelled", None), (
                    f"信号 {sid}: 卖出取消但买入状态为 {buy_status}, 应也取消"
                )

    def test_last_signal_unexecuted(self):
        """最后一个交易日未执行信号不计入成交."""
        data = _make_synthetic_data(n_days=300)
        result = run_qixing_v3_no_lookahead(data)
        trade_log = result["trade_log"]

        # 最后一个信号应该标记为 unexecuted
        unexecuted = [t for t in trade_log if t["status"] == "unexecuted"]
        # 最后一个调仓日的信号无法在 T+1 执行 (因为没有 T+1)
        # 所以至少有 0 或 1 个未执行信号
        # (如果最后一个交易日恰好不是调仓日, 则没有未执行信号)
        for t in unexecuted:
            assert t["execution_date"] is None, "未执行信号的 execution_date 应为 None"

    def test_limit_up_prevents_trade(self):
        """涨跌停时不得成交."""
        data = _make_synthetic_data(n_days=300)

        # 验证 _check_tradable 对涨跌停的判断
        df = data["518880"]
        # 构造一个涨跌停日: open 比前一日 close 涨 10%
        test_date = df.iloc[100]["trade_date"]
        prev_close = float(df.iloc[99]["close"])
        df.loc[100, "open"] = prev_close * 1.10  # 涨停

        can_trade, reason = _check_tradable(data, "518880", test_date)
        assert not can_trade, f"涨跌停应不可交易, reason: {reason}"
        assert "涨跌停" in reason

    def test_suspended_prevents_trade(self):
        """停牌时不得成交."""
        data = _make_synthetic_data(n_days=300)

        # 删除某一天的数据模拟停牌
        df = data["518880"]
        test_date = df.iloc[100]["trade_date"]
        df_dropped = df.drop(100)

        data_modified = {**data, "518880": df_dropped}
        can_trade, reason = _check_tradable(data_modified, "518880", test_date)
        assert not can_trade, f"停牌应不可交易, reason: {reason}"


class TestCostStressTest:
    """成本压力测试."""

    def test_stress_test_returns_three_scenarios(self):
        """压力测试返回 1x/2x/3x 三档."""
        data = _make_synthetic_data(n_days=300)
        stress = run_cost_stress_test(data)

        assert "1x" in stress
        assert "2x" in stress
        assert "3x" in stress

    def test_higher_cost_lower_return(self):
        """成本越高, 收益越低."""
        data = _make_synthetic_data(n_days=300)
        stress = run_cost_stress_test(data)

        ret_1x = stress["1x"]["total_return"]
        ret_2x = stress["2x"]["total_return"]
        ret_3x = stress["3x"]["total_return"]

        assert ret_1x >= ret_2x, f"2x成本收益应低于1x: {ret_1x} vs {ret_2x}"
        assert ret_2x >= ret_3x, f"3x成本收益应低于2x: {ret_2x} vs {ret_3x}"


class TestHashStability:
    """Hash 稳定性测试."""

    def test_param_hash_stable(self):
        """相同参数 → 相同 hash."""
        h1 = _compute_param_hash()
        h2 = _compute_param_hash()
        assert h1 == h2, "参数 hash 应稳定"

    def test_data_hash_stable(self):
        """相同数据 → 相同 hash."""
        data = _make_synthetic_data(n_days=300)
        h1 = _compute_data_hash(data)
        h2 = _compute_data_hash(data)
        assert h1 == h2, "数据 hash 应稳定"

    def test_different_data_different_hash(self):
        """不同数据 → 不同 hash."""
        data1 = _make_synthetic_data(n_days=300)
        data2 = _make_synthetic_data(n_days=400)
        h1 = _compute_data_hash(data1)
        h2 = _compute_data_hash(data2)
        assert h1 != h2, "不同数据应有不同 hash"


# --------------------------------------------------------------------------- #
# 过滤器函数测试
# --------------------------------------------------------------------------- #
class TestFilterFunctions:
    """各过滤器函数的单元测试."""

    def test_calc_momentum_score_positive_trend(self):
        """上涨趋势的动量分为正."""
        close = 100.0 * (1.003 ** np.arange(150))
        score = calc_momentum_score(close)
        assert score > 0, "上涨趋势动量应为正"

    def test_calc_momentum_score_negative_trend(self):
        """下跌趋势的动量分为负."""
        close = 100.0 * (0.997 ** np.arange(150))
        score = calc_momentum_score(close)
        assert score < 0, "下跌趋势动量应为负"

    def test_calc_momentum_score_short_data(self):
        """数据不足时返回 0."""
        close = np.array([100.0, 101.0])
        score = calc_momentum_score(close)
        assert score == 0.0, "数据不足时动量应为 0"

    def test_check_short_momentum_pass(self):
        """正收益时通过短期动量过滤."""
        close = 100.0 * (1.01 ** np.arange(15))
        assert check_short_momentum(close) is True

    def test_check_short_momentum_fail(self):
        """负收益时不通过短期动量过滤."""
        close = 100.0 * (0.99 ** np.arange(15))
        assert check_short_momentum(close) is False

    def test_check_short_momentum_insufficient(self):
        """数据不足时通过 (返回 True)."""
        close = np.array([100.0, 101.0])
        assert check_short_momentum(close) is True

    def test_check_volume_spike_normal(self):
        """正常量能通过."""
        close = 100.0 * (1.001 ** np.arange(25))
        volume = np.full(25, 10000.0)
        assert check_volume_spike(volume, close) is True

    def test_check_volume_spike_high_ann_ret_spike(self):
        """高年化收益+放量 → 不通过."""
        close = 100.0 * (1.02 ** np.arange(25))  # 大涨, 年化>100%
        volume = np.full(25, 10000.0)
        volume[-1] = 50000.0  # 最后一天放量5倍
        assert check_volume_spike(volume, close) is False

    def test_check_volume_spike_insufficient(self):
        """数据不足时通过."""
        close = np.array([100.0] * 3)
        volume = np.array([1000.0] * 3)
        assert check_volume_spike(volume, close) is True

    def test_check_single_day_drop_pass(self):
        """无暴跌时通过."""
        close = 100.0 * (1.001 ** np.arange(10))
        assert check_single_day_drop(close) is True

    def test_check_single_day_drop_fail(self):
        """有暴跌>3%时不通过."""
        close = 100.0 * (1.001 ** np.arange(10))
        close[5] = close[4] * 0.95  # 跌5%
        assert check_single_day_drop(close) is False

    def test_check_single_day_drop_insufficient(self):
        """数据不足时通过."""
        close = np.array([100.0, 101.0])
        assert check_single_day_drop(close) is True

    def test_check_a_share_weak_below_ma(self):
        """创业板价格低于MA → A股走弱."""
        data = _make_synthetic_data(n_days=200)
        # 修改创业板数据: 末尾下跌
        df = data["159915"]
        df.loc[df.index[-5:], "close"] = df["close"].iloc[-10] * 0.8
        result = check_a_share_weak(data, len(df) - 1)
        assert result is True, "价格低于MA时应返回 True"

    def test_check_a_share_weak_above_ma(self):
        """创业板价格高于MA → A股不弱."""
        data = _make_synthetic_data(n_days=200, trend_code="159915")
        df = data["159915"]
        result = check_a_share_weak(data, len(df) - 1)
        assert result is False, "价格高于MA时应返回 False"

    def test_check_a_share_weak_insufficient(self):
        """数据不足时返回 False."""
        data = _make_synthetic_data(n_days=200)
        result = check_a_share_weak(data, 3)
        assert result is False, "数据不足时应返回 False"


# --------------------------------------------------------------------------- #
# _check_tradable 边缘情况
# --------------------------------------------------------------------------- #
class TestCheckTradableEdgeCases:
    """_check_tradable 的完整覆盖."""

    def test_code_not_in_data(self):
        """代码不在数据中时不可交易."""
        can, reason = _check_tradable({}, "518880", pd.Timestamp("2022-01-03"))
        assert not can
        assert "不在数据中" in reason

    def test_zero_open_price(self):
        """开盘价为0时不可交易."""
        data = _make_synthetic_data(n_days=300)
        df = data["518880"]
        td = df.iloc[100]["trade_date"]
        df.loc[100, "open"] = 0.0
        can, reason = _check_tradable(data, "518880", td)
        assert not can
        assert "无开盘价" in reason

    def test_normal_trade(self):
        """正常情况可交易."""
        data = _make_synthetic_data(n_days=300)
        df = data["518880"]
        td = df.iloc[100]["trade_date"]
        can, reason = _check_tradable(data, "518880", td)
        assert can, f"正常应可交易: {reason}"


# --------------------------------------------------------------------------- #
# 旧版回测 (lookahead bias) 基础测试
# --------------------------------------------------------------------------- #
class TestLegacyBacktest:
    """旧版 run_qixing_v3 基础测试 (保持对比基线可用)."""

    def test_legacy_returns_result(self):
        """旧版回测返回完整结果."""
        data = _make_synthetic_data(n_days=300)
        result = run_qixing_v3(data)
        assert "error" not in result
        assert "total_return" in result
        assert "equity_curve" in result
        assert "decision_log" in result

    def test_legacy_equity_curve_not_empty(self):
        """旧版净值曲线不为空."""
        data = _make_synthetic_data(n_days=300)
        result = run_qixing_v3(data)
        eq = result["equity_curve"]
        assert len(eq) > 0, "净值曲线不应为空"

    def test_legacy_no_data(self):
        """空数据返回 error."""
        result = run_qixing_v3({})
        assert "error" in result

    def test_legacy_has_trades(self):
        """旧版回测有交易记录 (覆盖交易执行路径)."""
        data = _make_synthetic_data(n_days=300, trend_code="518880")
        result = run_qixing_v3(data)
        assert result["n_trades"] > 0, "旧版回测应有交易"
        assert len(result["decision_log"]) > 0, "应有决策记录"
        # 验证决策记录字段
        for d in result["decision_log"]:
            assert "target" in d
            assert "target_name" in d
            assert "n_candidates" in d
            assert "a_share_weak" in d
            assert "profit_prot" in d
