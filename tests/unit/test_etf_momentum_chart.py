"""Read-only charts: exact production momentum, date alignment and causality."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import archive_etf_momentum_chart as chart
import run_qixing_v3 as rq


def bars() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=65)
    return pd.concat(
        [
            pd.DataFrame({"code": code, "trade_date": dates, "close": 100 * rate ** np.arange(65)})
            for code, rate in [("518880", 1.01), ("501018", 0.995)]
        ],
        ignore_index=True,
    )


def test_exact_production_score_and_daily_return() -> None:
    raw = bars()
    view = chart.build_view(raw, cutoff=pd.Timestamp("2026-03-31"), days=30)
    assert len(view) == 60
    for code, group in view.groupby("code"):
        assert len(group) == 30
        history = raw[raw.code == code]
        for row in group.itertuples():
            closes = history[history.trade_date <= row.trade_date].close.to_numpy()
            assert row.momentum == pytest.approx(rq.calc_momentum_score(closes))
            assert row.daily_return == pytest.approx(closes[-1] / closes[-2] - 1)


def test_future_prices_cannot_change_past_chart() -> None:
    raw = bars()
    cutoff = raw.trade_date.unique()[54]
    before = chart.build_view(raw, cutoff=pd.Timestamp(cutoff), days=30)
    raw.loc[raw.trade_date > cutoff, "close"] = np.nan
    pd.testing.assert_frame_equal(
        before, chart.build_view(raw, cutoff=pd.Timestamp(cutoff), days=30)
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_prices_rejected(bad: float) -> None:
    raw = bars()
    raw.loc[20, "close"] = bad
    with pytest.raises(ValueError, match="positive finite"):
        chart.build_view(raw, cutoff=raw.trade_date.max(), days=30)


def test_duplicate_dates_rejected() -> None:
    raw = bars()
    with pytest.raises(ValueError, match="duplicate"):
        chart.build_view(pd.concat([raw, raw.iloc[[0]]]), cutoff=raw.trade_date.max(), days=30)


def test_missing_asset_date_rejected() -> None:
    raw = bars().drop(index=40)
    with pytest.raises(ValueError, match="missing"):
        chart.build_view(raw, cutoff=raw.trade_date.max(), days=30)


def test_insufficient_warmup_rejected() -> None:
    raw = bars().groupby("code").tail(49)
    with pytest.raises(ValueError, match="history"):
        chart.build_view(raw, cutoff=raw.trade_date.max(), days=30)


def test_no_silent_stale_cutoff() -> None:
    raw = bars()
    with pytest.raises(ValueError, match="cutoff"):
        chart.build_view(raw, cutoff=raw.trade_date.max() + pd.Timedelta(days=1), days=30)
