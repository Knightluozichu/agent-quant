"""Three-strategy comparison payload helpers."""

from __future__ import annotations

from scripts.trade_server import _series_payload


def test_series_payload_metrics_and_drawdown() -> None:
    dates = ["2020-06-19", "2020-06-22", "2020-06-23", "2021-01-04"]
    equity = [100000.0, 101000.0, 99000.0, 110000.0]
    payload = _series_payload(
        series_id="V5",
        name="V5",
        color="#1f7a4d",
        dates=dates,
        equity=equity,
        holdings=[None, "501018", "501018", None],
        n_trades=2,
    )
    assert payload["id"] == "V5"
    assert payload["metrics"]["total_return"] == 10.0
    assert payload["metrics"]["max_drawdown"] < 0
    assert payload["metrics"]["n_trades"] == 2
    assert payload["metrics"]["years"] > 0
    assert set(payload["metrics"]) >= {
        "total_return",
        "ann_return",
        "sharpe",
        "max_drawdown",
        "n_trades",
        "years",
        "start",
        "end",
    }
    assert payload["yearly"][0]["year"] == 2020
