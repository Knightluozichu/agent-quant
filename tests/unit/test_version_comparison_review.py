"""Historical selector identity and causal, no-account research contracts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts import review_versions_20260916 as review

ROOT = Path(__file__).resolve().parents[2]


def test_historical_loader_hash_and_no_module_side_effects(tmp_path):
    source = tmp_path / "old.py"
    source.write_text(
        "ETF_POOL = {'A': 'A'}\nraise RuntimeError('must not execute')\n"
        "def select_target(*args):\n return ETF_POOL\n"
    )
    selector = review.historical_selector(source, review.sha256(source), {"B": "B"})
    assert selector() == {"B": "B"}
    with pytest.raises(ValueError, match="hash"):
        review.historical_selector(source, "incorrect", {})


def test_archived_v3_is_not_current_gated_selector():
    source = ROOT / "artifacts/version_comparison_20260916/historical_sources/v3_pre_gate.py"
    if not source.exists():
        pytest.skip("frozen historical source bundle required")
    select = review.historical_selector(
        source, review.HISTORICAL["v3"]["sha256"], {"518880": "gold"}
    )
    close = np.full(140, 100.0)
    close[-60:] = 90
    close[-21:] = np.linspace(90, 99, 21)
    close[-3:] = [102, 98, 99]
    assert select.__globals__["check_single_day_drop"](close) is False
    assert review.rq.check_single_day_drop(close)
    frame = pd.DataFrame({"close": close, "volume": 10000})
    assert select({"518880": frame}, {"518880": 139}, None)[0] == review.rq.DEFENSE
    assert (
        review.rq.select_target({"518880": frame}, {"518880": 139}, None, pool={"518880": "gold"})[
            0
        ]
        == "518880"
    )
    assert select.__globals__["DROP_LOOKBACK"] == 5
    assert select.__globals__["A_SHARE_MA"] == 15


def test_review_refuses_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        review.run_comparison(tmp_path, tmp_path)


def test_review_rejects_changed_input(tmp_path):
    import json

    frozen = tmp_path / "frozen"
    (frozen / "inputs").mkdir(parents=True)
    (frozen / "inputs/518880.parquet").write_bytes(b"changed")
    (frozen / "inputs.json").write_text(
        json.dumps({"end": "2026-09-15", "input_sha256": {"518880": "wrong"}})
    )
    with pytest.raises(ValueError, match="Frozen input hash mismatch"):
        review.run_comparison(tmp_path / "result", frozen)


@pytest.mark.parametrize("version", ["v3", "v3_entry"])
def test_archived_selectors_causal_and_research_only(version, monkeypatch):
    spec = review.HISTORICAL[version]
    source = ROOT / "artifacts/version_comparison_20260916/historical_sources" / spec["file"]
    if not source.exists():
        pytest.skip("frozen historical source bundle required")
    pool = dict(review.live.ETF_POOL)
    selector = review.historical_selector(source, spec["sha256"], pool)
    monkeypatch.setattr(review.live, "select_target", selector)
    monkeypatch.setattr(
        review.live,
        "save_state_atomic",
        lambda *_: pytest.fail("research must not save real state"),
    )
    dates = pd.bdate_range("2025-01-01", periods=180).date.tolist()
    data = {}
    for n, code in enumerate([*pool, review.rq.DEFENSE]):
        prices = 10 + np.arange(len(dates)) * (n + 1) / 1000
        data[code] = pd.DataFrame(
            {
                "trade_date": dates,
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": 10000,
            }
        )
    grid = set(dates[130::5])
    kwargs = {"start": dates[130], "grid": grid, "mode": "V3-G"}
    full = review.replay_next_open(data, end=dates[-1], **kwargs)
    assert full["trades"]
    assert all(t["signal_date"] < t["date"] for t in full["trades"])
    cut = dates[160]
    prefix = review.replay_next_open(data, end=cut, **kwargs)
    for frame in data.values():
        frame.loc[frame.trade_date > cut, ["open", "close", "high", "low"]] *= 3
    changed = review.replay_next_open(data, end=dates[-1], **kwargs)
    for replay in (full, changed):
        pd.testing.assert_frame_equal(
            replay["curve"].query("trade_date <= @cut").reset_index(drop=True), prefix["curve"]
        )
    assert not any(d["v4_triggered"] for d in full["decisions"])
