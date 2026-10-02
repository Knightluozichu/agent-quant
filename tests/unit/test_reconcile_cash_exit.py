"""Manual cash reconciliation must never manufacture a buy or estimated fee."""

from copy import deepcopy
from decimal import Decimal

import pytest
from scripts.reconcile_cash_exit import CashExit, reconcile


def sample():
    return {
        "holding": "159985",
        "shares": 42400,
        "cash": 61.48,
        "entry_price": 2.363,
        "entry_date": "2026-09-01",
        "trade_log": [],
        "pending_order": {"status": "pending", "buy": {"code": "501018"}},
        "v4_state": {"candidate_history": [{"date": "2026-09-09"}]},
        "peak_equity": 105000,
        "last_run_date": "2026-09-10",
    }


def fill(**changes):
    values = {
        "code": "159985",
        "trade_date": "2026-09-14",
        "shares": 42400,
        "price": Decimal("2.303"),
        "gross_amount": Decimal("97647.20"),
        "cash_after": Decimal("97684.45"),
        "reconciliation_id": "sell-20260914",
    }
    return CashExit(**(values | changes))


def test_reconcile_reported_cash_without_buy_or_synthetic_fee():
    before = sample()
    original = deepcopy(before)
    result = reconcile(before, fill())
    assert before == original
    assert result["cash"] == 97684.45
    assert result["holding"] is None and result["shares"] == 0
    assert result["pending_order"]["status"] == "superseded"
    assert result["trade_log"][0]["gross_amount"] == 97647.20
    assert result["trade_log"][0]["amount"] == 97622.97
    assert result["trade_log"][0]["unclassified_cash_adjustment"] == -24.23
    assert "commission" not in result["trade_log"][0]
    assert result["v4_state"]["candidate_history"] == []
    assert result["peak_equity"] == original["peak_equity"]
    assert result["last_run_date"] == original["last_run_date"]
    assert reconcile(result, fill()) == result
    with pytest.raises(ValueError, match="conflict"):
        reconcile(result, fill(cash_after=Decimal("97000")))


@pytest.mark.parametrize(
    "changes",
    [
        {"gross_amount": Decimal("1")},
        {"price": Decimal("NaN")},
        {"cash_after": Decimal("Infinity")},
        {"cash_after": Decimal("-1")},
        {"shares": 42000},
        {"code": "513350"},
        {"trade_date": "2099-01-01"},
    ],
)
def test_invalid_reconciliation_rejected(changes):
    with pytest.raises(ValueError):
        reconcile(sample(), fill(**changes))


def test_cli_dry_run_apply_backup_and_retry(monkeypatch, tmp_path, capsys):
    import json
    import sys

    from scripts import live_signal, reconcile_cash_exit

    state_file = tmp_path / "state.json"
    original = live_signal.default_state(100000) | sample()
    state_file.write_text(json.dumps(original))
    monkeypatch.setitem(sys.modules, "live_signal", live_signal)
    monkeypatch.setattr(live_signal, "STATE_FILE", state_file)
    monkeypatch.setattr(live_signal, "STATE_TMP_FILE", tmp_path / "state.json.tmp")
    monkeypatch.setattr(live_signal, "LOCK_FILE", tmp_path / "quant_state.lock")
    argv = [
        "reconcile_cash_exit",
        "--code",
        "159985",
        "--trade-date",
        "2026-09-14",
        "--shares",
        "42400",
        "--price",
        "2.303",
        "--gross-amount",
        "97647.20",
        "--cash-after",
        "97684.45",
        "--reconciliation-id",
        "sell-20260914",
    ]
    before = state_file.read_bytes()
    monkeypatch.setattr(sys, "argv", argv)
    reconcile_cash_exit.main()
    assert state_file.read_bytes() == before
    monkeypatch.setattr(sys, "argv", [*argv, "--apply"])
    reconcile_cash_exit.main()
    after = state_file.read_bytes()
    assert json.loads(after)["cash"] == 97684.45
    assert (tmp_path / "state.json.bak").read_bytes() == before
    reconcile_cash_exit.main()
    assert state_file.read_bytes() == after
    capsys.readouterr()
