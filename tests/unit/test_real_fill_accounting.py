"""Real execution facts: no model slippage, explicit fees and transaction safety."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import live_signal as ls


@pytest.fixture
def account(monkeypatch, tmp_path):
    monkeypatch.setattr(ls, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(ls, "STATE_TMP_FILE", tmp_path / "state.json.tmp")
    monkeypatch.setattr(ls, "LOCK_FILE", tmp_path / "state.lock")

    def seed(holding=None, shares=0, cash=10000):
        state = ls.default_state(cash)
        state.update(holding=holding, shares=shares, cash=cash, entry_price=5.0)
        ls.STATE_FILE.write_text(json.dumps(state))
        return state

    return seed


def test_real_fees_no_simulated_costs(account):
    state = account("518880", 1000, 100)
    state["pending_order"] = {
        "date": "2026-09-16",
        "status": "pending",
        "sell": {"code": "518880", "shares": 900},
        "buy": {"code": "159985", "shares": 2000},
    }
    ls.STATE_FILE.write_text(json.dumps(state))
    out = ls.confirm_order(
        {"shares": 1000, "price": 5.0, "fees": 3.0},
        {"code": "159985", "shares": 2000, "price": 2.0, "fees": 4.0},
    )
    assert out["cash"] == pytest.approx(1093)
    assert out["trade_log"][0]["gross_amount"] == 5000
    assert out["trade_log"][1]["fees"] == 4
    assert out["trade_log"][1]["fee_status"] == "reported"


def test_unreported_fees_explicitly_not_broker_reconciled(account):
    account()
    out = ls.record_manual_trade("buy", "518880", 100, 5.0)
    assert out["cash"] == 9500
    assert out["trade_log"][-1]["fee_status"] == "unreported"
    assert out["trade_log"][-1]["fees"] is None


def test_same_asset_add_and_partial_sale_preserve_position(account):
    account("518880", 100, 1000)
    out = ls.record_manual_trade("buy", "518880", 100, 6.0, fees=2.0)
    assert out["shares"] == 200
    assert out["entry_price"] == 5.5
    out = ls.record_manual_trade("sell", "518880", 150, 7.0, fees=1.0)
    assert out["shares"] == 50 and out["holding"] == "518880"
    assert out["cash"] == 1447


@pytest.mark.parametrize("fees", [-1, float("nan"), float("inf")])
def test_invalid_fees_atomic(account, fees):
    account()
    before = ls.STATE_FILE.read_bytes()
    with pytest.raises(ValueError):
        ls.record_manual_trade("buy", "518880", 100, 5.0, fees=fees)
    assert ls.STATE_FILE.read_bytes() == before


@pytest.mark.parametrize("price", [float("nan"), float("inf"), -1])
def test_nonfinite_core_price_rejected(account, price):
    account()
    before = ls.STATE_FILE.read_bytes()
    with pytest.raises(ValueError):
        ls.record_manual_trade("buy", "518880", 100, price)
    assert ls.STATE_FILE.read_bytes() == before


def test_different_holding_rejected_inside_transaction(account):
    account("518880", 100, 10000)
    before = ls.STATE_FILE.read_bytes()
    with pytest.raises(ValueError, match="持仓"):
        ls.record_manual_trade("buy", "159985", 100, 2.0, fees=0)
    assert ls.STATE_FILE.read_bytes() == before


def test_historical_restricted_fill_is_fact_not_pool_permission(account):
    account()
    kwargs = {
        "td": "2026-09-14",
        "fees": 2.0,
        "historical_fill": True,
        "evidence_confirmed": True,
        "idempotency_key": "historical-fill-1",
    }
    out = ls.record_manual_trade("buy", "501018", 100, 2.0, **kwargs)
    assert out["cash"] == 9798
    assert "501018" not in ls.ETF_POOL
    assert out["trade_log"][-1]["source"] == "user_attested_execution"
    before = ls.STATE_FILE.read_bytes()
    with pytest.raises(ValueError, match="重复"):
        ls.record_manual_trade("buy", "501018", 100, 2.0, **kwargs)
    assert ls.STATE_FILE.read_bytes() == before


def test_historical_requires_evidence_fees_and_chronology(account):
    state = account()
    with pytest.raises(ValueError):
        ls.record_manual_trade("buy", "501018", 100, 2.0, historical_fill=True)
    state["trade_log"] = [{"date": "2026-09-15", "action": "sell"}]
    ls.STATE_FILE.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="早于"):
        ls.record_manual_trade(
            "buy",
            "501018",
            100,
            2.0,
            "2026-09-14",
            fees=0,
            historical_fill=True,
            evidence_confirmed=True,
            idempotency_key="old",
        )


def test_manual_partial_fill_blocks_whole_plan_confirmation(account):
    state = account("518880", 1000, 100)
    state["pending_order"] = {
        "date": "2026-09-16",
        "status": "pending",
        "sell": {"code": "518880", "shares": 1000},
        "buy": None,
    }
    ls.STATE_FILE.write_text(json.dumps(state))
    out = ls.record_manual_trade("sell", "518880", 100, 5.0, fees=1.0)
    assert out["pending_order"]["status"] == "pending"
    assert out["pending_order"]["has_manual_fills"] is True
    with pytest.raises(ValueError, match="手工成交"):
        ls.confirm_order({"shares": 900, "price": 5.0, "fees": 0.0}, None)


def test_equity_rebuild_handles_add_and_partial(account, monkeypatch):
    state = account()
    state["trade_log"] = [
        {"date": "2026-09-14", "action": "buy", "code": "518880", "shares": 100, "amount": 502},
        {"date": "2026-09-14", "action": "buy", "code": "518880", "shares": 100, "amount": 602},
        {"date": "2026-09-15", "action": "sell", "code": "518880", "shares": 150, "amount": 1049},
    ]
    monkeypatch.setattr(ls, "get_trading_dates", lambda data: ["2026-09-14", "2026-09-15"])
    monkeypatch.setattr(ls, "price_on", lambda *args: 7.0)
    curve = ls.build_equity_curve(state, {"518880": object()})
    assert curve[0]["value"] == 10296
    assert curve[1]["value"] == 10295


def test_fractional_actual_shares_atomic(account):
    state = account("518880", 100, 100)
    state["pending_order"] = {
        "date": "2026-09-16",
        "status": "pending",
        "sell": {"code": "518880", "shares": 100},
        "buy": None,
    }
    ls.STATE_FILE.write_text(json.dumps(state))
    before = ls.STATE_FILE.read_bytes()
    with pytest.raises(ValueError):
        ls.confirm_order({"shares": 100.1, "price": 5, "fees": 0}, None)
    assert ls.STATE_FILE.read_bytes() == before


def test_confirm_never_evicts_manual_receipts(account):
    state = account()
    state["confirm_receipts"] = {
        "manual:old": {"kind": "manual_execution"},
        **{f"confirm-{i}": {} for i in range(100)},
    }
    state["pending_order"] = {
        "date": "2026-09-16",
        "status": "pending",
        "sell": None,
        "buy": {"code": "518880", "shares": 100},
    }
    ls.STATE_FILE.write_text(json.dumps(state))
    out = ls.confirm_order(
        None,
        {"code": "518880", "shares": 100, "price": 5, "fees": 0},
        idempotency_key="new-confirm",
    )
    assert "manual:old" in out["confirm_receipts"]


@pytest.mark.parametrize("td", ["20260914", "2026-9-14"])
def test_execution_date_requires_canonical_iso(account, td):
    account()
    before = ls.STATE_FILE.read_bytes()
    with pytest.raises(ValueError):
        ls.record_manual_trade("buy", "518880", 100, 5, td, fees=0)
    assert ls.STATE_FILE.read_bytes() == before
