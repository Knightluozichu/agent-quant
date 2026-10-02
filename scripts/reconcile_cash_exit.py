"""Reconcile a user-reported full exit; dry-run unless --apply. No broker access."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class CashExit:
    code: str
    trade_date: str
    shares: int
    price: Decimal
    gross_amount: Decimal
    cash_after: Decimal
    reconciliation_id: str

    def validate(self) -> None:
        if not self.code.isdigit() or len(self.code) != 6 or not self.reconciliation_id:
            raise ValueError("Invalid code or reconciliation ID")
        if self.shares <= 0 or not isinstance(self.shares, int):
            raise ValueError("Shares must be a positive integer")
        for value in (self.price, self.gross_amount, self.cash_after):
            if not value.is_finite() or value < 0:
                raise ValueError("Amounts must be finite and nonnegative")
        if self.price <= 0 or self.gross_amount <= 0:
            raise ValueError("Price and gross amount must be positive")
        if (self.price * self.shares).quantize(Decimal("0.01")) != self.gross_amount:
            raise ValueError("Price, shares and reported gross amount disagree")
        if date.fromisoformat(self.trade_date) > datetime.now(ZoneInfo("Asia/Shanghai")).date():
            raise ValueError("Trade date cannot be in the future")


def reconcile(state: dict[str, Any], fill: CashExit) -> dict[str, Any]:
    """Use the existing legacy state schema; preserve history and risk high-water mark."""
    fill.validate()
    receipt = {k: str(v) for k, v in asdict(fill).items()}
    receipts = state.get("cash_exit_reconciliations", {})
    if fill.reconciliation_id in receipts:
        if receipts[fill.reconciliation_id] != receipt:
            raise ValueError("Reconciliation ID conflict")
        return copy.deepcopy(state)
    if state.get("holding") != fill.code or state.get("shares") != fill.shares:
        raise ValueError("Reported exit does not match the complete recorded holding")
    if state.get("entry_date") and state["entry_date"] > fill.trade_date:
        raise ValueError("Exit predates the recorded entry")
    if any(t.get("date", "") > fill.trade_date for t in state.get("trade_log", [])):
        raise ValueError("Later trades exist; reconcile the complete ledger instead")
    old_cash = Decimal(str(state["cash"]))
    if not old_cash.is_finite() or old_cash < 0:
        raise ValueError("Invalid prior cash")
    result = copy.deepcopy(state)
    net_change = fill.cash_after - old_cash
    adjustment = net_change - fill.gross_amount
    result.setdefault("trade_log", []).append(
        {
            "date": fill.trade_date,
            "action": "sell",
            "code": fill.code,
            "shares": fill.shares,
            "price": float(fill.price),
            "gross_amount": float(fill.gross_amount),
            "amount": float(net_change),
            "unclassified_cash_adjustment": float(adjustment),
            "cash_before": float(old_cash),
            "cash_after": float(fill.cash_after),
            "reconciliation_id": fill.reconciliation_id,
            "note": "User-reported full exit and cash; residual is unclassified, not assumed fees. "
            "Execution time and broker commission were not supplied. No buy executed.",
        }
    )
    pending = result.get("pending_order")
    if pending and pending.get("status") == "pending":
        pending.update(
            status="superseded",
            reconciliation_id=fill.reconciliation_id,
            superseded_reason="Actual full exit; proposed buy not executed",
        )
    result.update(
        cash=float(fill.cash_after),
        holding=None,
        shares=0,
        entry_price=0.0,
        entry_date=None,
        h3_holding=None,
        h3_peak=0.0,
        account_reported_total=float(fill.cash_after),
        cash_reported=float(fill.cash_after),
        account_reported_at=fill.trade_date,
        cash_reported_at=fill.trade_date,
    )
    result.setdefault("v4_state", {})["candidate_history"] = []
    result.setdefault("cash_exit_reconciliations", {})[fill.reconciliation_id] = receipt
    return result


def main() -> None:
    import live_signal  # existing file-lock/backup/atomic-write implementation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", required=True)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--shares", type=int, required=True)
    parser.add_argument("--price", type=Decimal, required=True)
    parser.add_argument("--gross-amount", type=Decimal, required=True)
    parser.add_argument("--cash-after", type=Decimal, required=True)
    parser.add_argument("--reconciliation-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = vars(parser.parse_args())
    apply = args.pop("apply")
    fill = CashExit(**args)
    before = live_signal.load_state()
    if before is None:
        raise ValueError("Account not initialized")
    after = reconcile(before, fill)
    changed = after != before
    if apply and changed:
        with live_signal.state_transaction() as current:
            updated = reconcile(current, fill)
            current.clear()
            current.update(updated)
    print(
        json.dumps(
            {
                "applied": apply and changed,
                "changed": changed,
                "cash": after["cash"],
                "holding": after["holding"],
                "shares": after["shares"],
                "last_trade": after["trade_log"][-1],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
