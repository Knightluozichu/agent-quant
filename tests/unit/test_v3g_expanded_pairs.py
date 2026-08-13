"""Structural tests for the 8+8 V3-G expanded-pool experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from exp_v3g_expanded_pairs import (  # noqa: E402
    CLASS_GROUPS,
    ORIGINAL_CODES,
    PEER_OF,
    apply_known_splits,
)


def test_every_original_has_one_unique_peer_and_groups_cover_sixteen() -> None:
    assert set(PEER_OF) == set(ORIGINAL_CODES)
    assert len(set(PEER_OF.values())) == 8
    covered = {code for group in CLASS_GROUPS for code in group}
    assert covered == set(ORIGINAL_CODES) | set(PEER_OF.values())
    assert len(covered) == 16


def test_nasdaq_peer_split_is_adjusted_only_before_split_date() -> None:
    frame = pd.DataFrame({
        "trade_date": [
            pd.Timestamp("2022-07-04").date(),
            pd.Timestamp("2022-07-05").date(),
        ],
        "open": [2.4, 0.6],
        "high": [2.4, 0.6],
        "low": [2.4, 0.6],
        "close": [2.4, 0.6],
        "volume": [100.0, 400.0],
        "symbol": ["159941", "159941"],
    })

    adjusted = apply_known_splits("159941", frame)

    assert adjusted.loc[0, "close"] == 0.6
    assert adjusted.loc[0, "volume"] == 400.0
    assert adjusted.loc[1, "close"] == 0.6
    assert adjusted.loc[1, "volume"] == 400.0
