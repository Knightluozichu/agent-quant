"""Input safety for frozen oil-proxy research; no account eligibility inference."""

from datetime import date

import pandas as pd
import pytest
from scripts.review_oil_proxy_20260916 import prepare_bars


def test_prepare_bars_clips_future_and_rejects_bad_or_duplicate_prices():
    frame = pd.DataFrame(
        {
            "date": ["2026-09-14", "2026-09-15", "2026-09-16"],
            "open": [1.0, 1.0, 2.0],
            "close": [1.0, 1.1, 2.0],
            "high": [1.0, 1.1, 2.0],
            "low": [1.0, 1.0, 2.0],
            "volume": [100, 200, 300],
        }
    )
    result = prepare_bars(frame, "561360", date(2026, 9, 15))
    assert len(result) == 2
    assert result.symbol.tolist() == ["561360", "561360"]
    with pytest.raises(ValueError, match="duplicate"):
        prepare_bars(pd.concat([frame, frame]), "561360", date(2026, 9, 15))
    frame.loc[0, "open"] = 0
    with pytest.raises(ValueError, match="OHLC"):
        prepare_bars(frame, "561360", date(2026, 9, 15))


def test_prepare_bars_requires_observed_cutoff():
    frame = pd.DataFrame(
        {
            "date": ["2026-09-14"],
            "open": [1.0],
            "close": [1.0],
            "high": [1.0],
            "low": [1.0],
            "volume": [100],
        }
    )
    with pytest.raises(ValueError, match="cutoff"):
        prepare_bars(frame, "561360", date(2026, 9, 15))
