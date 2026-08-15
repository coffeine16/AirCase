import pytest
import numpy as np
import pandas as pd

from intelligence.models.forecast.climatology import build_climatology, lookup_climatology


def _panel():
    hours = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    return pd.DataFrame({
        "cell": ["A"] * 72, "ward_id": ["W1"] * 72, "city": ["bengaluru"] * 72,
        "ts": hours, "pm25_station": [40.0 + (i % 24) for i in range(72)],
    })


def test_build_climatology_has_all_scales():
    tables = build_climatology(_panel())
    assert set(tables) == {"cell_dow_hour", "cell_month", "ward_dow_hour",
                            "ward_month", "city_dow_hour", "city_month"}


def test_lookup_prefers_cell_then_falls_back():
    tables = build_climatology(_panel())
    ts = pd.Timestamp("2024-01-02T05:00:00", tz="UTC")

    cell_val = lookup_climatology(tables, "A", "W1", "bengaluru", ts, scale="dow_hour")
    assert cell_val == pytest.approx(45.0)   # hour 5 -> 40 + 5

    unseen_cell_val = lookup_climatology(tables, "ZZZ", "W1", "bengaluru", ts, scale="dow_hour")
    assert unseen_cell_val == pytest.approx(45.0)   # falls back to ward, same data

    unseen_everything = lookup_climatology(tables, "ZZZ", "WZZZ", "chennai", ts, scale="dow_hour")
    assert np.isnan(unseen_everything)   # no city match either -> honestly NaN, not a guess
