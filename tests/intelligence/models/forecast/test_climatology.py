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


def test_exclude_cell_removes_the_station_from_every_scope():
    # a ward containing exactly ONE station is the common case, and there the
    # "ward fallback" IS that station's own history in disguise. Blanking only
    # the cell-scope lookup (what features.py used to do) leaves the leak intact.
    tables = build_climatology(_panel(), exclude_cell="A")

    for name, table in tables.items():
        assert len(table) == 0, f"{name} still carries the excluded station's history"


def test_exclude_cell_keeps_other_stations():
    two = pd.concat([_panel(), _panel().assign(cell="B", pm25_station=200.0)], ignore_index=True)
    tables = build_climatology(two, exclude_cell="A")

    assert "A" not in tables["cell_dow_hour"].index.get_level_values("cell")
    assert "B" in tables["cell_dow_hour"].index.get_level_values("cell")
    # the ward blend is now B alone, not a 40-ish/200 mix that still contains A
    assert tables["ward_dow_hour"].min() == pytest.approx(200.0)


def test_attach_climatology_uses_only_the_panel_it_is_given():
    """I3: a test row's climatology must never be computed from that row's own
    future. Week 1 reads 40, week 2 reads 200 — a full-panel climatology sees
    both (median 120) in every hour-of-week bucket; a train-only one sees 40."""
    from shared.grid import city_cells
    from intelligence.models.forecast.features import build_features, attach_climatology

    cell = city_cells()[0]
    hours = pd.date_range("2024-01-01", periods=336, freq="h", tz="UTC")
    split = hours[167]
    rows = [{"cell": cell, "ts": h, "ward_id": "W1", "ward_name": "Ward 1", "city": "bengaluru",
             "pm25_station": 40.0 if h <= split else 200.0,
             "wind_from_deg": 90.0, "wind_ms": 2.0, "blh_m": 400.0, "temp_c": 27.0,
             "fires_6h": 0, "frp_6h": 0.0, "lu_industrial": 0, "lu_construction": 0,
             "lu_waste_burning": 0, "lu_traffic": 0, "lu_road": 1, "lu_sensitive": 0,
             "hour": h.hour, "dow": h.dayofweek} for h in hours]
    panel = pd.DataFrame(rows)

    frame = build_features(panel, horizons=[3])
    leaky = frame[frame.ts > split]["clim_dow_hour"]
    honest = attach_climatology(frame, build_climatology(panel[panel.ts <= split]))
    honest = honest[honest.ts > split]["clim_dow_hour"]

    assert leaky.max() > 100          # full-panel climatology saw the test window
    assert honest.max() == pytest.approx(40.0)   # train-only cannot have
