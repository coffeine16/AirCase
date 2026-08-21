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


def test_month_scale_smooths_across_the_calendar_month_boundary():
    # Regression test for the climatology month-boundary fix, using a
    # SMOOTH ramp (the real motivating case -- Delhi's stubble-burning
    # season ramps continuously, it does not step) rather than an
    # artificial step function. A step function is the adversarial worst
    # case for ANY median-based smoother (median is a majority-vote
    # statistic, not an averaging one, so it cannot blend a genuine
    # discontinuity) and would fail this kind of assertion regardless of
    # whether the month-boundary bug is fixed -- confirmed directly: a
    # 50-vs-150 step still returns the hard 50.0 at Nov-30 here, because
    # its 31-day window has a 16-vs-15 day majority on the Nov side. That
    # is mathematically correct median behaviour, not a bug, and not what
    # this fix is for. A LINEAR ramp is the right adversarial case for a
    # boundary-ARTIFACT test instead: for a symmetric window over linear
    # data, the median of the window equals the value at its own center
    # exactly, so a correctly-smoothed climatology should show NO special
    # jump at the calendar-month seam specifically -- just the ramp's own
    # constant day-to-day rate, same as everywhere else in the range.
    hours = pd.date_range("2024-10-01", "2025-01-31 23:00", freq="h", tz="UTC")
    start = hours[0]
    panel = pd.DataFrame({
        "cell": ["A"] * len(hours), "ward_id": ["W1"] * len(hours), "city": ["bengaluru"] * len(hours),
        "ts": hours, "pm25_station": [50.0 + 1.5 * (h - start).days for h in hours],
    })
    tables = build_climatology(panel)

    nov30 = lookup_climatology(tables, "A", "W1", "bengaluru",
                                pd.Timestamp("2024-11-30T12:00:00", tz="UTC"), scale="month")
    dec1 = lookup_climatology(tables, "A", "W1", "bengaluru",
                               pd.Timestamp("2024-12-01T12:00:00", tz="UTC"), scale="month")
    assert np.isfinite(nov30) and np.isfinite(dec1)
    # The true underlying ramp moves exactly 1.5/day. A hard calendar-month
    # bucket would instead jump by (Dec's month-median - Nov's month-median)
    # here ~45.75 -- 30x the real one-day change. The fix should reproduce
    # the ramp's own rate at the boundary, not a month's worth of drift.
    assert dec1 - nov30 == pytest.approx(1.5, abs=0.5)
