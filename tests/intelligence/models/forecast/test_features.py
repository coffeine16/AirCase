# tests/intelligence/models/forecast/test_features.py
import numpy as np
import pandas as pd
import pytest

from shared.grid import city_cells
from intelligence.models.forecast.features import build_features, FEATURE_COLUMNS


def _tiny_panel():
    cells = city_cells()[:3]
    hours = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    rows = []
    for i, c in enumerate(cells):
        for h in hours:
            rows.append({
                "cell": c, "ts": h, "ward_id": "W1", "ward_name": "Ward 1",
                "city": "bengaluru",
                # TWO stations, not one: with a single station the excluded
                # cell's composite has nothing left to compose from and comes
                # back NaN, which satisfies the loso assertion below via its
                # isnan disjunct without ever exercising the `!=` half.
                "pm25_station": 50.0 + i * 10 if i < 2 else np.nan,
                "wind_from_deg": 90.0, "wind_ms": 2.0, "blh_m": 400.0, "temp_c": 27.0,
                "fires_6h": 0, "frp_6h": 0.0,
                "lu_industrial": 0, "lu_construction": 0, "lu_waste_burning": 0,
                "lu_traffic": 0, "lu_road": 1, "lu_sensitive": 0,
                "hour": h.hour, "dow": h.dayofweek,
            })
    return pd.DataFrame(rows)


def test_build_features_has_trust_and_spatial_columns():
    frame = build_features(_tiny_panel(), horizons=[3, 6])

    for col in ("has_station", "distance_to_nearest_station_km", "nearby_stations_delta",
                "pos_0", "pos_1", "pos_2", "pos_3", "pos_4", "pos_5", "pos_6",
                "fire_pressure_regional", "clim_dow_hour", "clim_month",
                "target_hour", "target_dow", "target_month", "horizon", "city", "y"):
        assert col in frame.columns, f"missing {col}"


def test_build_features_station_cell_has_station_true():
    frame = build_features(_tiny_panel(), horizons=[3])
    station_cell = city_cells()[0]
    rows = frame[frame.cell == station_cell]
    assert (rows.has_station == True).all()
    assert (rows.distance_to_nearest_station_km == 0.0).all()


def test_build_features_loso_exclude_masks_own_history():
    station_cell = city_cells()[0]
    normal = build_features(_tiny_panel(), horizons=[3])
    loso = build_features(_tiny_panel(), horizons=[3], loso_exclude=station_cell)

    normal_row = normal[normal.cell == station_cell].iloc[-1]
    loso_row = loso[loso.cell == station_cell].iloc[-1]
    # lag_0 must differ once the station is excluded from its own composite —
    # otherwise the LOSO test would trivially see its own real data. With the
    # second station present this is a real value (~60, the other station's),
    # not a vacuous NaN.
    assert np.isfinite(loso_row["lag_0"])
    assert normal_row["lag_0"] != loso_row["lag_0"]
    # roll_med_24/168 must ALSO be masked for the excluded station's own
    # rows -- these bypass the composite fill entirely (they're computed
    # straight from the raw per-cell groupby), so without an explicit null
    # a "held-out" station would still see the median of its own real
    # last-24h/168h history, defeating the point of loso_exclude.
    assert np.isnan(loso_row["roll_med_24"])
    assert np.isnan(loso_row["roll_med_168"])


def test_build_features_loso_exclude_falls_back_to_ward_climatology():
    # two stations in the same ward, different constant values: excluding
    # one, its OWN cell-level climatology entry must not leak through --
    # lookup_climatology's already-proven cell->ward->city fallback (Task 4)
    # should kick in, blending BOTH stations' history, not returning the
    # excluded station's own value verbatim.
    cells = city_cells()[:3]
    hours = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    rows = []
    for i, c in enumerate(cells):
        for h in hours:
            rows.append({
                "cell": c, "ts": h, "ward_id": "W1", "ward_name": "Ward 1", "city": "bengaluru",
                "pm25_station": 50.0 if i == 0 else (90.0 if i == 1 else np.nan),
                "wind_from_deg": 90.0, "wind_ms": 2.0, "blh_m": 400.0, "temp_c": 27.0,
                "fires_6h": 0, "frp_6h": 0.0, "lu_industrial": 0, "lu_construction": 0,
                "lu_waste_burning": 0, "lu_traffic": 0, "lu_road": 1, "lu_sensitive": 0,
                "hour": h.hour, "dow": h.dayofweek,
            })
    panel = pd.DataFrame(rows)
    station_cell = cells[0]

    loso = build_features(panel, horizons=[3], loso_exclude=station_cell)
    # NOT .iloc[-1]: with a 72h panel and horizon=3, the LAST row's target
    # time (ts + 3h) falls past the end of the panel entirely, so NO scope
    # (cell, ward, or city) has a climatology entry for it and the lookup
    # returns NaN regardless of whether the leak fix is present -- the
    # assertion would pass vacuously either way. .iloc[0] is the first row
    # surviving the lag_24 warm-up filter; its target time is well inside
    # the panel, so both scopes have real entries and the comparison
    # actually discriminates leak-vs-fallback.
    loso_row = loso[loso.cell == station_cell].iloc[0]

    # cell[0]'s own cell-level climatology is exactly 50.0 (its own constant
    # reading); if that leaked through unmasked, clim_dow_hour would be
    # exactly 50.0. It must not be -- the ward blend (50 and 90) differs.
    assert loso_row["clim_dow_hour"] != pytest.approx(50.0)
    assert loso_row["clim_month"] != pytest.approx(50.0)


def test_loso_cell_loses_its_station_identity_too():
    # loso_exclude masks the station's READINGS; it must mask its IDENTITY as
    # well. A held-out cell still reporting has_station=True / distance 0.0 is
    # a train/serve mismatch — a real station-less cell never looks like that.
    station_cell = city_cells()[0]
    loso = build_features(_tiny_panel(), horizons=[3], loso_exclude=station_cell)
    rows = loso[loso.cell == station_cell]

    assert not rows.has_station.any()
    assert (rows.distance_to_nearest_station_km > 0).all()


def test_build_features_wires_fires_into_fire_pressure():
    from shared.grid import cell_center

    panel = _tiny_panel()
    lat, lon = cell_center(city_cells()[0])
    # due EAST of the cell: the fire->cell bearing is 270, which is exactly
    # where a wind_from_deg=90 wind blows, so alignment is 1.0
    fires = pd.DataFrame({"ts": panel.ts.unique()[:24], "lat": lat, "lon": lon + 0.01,
                          "frp": 100.0, "confidence": 90})

    without = build_features(panel, horizons=[3])
    with_fires = build_features(panel, horizons=[3], fires=fires)

    assert (without.fire_pressure_regional == 0).all()
    assert with_fires.fire_pressure_regional.max() > 0


def test_build_features_stays_fast_on_a_moderate_panel():
    # Canary against the horizon-loop regression: build_features used to
    # recompute every composite, positional block and climatology lookup once
    # PER HORIZON, which extrapolated to ~90 minutes for one call on the real
    # operational panel. The bound is deliberately loose — this is here to
    # catch a 100x regression, not a 10% one.
    import time

    cells = city_cells()[:60]
    hours = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC")
    rows = [{"cell": c, "ts": h, "ward_id": "W1", "ward_name": "Ward 1", "city": "bengaluru",
             "pm25_station": 50.0 + i if i < 5 else np.nan,
             "wind_from_deg": 90.0, "wind_ms": 2.0, "blh_m": 400.0, "temp_c": 27.0,
             "fires_6h": 0, "frp_6h": 0.0, "lu_industrial": 0, "lu_construction": 0,
             "lu_waste_burning": 0, "lu_traffic": 0, "lu_road": 1, "lu_sensitive": 0,
             "hour": h.hour, "dow": h.dayofweek}
            for i, c in enumerate(cells) for h in hours]
    panel = pd.DataFrame(rows)

    # Full production horizon count (24), not a 4-horizon sample: the whole point
    # of the regression this guards against is the horizon LOOP re-doing the
    # horizon-independent work, so the canary has to pay that multiplier to be
    # able to see it. At 4 horizons the pre-fix path (~3.7s) and the fixed path
    # (~0.1s) both clear a loose bound and the test proves nothing.
    horizons = list(range(3, 73, 3))
    t0 = time.perf_counter()
    frame = build_features(panel, horizons=horizons)
    elapsed = time.perf_counter() - t0

    assert not frame.empty
    assert elapsed < 10.0, (
        f"build_features took {elapsed:.1f}s on 60 cells x 120 hours x {len(horizons)} "
        "horizons — the pre-fix per-horizon recomputation took ~22s on this same "
        "fixture, so this bound catches that regression coming back."
    )
