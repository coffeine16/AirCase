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


def _panel_with_varying_weather():
    """Same shape as _tiny_panel but weather actually varies by hour --
    _tiny_panel's constant weather can't distinguish an issue-time lookup
    from a target-time one, since both would return the same value."""
    cells = city_cells()[:3]
    hours = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    rows = []
    for i, c in enumerate(cells):
        for h in hours:
            hour_idx = int((h - hours[0]) / pd.Timedelta(hours=1))
            rows.append({
                "cell": c, "ts": h, "ward_id": "W1", "ward_name": "Ward 1",
                "city": "bengaluru",
                "pm25_station": 50.0 + i * 10 if i < 2 else np.nan,
                # Distinct per-hour value, easy to trace: temp_c == hour index.
                "wind_from_deg": 90.0, "wind_ms": 2.0,
                "blh_m": 300.0 + hour_idx, "temp_c": float(hour_idx),
                "fires_6h": 0, "frp_6h": 0.0,
                "lu_industrial": 0, "lu_construction": 0, "lu_waste_burning": 0,
                "lu_traffic": 0, "lu_road": 1, "lu_sensitive": 0,
                "hour": h.hour, "dow": h.dayofweek,
            })
    return pd.DataFrame(rows), hours


def _wind_test_cells():
    """A query cell with one REAL near station (~0.9km) and one REAL far
    station (~5.7km) -- picked by actual haversine distance, not just list
    order, since two of city_cells()'s first three entries turned out to be
    almost exactly equidistant from the third (a real trap: widening the
    decay scale by the SAME factor for two already-near-equal distances
    doesn't change their weight RATIO at all, so that geometry could never
    show this effect regardless of whether the wiring works)."""
    from shared.grid import neighbors, cell_center, haversine_km
    cells = city_cells()
    center = cells[len(cells) // 2]
    near = neighbors(center, k=1)[0]
    candidates = neighbors(center, k=6)
    c0 = cell_center(center)
    far = max(candidates, key=lambda c: haversine_km(*c0, *cell_center(c)))
    return center, near, far


def _panel_with_wind_ms(wind_ms):
    center, near, far = _wind_test_cells()
    cells = [near, far, center]   # station, station, non-station query cell
    hours = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    rows = []
    for i, c in enumerate(cells):
        for h in hours:
            rows.append({
                "cell": c, "ts": h, "ward_id": "W1", "ward_name": "Ward 1",
                "city": "bengaluru",
                "pm25_station": 20.0 + i * 60.0 if i < 2 else np.nan,   # far-apart values: 20.0, 80.0
                # wind_from_deg=29.57 (verified numerically) puts the FAR
                # station exactly downwind of the query cell -- wind speed
                # widens reach ONLY along the downwind axis (the physics
                # fix), so the geometry has to actually BE downwind for
                # this test to mean anything; an arbitrary fixed bearing
                # (the original 90.0) is no longer guaranteed to show any
                # effect, and isn't supposed to be.
                "wind_from_deg": 29.57, "wind_ms": wind_ms, "blh_m": 400.0, "temp_c": 27.0,
                "fires_6h": 0, "frp_6h": 0.0,
                "lu_industrial": 0, "lu_construction": 0, "lu_waste_burning": 0,
                "lu_traffic": 0, "lu_road": 1, "lu_sensitive": 0,
                "hour": h.hour, "dow": h.dayofweek,
            })
    return pd.DataFrame(rows), cells


def test_build_features_wind_speed_changes_the_composite_fill():
    """End-to-end proof the wind_ms wiring reaches production, not just
    spatial.py's own unit tests: a non-station cell's composite-filled lag
    should differ between a calm and a windy panel, since its two real
    stations (20.0 and 80.0, deliberately far apart) sit at DIFFERENT real
    distances from it (~0.9km and ~5.7km) and wind speed widens reach
    ALONG THE DOWNWIND AXIS specifically (the physics fix -- speed no
    longer widens reach isotropically). wind_from_deg is chosen (29.57,
    verified numerically) to put the FAR station exactly downwind, so this
    geometry actually exercises the effect instead of relying on an
    arbitrary bearing that might land crosswind, where wind speed is
    correctly supposed to have no effect at all."""
    calm_panel, cells = _panel_with_wind_ms(0.2)
    windy_panel, _ = _panel_with_wind_ms(20.0)

    calm = build_features(calm_panel, horizons=[3])
    windy = build_features(windy_panel, horizons=[3])

    non_station_cell = cells[2]
    calm_val = calm[(calm.cell == non_station_cell) & (calm.horizon == 3)]["lag_0"].dropna()
    windy_val = windy[(windy.cell == non_station_cell) & (windy.horizon == 3)]["lag_0"].dropna()
    assert len(calm_val) > 0 and len(windy_val) > 0
    assert not np.allclose(calm_val.values, windy_val.values), \
        "wind_ms must actually reach composite_grid through build_features, not just spatial.py's own tests"
    # the far (80.0) station should count for MORE under high wind, pulling
    # the blend UP -- not just "different", but different in the physically
    # correct direction.
    assert windy_val.mean() > calm_val.mean()


def test_build_features_reads_weather_at_target_time_not_issue_time():
    """A1: _MET must be looked up at ts+horizon, not ts. temp_c == hour
    index in this fixture, so a row issued at hour 10 with horizon=6 must
    carry temp_c == 16 (hour 16's value) -- if the lookup were still at
    issue time (the pre-A1 behaviour), it would carry temp_c == 10, and if
    it silently fell back to some other hour, it would carry neither."""
    panel, hours = _panel_with_varying_weather()
    frame = build_features(panel, horizons=[6], restrict_to_station_cells=True)

    # hour 30: far enough in for lag_24 (dropna requires it), target (+6=36)
    # still safely inside the 72h panel.
    issue_hour = hours[30]
    row = frame[frame.ts == issue_hour].iloc[0]
    assert row["horizon"] == 6
    assert row["temp_c"] == 36.0          # target hour's value (30 + 6)
    assert row["temp_c"] != 30.0          # NOT the issue hour's value
    assert row["blh_m"] == 336.0          # 300 + 36, same target-time rule


def test_build_features_weather_is_nan_past_the_panels_own_coverage():
    """A row near the panel's tail whose target time (ts+horizon) falls
    outside the panel's own hourly range must get NaN weather, not a stale
    or wrapped-around value -- the same honest-missing convention `y`
    already uses for labels that shift past the panel's end."""
    panel, hours = _panel_with_varying_weather()
    frame = build_features(panel, horizons=[6], restrict_to_station_cells=True)

    last_issue_hour = hours[-1]
    row = frame[frame.ts == last_issue_hour].iloc[0]
    assert pd.isna(row["temp_c"])
    assert pd.isna(row["blh_m"])


def test_feature_columns_uses_cyclical_encoding_not_raw_periodic_values():
    """target_hour/dow/month/doy are periodic; the MODEL must only ever see
    their sin/cos pairs, never the raw ordinal value -- a raw encoding puts
    23:00 and 00:00 numerically far apart when they are adjacent in
    reality."""
    for raw in ("target_hour", "target_dow", "target_month", "target_doy"):
        assert raw not in FEATURE_COLUMNS, f"{raw} must not be a raw model feature"
    for cyc in ("target_hour_sin", "target_hour_cos"):
        assert cyc in FEATURE_COLUMNS, f"missing cyclical feature {cyc}"


def test_feature_columns_uses_wind_vector_not_separate_speed_and_direction():
    """wind is a VECTOR (speed and direction together), not two
    independently-useful scalars -- the model must see the fused wind_u/
    wind_v components, never wind_ms or wind_from_deg (raw or cyclical) as
    separate features it would have to learn the interaction between."""
    for raw in ("wind_ms", "wind_from_deg", "wind_from_deg_sin", "wind_from_deg_cos"):
        assert raw not in FEATURE_COLUMNS, f"{raw} must not be a separate model feature"
    for vec in ("wind_u", "wind_v"):
        assert vec in FEATURE_COLUMNS, f"missing wind vector feature {vec}"


def test_wind_u_v_match_the_standard_meteorological_formula():
    """Not just present -- actually computed correctly. wind_from_deg=90
    (wind FROM the east, blowing due west) at wind_ms=2.0 must give
    wind_u=-2.0 (westward = negative eastward component), wind_v=0.0 (no
    north/south component) -- the standard u=-speed*sin(from), v=-speed*
    cos(from) convention, not some other sign or axis choice."""
    frame = build_features(_tiny_panel(), horizons=[3])
    row = frame.iloc[0]
    assert row["wind_u"] == pytest.approx(-2.0, abs=1e-9)
    assert row["wind_v"] == pytest.approx(0.0, abs=1e-9)


def test_target_hour_23_and_0_land_close_in_cyclical_space():
    """The wrap-adjacency proof, same shape as fusion's: hour 23 and hour 0
    must be close in (sin, cos) space despite being 23 apart as raw
    integers, and clearly closer than two hours on opposite sides of the
    clock (23 vs 12)."""
    panel, hours = _panel_with_varying_weather()
    frame = build_features(panel, horizons=[0], restrict_to_station_cells=True)

    def row_at(hour_idx):
        return frame[frame.ts == hours[hour_idx]].iloc[0]

    r23, r0, r12 = row_at(47), row_at(48), row_at(36)   # hour-of-day 23, 0, 12
    assert r23["target_hour"] == 23 and r0["target_hour"] == 0 and r12["target_hour"] == 12

    def dist(a, b):
        return np.hypot(a["target_hour_sin"] - b["target_hour_sin"],
                         a["target_hour_cos"] - b["target_hour_cos"])

    assert dist(r23, r0) < 0.3
    assert dist(r23, r12) > 1.5


def test_build_features_has_trust_and_spatial_columns():
    frame = build_features(_tiny_panel(), horizons=[3, 6])

    for col in ("has_station", "distance_to_nearest_station_km", "nearby_stations_delta",
                "pos_0", "pos_1", "pos_2", "pos_3", "pos_4", "pos_5", "pos_6",
                "fire_pressure_regional", "clim_dow_hour", "clim_month",
                "target_hour", "target_dow", "target_month", "target_doy",
                "horizon", "city", "y"):
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


def test_restrict_to_station_cells_matches_the_unrestricted_labeled_rows():
    """restrict_to_station_cells=True exists purely to avoid materializing
    horizon-expanded rows that .dropna(subset=["y"]) was always going to
    discard anyway (see the real multi-city panel this was measured against:
    ~50M base rows, <40 station cells, 27GB to horizon-expand unrestricted).
    It must never change WHICH labeled rows a training call sees -- only how
    much gets built along the way."""
    panel = _tiny_panel()
    horizons = [3, 24]

    unrestricted = build_features(panel, horizons).dropna(subset=["y"])
    restricted = build_features(panel, horizons, restrict_to_station_cells=True).dropna(subset=["y"])

    assert not restricted.empty, "restricting to station cells dropped every labeled row"
    key_cols = ["cell", "ts", "horizon"]
    u = unrestricted.sort_values(key_cols).reset_index(drop=True)
    r = restricted.sort_values(key_cols).reset_index(drop=True)
    assert u[key_cols].equals(r[key_cols]), "restricting changed WHICH rows survive to training"
    assert np.allclose(u["y"], r["y"])
    assert np.allclose(u["lag_0"], r["lag_0"], equal_nan=True)

    # and it must not disturb loso_exclude's own held-out rows, which are the
    # whole reason spatial_loso passes both flags together
    held_out = panel.cell.iloc[0]
    loso_unrestricted = build_features(panel, horizons, loso_exclude=held_out)
    loso_restricted = build_features(panel, horizons, loso_exclude=held_out,
                                      restrict_to_station_cells=True)
    held_u = loso_unrestricted[loso_unrestricted.cell == held_out].sort_values("horizon")
    held_r = loso_restricted[loso_restricted.cell == held_out].sort_values("horizon")
    assert len(held_u) == len(held_r) and len(held_r) > 0
    assert np.allclose(held_u["lag_0"].to_numpy(), held_r["lag_0"].to_numpy(), equal_nan=True)


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
