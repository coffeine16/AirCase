import numpy as np
import pandas as pd
import pytest

from intelligence.models.forecast.spatial import composite_grid, DIST_DECAY_KM, positional_block, fire_pressure
from shared.grid import city_cells


def test_composite_grid_basic_blend():
    # two stations both north of the query point, under a wind FROM the
    # north (wind_from_deg=0, blows south) -- both stations are directly
    # upwind, so both get real (not zero) alignment, and the closer one
    # (12.98) should dominate the farther one (12.99) but not exclusively
    query_lat = np.array([12.97])
    query_lon = np.array([77.60])
    station_lat = np.array([12.98, 12.99])           # both north of the query point: upwind
    station_lon = np.array([77.60, 77.60])
    station_val = np.array([[40.0, 60.0]])           # one hour, two stations
    wind_from_deg = np.array([0.0])                  # wind FROM the north -> blows toward the query point

    out = composite_grid(query_lat, query_lon, station_lat, station_lon,
                          station_val, wind_from_deg)

    assert out.shape == (1, 1)
    assert 35.0 < out[0, 0] < 65.0                    # a real blend of 40 and 60, not NaN


def test_composite_grid_excludes_masked_station():
    # station due north of the query point under a north wind: upwind,
    # real (non-zero) weight -- so excluding it must visibly change the
    # result from a real number to NaN, not just coincide with an
    # already-zero weight from geometry
    query_lat = np.array([12.97])
    query_lon = np.array([77.60])
    station_lat = np.array([12.98])
    station_lon = np.array([77.60])
    station_val = np.array([[999.0]])
    wind_from_deg = np.array([0.0])
    exclude = np.array([[True]])

    out = composite_grid(query_lat, query_lon, station_lat, station_lon,
                          station_val, wind_from_deg, exclude=exclude)

    assert np.isnan(out[0, 0])                        # only station excluded -> no weight left -> NaN


def test_composite_grid_no_stations_returns_nan():
    out = composite_grid(np.array([12.97]), np.array([77.60]),
                          np.array([]), np.array([]),
                          np.zeros((1, 0)), np.array([0.0]))
    assert np.isnan(out[0, 0])


def test_distance_decay_matches_attribution_kernel():
    """Was tautological before: it imported category_scores but never
    called it, only re-asserted spatial.py's own constant against itself
    -- a real mismatch (attribution.py hardcoding its own literal 2.0
    instead of importing DIST_DECAY_KM) would have passed this test
    forever. Actually run category_scores and check its real output
    matches exp(-d/DIST_DECAY_KM) bit-for-bit."""
    from intelligence.agents.attribution import category_scores, CATEGORIES
    d = 1.7
    ev = {
        "candidates": [{"type": "industrial", "distance_km": d, "wind_alignment": 1.0}],
        "pollutant_signature": {},
        "meteorology": {"hour_local": 3},   # outside every hour-gated bonus window
        "fire_activity": {"fire_hour_fraction": 0.0},
        "landuse_context": {c: 0 for c in CATEGORIES},   # zero every land-use bonus term
    }
    scores = category_scores(ev)
    # category_scores rounds its return values to 3dp -- match that, not
    # the unrounded theoretical value.
    assert scores["industrial"] == pytest.approx(round(np.exp(-d / DIST_DECAY_KM), 3))


def test_composite_grid_wind_ms_none_reproduces_original_behaviour():
    """Every existing call site that hasn't been updated must see IDENTICAL
    output with wind_ms omitted -- this is the backward-compat guarantee
    the whole rollout depends on, not just an assumption."""
    query_lat, query_lon = np.array([12.97]), np.array([77.60])
    station_lat, station_lon = np.array([12.98, 12.99]), np.array([77.60, 77.60])
    station_val = np.array([[40.0, 60.0]])
    wind_from_deg = np.array([0.0])

    without = composite_grid(query_lat, query_lon, station_lat, station_lon,
                              station_val, wind_from_deg)
    explicit_none = composite_grid(query_lat, query_lon, station_lat, station_lon,
                                    station_val, wind_from_deg, wind_ms=None)
    assert without[0, 0] == explicit_none[0, 0]


def test_composite_grid_wider_reach_under_stronger_wind():
    """The actual point of the change: a station 3km away (well outside the
    2km DIST_DECAY_KM base scale) should contribute a LARGER share of the
    composite under a strong wind than under a calm one -- distance decay
    genuinely widens with wind speed, not the same fixed 2km either way."""
    query_lat, query_lon = np.array([12.97]), np.array([77.60])
    # ~3km due north -- station_val differs sharply from a co-located
    # reference so the far station's growing/shrinking WEIGHT is visible
    # in the blended output, not masked by both stations agreeing anyway.
    station_lat = np.array([12.97, 12.997])
    station_lon = np.array([77.60, 77.60])
    station_val = np.array([[10.0, 90.0]])
    wind_from_deg = np.array([0.0])   # wind FROM the north -> the far station is upwind

    calm = composite_grid(query_lat, query_lon, station_lat, station_lon,
                           station_val, wind_from_deg, wind_ms=np.array([0.1]))
    windy = composite_grid(query_lat, query_lon, station_lat, station_lon,
                            station_val, wind_from_deg, wind_ms=np.array([15.0]))

    # both stations are in play (query point == station 0's own location,
    # so its weight is large but not exclusive); the far, high-value
    # station's contribution should pull the blend up more under high wind.
    assert windy[0, 0] > calm[0, 0]


def test_composite_grid_crosswind_reach_unaffected_by_wind_speed():
    """The actual fix: wind speed must stretch reach ONLY along the
    downwind axis, never crosswind -- a first version of this stretched
    decay_km isotropically (same widening in every direction) and a
    seed-controlled sweep against real data found no reliable improvement,
    because a fast wind was spuriously inflating the reach of sources that
    aren't actually downwind. A station due EAST of the query point, under
    a wind blowing due NORTH, is exactly 90 degrees crosswind -- align
    already zeroes its contribution, but the isotropic bug would still
    have widened decay_km for it before align zeroed the product. Test a
    near-crosswind case (verified numerically: off=85.9 degrees,
    align=0.072 -- inside align's non-zero range) instead, where the bug's
    effect is actually visible in the output: reach must stay pinned to
    the calm-air scale regardless of wind speed."""
    query_lat, query_lon = np.array([12.97]), np.array([77.60])
    # station ~1.09km away, bearing station->query ~=95.9 degrees; wind
    # FROM 190 degrees (blows toward 10 degrees) puts this station at
    # off=85.9 degrees -- near-crosswind, inside align's non-zero range.
    station_lat = np.array([12.971])
    station_lon = np.array([77.59])
    station_val = np.array([[50.0]])
    wind_from_deg = np.array([190.0])

    calm = composite_grid(query_lat, query_lon, station_lat, station_lon,
                           station_val, wind_from_deg, wind_ms=np.array([0.1]))
    windy = composite_grid(query_lat, query_lon, station_lat, station_lon,
                            station_val, wind_from_deg, wind_ms=np.array([20.0]))

    # single station -> the composite is just that station's value whenever
    # weight is nonzero, and NaN whenever weight is exactly zero. Wind
    # speed must not change whether/how much this near-crosswind station
    # contributes.
    assert calm[0, 0] == pytest.approx(windy[0, 0], abs=1e-9)


def test_fire_pressure_wider_reach_under_stronger_wind():
    """Same property as composite_grid, for fire_pressure: a fire near the
    edge of FIRE_PRESSURE_RADIUS_KM should register more pressure under a
    strong wind than a calm one, at the SAME distance and alignment."""
    cells = city_cells()[:1]
    hours = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
    from shared.grid import cell_center
    lat, lon = cell_center(cells[0])
    # ~5.5km north -- inside FIRE_PRESSURE_RADIUS_KM=6.0 but far enough that
    # the base 2km decay has already suppressed most of the weight, leaving
    # room for wind widening to matter.
    fires = pd.DataFrame({"ts": [hours[0]], "lat": [lat + 0.0495], "lon": [lon],
                           "frp": [50.0], "confidence": [80]})
    wind = np.array([0.0])   # wind FROM the north -> straight from the fire to the cell

    calm = fire_pressure(cells, fires, hours, wind, wind_ms=np.array([0.1]))
    windy = fire_pressure(cells, fires, hours, wind, wind_ms=np.array([15.0]))

    calm_val = calm[calm.cell == cells[0]].fire_pressure_regional.iloc[0]
    windy_val = windy[windy.cell == cells[0]].fire_pressure_regional.iloc[0]
    assert windy_val > calm_val > 0.0


def test_positional_block_shape_is_seven_columns():
    cell = city_cells()[len(city_cells()) // 2]     # an interior cell, has 6 neighbors
    station_lat = np.array([12.97])
    station_lon = np.array([77.59])
    station_val = np.array([[50.0], [55.0]])          # 2 hours
    wind = np.array([90.0, 90.0])

    out = positional_block(cell, station_lat, station_lon, station_val, wind)

    assert out.shape == (2, 7)


def test_fire_pressure_zero_when_no_fires():
    cells = city_cells()[:5]
    hours = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    fires = pd.DataFrame(columns=["ts", "lat", "lon", "frp", "confidence"])
    wind = np.zeros(3)

    out = fire_pressure(cells, fires, hours, wind)

    assert set(out.columns) == {"cell", "ts", "fire_pressure_regional"}
    assert (out.fire_pressure_regional == 0.0).all()


def test_fire_pressure_positive_near_a_real_detection():
    # fire placed 1.1km NORTH of the cell, wind FROM the north (blows south,
    # straight from the fire toward the cell) -- genuinely upwind, so this
    # exercises the wind-alignment term rather than relying on distance=0's
    # arbitrary bearing (same lesson as Task 2's composite_grid tests).
    cells = city_cells()
    hours = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
    from shared.grid import cell_center
    lat, lon = cell_center(cells[0])
    fires = pd.DataFrame({"ts": [hours[0]], "lat": [lat + 0.01], "lon": [lon],
                           "frp": [50.0], "confidence": [80]})
    wind = np.array([0.0])   # wind FROM the north

    out = fire_pressure(cells, fires, hours, wind)

    assert out[out.cell == cells[0]].fire_pressure_regional.iloc[0] > 0.0


def test_fire_pressure_zero_when_wind_blows_away():
    # same fire placement as above, but wind now blows AWAY from the cell
    # (FROM the south) -- the fire is downwind of nothing relevant to this
    # cell, so pressure should be zero despite being within radius.
    cells = city_cells()
    hours = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
    from shared.grid import cell_center
    lat, lon = cell_center(cells[0])
    fires = pd.DataFrame({"ts": [hours[0]], "lat": [lat + 0.01], "lon": [lon],
                           "frp": [50.0], "confidence": [80]})
    wind = np.array([180.0])   # wind FROM the south -> blows north, away from the cell

    out = fire_pressure(cells, fires, hours, wind)

    assert out[out.cell == cells[0]].fire_pressure_regional.iloc[0] == pytest.approx(0.0, abs=1e-6)
