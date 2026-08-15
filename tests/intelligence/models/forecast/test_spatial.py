import numpy as np
import pandas as pd

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
    from intelligence.agents.attribution import category_scores
    assert DIST_DECAY_KM == 2.0   # same exp(-d/2) kernel used in category_scores


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

    out = fire_pressure(cells, fires, hours)

    assert set(out.columns) == {"cell", "ts", "fire_pressure_regional"}
    assert (out.fire_pressure_regional == 0.0).all()


def test_fire_pressure_positive_near_a_real_detection():
    cells = city_cells()
    hours = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
    from shared.grid import cell_center
    lat, lon = cell_center(cells[0])
    fires = pd.DataFrame({"ts": [hours[0]], "lat": [lat], "lon": [lon],
                           "frp": [50.0], "confidence": [80]})

    out = fire_pressure(cells, fires, hours)

    assert out[out.cell == cells[0]].fire_pressure_regional.iloc[0] > 0.0
