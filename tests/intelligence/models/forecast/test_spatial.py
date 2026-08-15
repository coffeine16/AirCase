import numpy as np

from intelligence.models.forecast.spatial import composite_grid, DIST_DECAY_KM


def test_composite_grid_basic_blend():
    # two stations, one query point exactly between them (same distance,
    # opposite bearings) -> composite should land near their average when
    # wind gives them equal alignment (wind blowing directly along the line)
    query_lat = np.array([12.97])
    query_lon = np.array([77.60])
    station_lat = np.array([12.97, 12.97])
    station_lon = np.array([77.59, 77.61])          # west and east of the query point
    station_val = np.array([[40.0, 60.0]])           # one hour, two stations
    wind_from_deg = np.array([0.0])                  # wind from due north: no preferential alignment either way

    out = composite_grid(query_lat, query_lon, station_lat, station_lon,
                          station_val, wind_from_deg)

    assert out.shape == (1, 1)
    assert 45.0 < out[0, 0] < 55.0                    # roughly the mean of 40 and 60


def test_composite_grid_excludes_masked_station():
    query_lat = np.array([12.97])
    query_lon = np.array([12.97])
    station_lat = np.array([12.97])
    station_lon = np.array([12.97])                  # station AT the query point: max possible weight
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
