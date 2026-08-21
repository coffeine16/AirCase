import numpy as np
import pandas as pd

from intelligence.models.forecast.spatial import composite_grid
from intelligence.models.forecast.native.native_composite import composite_grid_native
from shared.grid import cell_center


def _load_real_station_setup(city: str, n_hours: int = 500):
    """Real per-city historical data, not synthetic -- this project has
    been burned before by synthetic fixtures that hid real bugs (see
    CLAUDE.md's "the synthetic world's instruments were too kind")."""
    panel = pd.read_parquet(f"data/historical/{city}/panel.parquet")
    station_cells = sorted(panel.loc[panel.pm25_station.notna(), "cell"].unique())
    assert len(station_cells) >= 3, f"{city} needs >= 3 real stations for this test"
    station_lat = np.array([cell_center(c)[0] for c in station_cells])
    station_lon = np.array([cell_center(c)[1] for c in station_cells])

    wide = (panel[panel.cell.isin(station_cells)]
            .pivot(index="ts", columns="cell", values="pm25_station")
            .reindex(columns=station_cells).iloc[:n_hours])
    station_val = wide.to_numpy(dtype=np.float64)

    ref_cell_rows = panel[panel.cell == station_cells[0]].set_index("ts").iloc[:n_hours]
    wind_from_deg = ref_cell_rows["wind_from_deg"].to_numpy(dtype=np.float64)
    wind_ms = ref_cell_rows["wind_ms"].to_numpy(dtype=np.float64)

    query_cells = sorted(panel.cell.unique())[:10]
    query_lat = np.array([cell_center(c)[0] for c in query_cells])
    query_lon = np.array([cell_center(c)[1] for c in query_cells])
    return query_lat, query_lon, station_lat, station_lon, station_val, wind_from_deg, wind_ms


def test_composite_grid_native_matches_pandas_no_wind_ms():
    query_lat, query_lon, station_lat, station_lon, station_val, wind_from_deg, _ = \
        _load_real_station_setup("chennai")
    expected = composite_grid(query_lat, query_lon, station_lat, station_lon,
                               station_val, wind_from_deg, wind_ms=None)
    actual = composite_grid_native(query_lat, query_lon, station_lat, station_lon,
                                    station_val, wind_from_deg, wind_ms=None)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_composite_grid_native_matches_pandas_with_wind_ms():
    query_lat, query_lon, station_lat, station_lon, station_val, wind_from_deg, wind_ms = \
        _load_real_station_setup("chennai")
    expected = composite_grid(query_lat, query_lon, station_lat, station_lon,
                               station_val, wind_from_deg, wind_ms=wind_ms)
    actual = composite_grid_native(query_lat, query_lon, station_lat, station_lon,
                                    station_val, wind_from_deg, wind_ms=wind_ms)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_composite_grid_native_matches_pandas_with_exclude():
    query_lat, query_lon, station_lat, station_lon, station_val, wind_from_deg, wind_ms = \
        _load_real_station_setup("chennai")
    n_q, n_s = len(query_lat), len(station_lat)
    exclude = np.zeros((n_q, n_s), dtype=bool)
    exclude[0, 0] = True
    expected = composite_grid(query_lat, query_lon, station_lat, station_lon,
                               station_val, wind_from_deg, wind_ms=wind_ms, exclude=exclude)
    actual = composite_grid_native(query_lat, query_lon, station_lat, station_lon,
                                    station_val, wind_from_deg, wind_ms=wind_ms, exclude=exclude)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_composite_grid_native_matches_pandas_on_a_second_city():
    """Chennai alone could hide a bug that only shows up with a different
    station count/geometry -- one more real city, not a second synthetic
    fixture."""
    query_lat, query_lon, station_lat, station_lon, station_val, wind_from_deg, wind_ms = \
        _load_real_station_setup("hyderabad")
    expected = composite_grid(query_lat, query_lon, station_lat, station_lon,
                               station_val, wind_from_deg, wind_ms=wind_ms)
    actual = composite_grid_native(query_lat, query_lon, station_lat, station_lon,
                                    station_val, wind_from_deg, wind_ms=wind_ms)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_composite_grid_native_matches_pandas_with_genuine_2d_wind():
    """Test real per-query-cell wind broadcasting (n_t, n_q) arrays, not
    just 1-D broadcast to every column. Each query cell gets its own wind
    reading from the panel."""
    panel = pd.read_parquet("data/historical/chennai/panel.parquet")
    station_cells = sorted(panel.loc[panel.pm25_station.notna(), "cell"].unique())
    station_lat = np.array([cell_center(c)[0] for c in station_cells])
    station_lon = np.array([cell_center(c)[1] for c in station_cells])

    wide = (panel[panel.cell.isin(station_cells)]
            .pivot(index="ts", columns="cell", values="pm25_station")
            .reindex(columns=station_cells).iloc[:500])
    station_val = wide.to_numpy(dtype=np.float64)

    # Use a wider set of query cells to genuinely exercise per-cell wind variation.
    query_cells = sorted(panel.cell.unique())[:15]
    query_lat = np.array([cell_center(c)[0] for c in query_cells])
    query_lon = np.array([cell_center(c)[1] for c in query_cells])
    n_q = len(query_cells)
    n_t = station_val.shape[0]

    # Build genuine (n_t, n_q) wind arrays by extracting each query cell's own wind data.
    wind_from_deg_2d = np.empty((n_t, n_q), dtype=np.float64)
    wind_ms_2d = np.empty((n_t, n_q), dtype=np.float64)
    for qi, qc in enumerate(query_cells):
        qc_rows = panel[panel.cell == qc].set_index("ts").iloc[:n_t]
        wind_from_deg_2d[:, qi] = qc_rows["wind_from_deg"].to_numpy(dtype=np.float64)
        wind_ms_2d[:, qi] = qc_rows["wind_ms"].to_numpy(dtype=np.float64)

    # Test with per-query-cell wind (not broadcast from a single cell).
    expected = composite_grid(query_lat, query_lon, station_lat, station_lon,
                               station_val, wind_from_deg_2d, wind_ms=wind_ms_2d)
    actual = composite_grid_native(query_lat, query_lon, station_lat, station_lon,
                                    station_val, wind_from_deg_2d, wind_ms=wind_ms_2d)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9, equal_nan=True)
