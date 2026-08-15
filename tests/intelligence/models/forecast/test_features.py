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
                "pm25_station": 50.0 + i * 10 if i == 0 else np.nan,   # only cells[0] is a station
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
    # otherwise the LOSO test would trivially see its own real data
    assert normal_row["lag_0"] != loso_row["lag_0"] or np.isnan(loso_row["lag_0"])
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
    loso_row = loso[loso.cell == station_cell].iloc[-1]

    # cell[0]'s own cell-level climatology is exactly 50.0 (its own constant
    # reading); if that leaked through unmasked, clim_dow_hour would be
    # exactly 50.0. It must not be -- the ward blend (50 and 90) differs.
    assert loso_row["clim_dow_hour"] != pytest.approx(50.0)
