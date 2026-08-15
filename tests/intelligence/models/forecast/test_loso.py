import numpy as np
import pandas as pd

from shared.grid import city_cells
from intelligence.models.forecast.validation import spatial_loso, run_city_loso


def _panel_with_two_stations():
    cells = city_cells()[:4]
    hours = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    rows = []
    for i, c in enumerate(cells):
        is_station = i < 2
        for h in hours:
            rows.append({
                "cell": c, "ts": h, "ward_id": "W1", "ward_name": "Ward 1", "city": "bengaluru",
                "pm25_station": float(50 + i * 5 + rng.normal(0, 3)) if is_station else np.nan,
                "wind_from_deg": 90.0, "wind_ms": 2.0, "blh_m": 400.0, "temp_c": 27.0,
                "fires_6h": 0, "frp_6h": 0.0, "lu_industrial": 0, "lu_construction": 0,
                "lu_waste_burning": 0, "lu_traffic": 0, "lu_road": 1, "lu_sensitive": 0,
                "hour": h.hour, "dow": h.dayofweek,
            })
    return pd.DataFrame(rows)


def test_spatial_loso_runs_one_fold_per_station():
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    result = spatial_loso(_panel_with_two_stations(), horizons=[3], feature_cols=FEATURE_COLUMNS)

    assert result["n_stations"] == 2
    assert set(result["per_station"]) == set(city_cells()[:2])
    assert "overall_rmse" in result


def test_run_city_loso_covers_every_city():
    panels = {c: _panel_with_two_stations().assign(city=c) for c in ("bengaluru", "delhi")}
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    result = run_city_loso(panels, horizons=[3], feature_cols=FEATURE_COLUMNS)

    assert set(result["per_city"]) == {"bengaluru", "delhi"}
