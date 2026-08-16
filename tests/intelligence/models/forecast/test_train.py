import json

import numpy as np
import pandas as pd

from shared.grid import city_cells
from intelligence.models.forecast.train import train_and_promote


def _panel(city, n_hours=400, seed=0):
    cells = city_cells()[:3]
    hours = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    rows = []
    for i, c in enumerate(cells):
        is_station = i < 2
        for h in hours:
            rows.append({
                "cell": c, "ts": h, "ward_id": "W1", "ward_name": "Ward 1", "city": city,
                "pm25_station": float(50 + i * 5 + rng.normal(0, 3)) if is_station else np.nan,
                "wind_from_deg": 90.0, "wind_ms": 2.0, "blh_m": 400.0, "temp_c": 27.0,
                "fires_6h": 0, "frp_6h": 0.0, "lu_industrial": 0, "lu_construction": 0,
                "lu_waste_burning": 0, "lu_traffic": 0, "lu_road": 1, "lu_sensitive": 0,
                "hour": h.hour, "dow": h.dayofweek,
            })
    return pd.DataFrame(rows)


def test_train_and_promote_writes_versioned_artifacts(tmp_path):
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    panels = {"bengaluru": _panel("bengaluru"), "delhi": _panel("delhi", seed=1)}

    manifest = train_and_promote(panels, horizons=[3, 6], feature_cols=FEATURE_COLUMNS,
                                  out_dir=tmp_path)

    assert (tmp_path / manifest["version"] / "model_p10.txt").exists()
    assert (tmp_path / manifest["version"] / "model_p50.txt").exists()
    assert (tmp_path / manifest["version"] / "model_p90.txt").exists()
    assert (tmp_path / "manifest.json").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["version"] == manifest["version"]
    assert "spatial_loso_rmse" in manifest["eval"]
    assert "city_loso" in manifest["eval"]
    assert "quiet_vs_event" in manifest["eval"]
    assert set(manifest["eval"]["quiet_vs_event"]) == {"quiet_rmse", "event_rmse", "n_quiet", "n_event"}


def test_train_and_promote_refuses_regression(tmp_path):
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    panels = {"bengaluru": _panel("bengaluru"), "delhi": _panel("delhi", seed=1)}

    first = train_and_promote(panels, horizons=[3], feature_cols=FEATURE_COLUMNS, out_dir=tmp_path)

    # a deliberately bad prior manifest (impossibly good recorded skill) must block promotion
    fake_prior = {**first, "eval": {**first["eval"], "spatial_loso_rmse": 0.0}}
    second = train_and_promote(panels, horizons=[3], feature_cols=FEATURE_COLUMNS,
                                out_dir=tmp_path, prior_manifest=fake_prior)

    assert second["promoted"] is False
    assert second["version"] != first["version"]   # trained, just not promoted
