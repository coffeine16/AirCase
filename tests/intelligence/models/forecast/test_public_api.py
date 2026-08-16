def test_public_api_importable():
    from intelligence.models.forecast import run, evaluate, HORIZONS, TEST_TAIL_DAYS
    assert callable(run)
    assert callable(evaluate)
    assert isinstance(HORIZONS, list)
    assert isinstance(TEST_TAIL_DAYS, int)


def test_run_pipeline_imports_forecast_run():
    # this is the exact import scripts/run_pipeline.py uses — a broken
    # package export would fail here before ever touching real data
    import importlib
    mod = importlib.import_module("intelligence.models.forecast")
    assert hasattr(mod, "run")


def test_predict_field_uses_served_manifest_cities_not_current_panels():
    # Simulates a refused promotion falling back to a PRIOR model trained on
    # a DIFFERENT, wider city set than the current run's panels -- exactly
    # the scenario this session's own city-registry expansion made concrete.
    # city_categories must come from served_manifest["cities"], never
    # panels.keys(), or LightGBM's integer-coded categorical predictions
    # silently corrupt (not crash -- corrupt). A "does it raise" test would
    # NOT catch this; each city gets a strong, distinct trained offset so a
    # wrong category code produces a MEASURABLY wrong prediction.
    import numpy as np
    import pandas as pd
    from shared.grid import city_cells
    from intelligence.models.forecast import _predict_field
    from intelligence.models.forecast.model import train_quantile_models

    rng = np.random.default_rng(0)
    offsets = {"bengaluru": 0.0, "chennai": 50.0, "delhi": 100.0, "unknown": 25.0}
    rows = [{"lag_0": rng.uniform(20, 30), "wind_ms": rng.uniform(0, 10),
             "city": city, "y": off + rng.normal(0, 2)}
            for city, off in offsets.items() for _ in range(200)]
    train_frame = pd.DataFrame(rows)
    train_frame["city"] = train_frame["city"].astype("category")
    served_models = train_quantile_models(train_frame, ["lag_0", "wind_ms", "city"], num_boost_round=100)

    # the served model's REAL training cities (what the manifest records)
    served_manifest = {"cities": ["bengaluru", "chennai", "delhi"]}

    # the CURRENT run's panel is ONLY "delhi" -- alphabetically first in a
    # 1-city category list (code 0), but delhi is code 2 in the real
    # 3-city+unknown list. A wrong category source silently remaps delhi's
    # rows onto whatever trained as code 0 (bengaluru's ~0 offset) instead
    # of delhi's real ~100 offset.
    cell = city_cells()[0]
    hours = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    rows2 = [{"cell": cell, "ts": h, "ward_id": "W1", "ward_name": "Ward 1", "city": "delhi",
              "pm25_station": 100.0, "wind_from_deg": 90.0, "wind_ms": 5.0, "blh_m": 400.0,
              "temp_c": 27.0, "fires_6h": 0, "frp_6h": 0.0, "lu_industrial": 0, "lu_construction": 0,
              "lu_waste_burning": 0, "lu_traffic": 0, "lu_road": 1, "lu_sensitive": 0,
              "hour": h.hour, "dow": h.dayofweek} for h in hours]
    panels = {"delhi": pd.DataFrame(rows2)}

    fields = _predict_field(panels, served_manifest, served_models, ["lag_0", "wind_ms", "city"])

    assert "delhi" in fields and fields["delhi"], "no predictions produced — check the panel fixture"
    preds = [row["pm25_hat"] for row in fields["delhi"]]
    # correct categories put "delhi" at ITS real trained code (2) -> predictions
    # land near delhi's ~100 offset. A wrong (panels-derived, 1-city) category
    # list would put "delhi" at code 0, which trained as bengaluru's ~0 offset.
    assert all(p > 60 for p in preds), (
        f"predictions {preds} suggest the wrong city category source was used "
        f"(expected near delhi's ~100 offset, not bengaluru's ~0)")
