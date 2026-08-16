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


def test_predict_field_pins_city_categories_to_the_served_manifest():
    """WHAT THIS DOES AND DOES NOT VERIFY.

    Verified: the frame handed to the model carries the category set the
    SERVED model was trained on (`served_manifest["cities"]`), not the current
    run's `panels.keys()` — which differ whenever a refused promotion falls
    back to a prior model trained on a wider city set.

    NOT verified — and an earlier version of this test wrongly claimed it was:
    that the wrong source corrupts predictions. Measured directly: LightGBM
    realigns a pandas Categorical column by category NAME at predict time
    (`cat.set_categories()` is applied for you), so with the Categorical input
    _predict_field actually passes, both category sources yield identical
    numbers. The pin is defensive — it holds if that convention changes, or if
    a future caller passes a raw string city column, which gets no realignment.
    """
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

    served_manifest = {"cities": ["bengaluru", "chennai", "delhi"]}

    cell = city_cells()[0]
    hours = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    rows2 = [{"cell": cell, "ts": h, "ward_id": "W1", "ward_name": "Ward 1", "city": "delhi",
              "pm25_station": 100.0, "wind_from_deg": 90.0, "wind_ms": 5.0, "blh_m": 400.0,
              "temp_c": 27.0, "fires_6h": 0, "frp_6h": 0.0, "lu_industrial": 0, "lu_construction": 0,
              "lu_waste_burning": 0, "lu_traffic": 0, "lu_road": 1, "lu_sensitive": 0,
              "hour": h.hour, "dow": h.dayofweek} for h in hours]
    panels = {"delhi": pd.DataFrame(rows2)}   # ONE city, vs the manifest's three

    import intelligence.models.forecast as F
    captured = []
    real = F.predict_quantiles
    monkeypatched = lambda models, frame, cols: (captured.append(frame), real(models, frame, cols))[1]
    F.predict_quantiles = monkeypatched
    try:
        fields = _predict_field(panels, served_manifest, served_models, ["lag_0", "wind_ms", "city"])
    finally:
        F.predict_quantiles = real

    assert "delhi" in fields and fields["delhi"], "no predictions produced — check the panel fixture"
    assert captured, "predict_quantiles was never called"
    # the discriminating assertion: panels.keys() would give ['delhi','unknown']
    assert list(captured[0]["city"].cat.categories) == ["bengaluru", "chennai", "delhi", "unknown"]


def test_evaluate_never_trains_on_nan_labels():
    # build_features emits a row per (cell, hour, horizon) and non-station
    # cells have no label at all, yet survive the lag filter because the
    # composite fills their lags. LightGBM silently trains on NaN labels.
    import numpy as np
    import pandas as pd
    from shared.grid import city_cells
    import intelligence.models.forecast as F

    cells = city_cells()[:4]
    hours = pd.date_range("2024-01-01", periods=24 * 20, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    rows = [{"cell": c, "ts": h, "ward_id": "W1", "ward_name": "Ward 1", "city": "bengaluru",
             "pm25_station": float(50 + i * 5 + rng.normal(0, 3)) if i < 2 else np.nan,
             "wind_from_deg": 90.0, "wind_ms": 2.0, "blh_m": 400.0, "temp_c": 27.0,
             "fires_6h": 0, "frp_6h": 0.0, "lu_industrial": 0, "lu_construction": 0,
             "lu_waste_burning": 0, "lu_traffic": 0, "lu_road": 1, "lu_sensitive": 0,
             "hour": h.hour, "dow": h.dayofweek}
            for i, c in enumerate(cells) for h in hours]

    seen = []
    real = F.train_quantile_models
    F.train_quantile_models = lambda train, *a, **k: (seen.append(train), real(train, *a, **k))[1]
    try:
        F.evaluate(pd.DataFrame(rows))
    finally:
        F.train_quantile_models = real

    assert seen, "evaluate trained nothing"
    for train_frame in seen:
        assert train_frame["y"].notna().all()
    # ...and only ONE model for all 24 horizons, not one per horizon
    assert len(seen) == 1
