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
