import numpy as np
import pandas as pd
import pytest

from intelligence.models.forecast import HORIZONS
from intelligence.models.forecast.features import FEATURE_COLUMNS, station_cells_only
from intelligence.models.forecast.native.streaming import run_final_fit_native

REAL_CITIES = ["chennai", "hyderabad", "ahmedabad"]


def _load_panels(cities):
    panels = {}
    for c in cities:
        p = pd.read_parquet(f"data/historical/{c}/panel.parquet")
        p["city"] = c
        panels[c] = station_cells_only(p)
    return panels


def test_final_fit_native_produces_a_model_per_quantile():
    panels = _load_panels(REAL_CITIES)
    models = run_final_fit_native(panels, HORIZONS, FEATURE_COLUMNS)
    assert set(models.keys()) == {0.1, 0.5, 0.9}
    for q, model in models.items():
        assert model.num_trees() > 0


def test_final_fit_native_predictions_are_plausible_pm25_range():
    """Not a bit-exact parity test (there's no pandas final-fit run to
    directly diff against without also running the expensive real pandas
    path here) -- a sanity check that the trained model isn't degenerate,
    matching the check already done manually on the real v2 model this
    session (quantile ordering, plausible value range)."""
    panels = _load_panels(REAL_CITIES)
    models = run_final_fit_native(panels, HORIZONS, FEATURE_COLUMNS)

    from intelligence.models.forecast.features import build_features
    from intelligence.models.forecast.climatology import build_climatology
    from intelligence.models.forecast.model import mask_unknown_city

    full_panel = pd.concat(list(panels.values()), ignore_index=True)
    clim = build_climatology(full_panel)
    frame = mask_unknown_city(
        build_features(full_panel, HORIZONS, restrict_to_station_cells=True,
                        clim_tables=clim).dropna(subset=["y"]))
    sample = frame.sample(n=min(2000, len(frame)), random_state=0)
    X = sample[FEATURE_COLUMNS].copy()
    X["city"] = X["city"].astype("category")

    p10 = models[0.1].predict(X)
    p50 = models[0.5].predict(X)
    p90 = models[0.9].predict(X)

    assert np.mean(p10 <= p50) > 0.95
    assert np.mean(p50 <= p90) > 0.95
    assert -50 < np.median(p50) < 500  # plausible PM2.5 scale, generous bounds
