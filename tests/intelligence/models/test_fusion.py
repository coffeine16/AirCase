import numpy as np
import pandas as pd
import pytest

from intelligence.models.fusion import _with_cyclical, _train, _predict, _city_baseline, FEATURES


def test_with_cyclical_matches_known_trig_values():
    df = pd.DataFrame({"hour": [0, 6, 12, 18], "dow": [0, 3, 6, 6],
                        "wind_from_deg": [0.0, 90.0, 180.0, 270.0]})
    out = _with_cyclical(df)

    assert out["hour_sin"].tolist() == pytest.approx([0.0, 1.0, 0.0, -1.0], abs=1e-9)
    assert out["hour_cos"].tolist() == pytest.approx([1.0, 0.0, -1.0, 0.0], abs=1e-9)
    assert out["wind_from_deg_sin"].tolist() == pytest.approx([0.0, 1.0, 0.0, -1.0], abs=1e-9)
    assert out["wind_from_deg_cos"].tolist() == pytest.approx([1.0, 0.0, -1.0, 0.0], abs=1e-9)


def test_with_cyclical_wraps_hour_23_next_to_hour_0():
    """The whole point: hour 23 and hour 0 must land close together in
    (sin, cos) space, even though they are 23 apart as raw integers -- a
    raw-encoded model has to learn this adjacency from data; this makes it
    true by construction. Contrast against a NON-adjacent hour (12, the
    other side of the clock) to prove the closeness is really about the
    wrap, not just "any two hours are close"."""
    df = pd.DataFrame({"hour": [23, 0, 12], "dow": [0, 0, 0], "wind_from_deg": [0.0, 0.0, 0.0]})
    out = _with_cyclical(df)

    def dist(i, j):
        return np.hypot(out["hour_sin"][i] - out["hour_sin"][j],
                         out["hour_cos"][i] - out["hour_cos"][j])

    assert dist(0, 1) < 0.3          # 23:00 and 00:00: wraps, must be close
    assert dist(0, 2) > 1.5          # 23:00 and 12:00: genuinely far apart
    assert dist(0, 1) < dist(0, 2)


def _panel(n_stations=4, n_hours=60, seed=0):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    rows = []
    for i in range(n_stations):
        for h in hours:
            rows.append({
                "cell": f"cell_{i}", "ts": h,
                "pm25_station": 50.0 + i * 5 + rng.normal(0, 3),
                "no2_col": rng.normal(20, 5), "wind_from_deg": float(rng.uniform(0, 360)),
                "wind_ms": 2.0, "blh_m": 400.0, "temp_c": 27.0,
                "fires_6h": 0, "frp_6h": 0.0,
                "lu_industrial": 0, "lu_construction": 0, "lu_waste_burning": 0,
                "lu_traffic": 0, "lu_road": 1,
                "hour": h.hour, "dow": h.dayofweek,
            })
    return pd.DataFrame(rows)


def test_train_and_predict_run_end_to_end_with_cyclical_features():
    """Smoke test for the FEATURES/_with_cyclical refactor: _train indexes
    df[FEATURES] internally, so a mismatch between what _with_cyclical
    produces and what FEATURES expects would raise KeyError here."""
    df = _panel()
    baseline = _city_baseline(df)
    model = _train(df, baseline, rounds=20)
    pred = _predict(model, df, baseline)

    assert len(pred) == len(df)
    assert np.isfinite(pred).all()
    assert set(model.feature_name()) == set(FEATURES)
