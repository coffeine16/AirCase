import numpy as np
import pandas as pd
import pytest

from intelligence.models.forecast.model import (
    UNKNOWN_CITY, mask_unknown_city, train_quantile_models,
    predict_quantiles, train_ceiling_baseline,
)

FEATURES = ["lag_0", "wind_ms", "city"]


def _toy_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    lag0 = rng.uniform(20, 100, n)
    wind = rng.uniform(0, 10, n)
    city = rng.choice(["bengaluru", "delhi", "chennai"], n)
    y = lag0 + rng.normal(0, 5, n)
    frame = pd.DataFrame({"lag_0": lag0, "wind_ms": wind, "city": city, "y": y})
    frame["city"] = frame["city"].astype("category")
    return frame


def test_mask_unknown_city_relabels_a_fraction():
    frame = _toy_frame()
    masked = mask_unknown_city(frame, frac=0.2, seed=1)
    assert (masked.city == UNKNOWN_CITY).mean() == pytest.approx(0.2, abs=0.05)
    assert (frame.city != UNKNOWN_CITY).all()   # original untouched


def test_train_and_predict_quantiles_are_ordered():
    frame = _toy_frame()
    frame = mask_unknown_city(frame, frac=0.05)
    frame["city"] = frame["city"].cat.set_categories(
        list(frame.city.cat.categories) + [UNKNOWN_CITY] if UNKNOWN_CITY not in frame.city.cat.categories else frame.city.cat.categories)
    train, valid = frame.iloc[:200], frame.iloc[200:]

    models = train_quantile_models(train, FEATURES, num_boost_round=50, valid=valid)
    pred = predict_quantiles(models, valid, FEATURES)

    assert list(pred.columns) == ["pm25_p10", "pm25_p50", "pm25_p90"]
    assert (pred.pm25_p10 <= pred.pm25_p50 + 1e-6).all()
    assert (pred.pm25_p50 <= pred.pm25_p90 + 1e-6).all()


def test_predict_handles_unseen_city_value():
    frame = _toy_frame()
    frame = mask_unknown_city(frame, frac=0.1)
    models = train_quantile_models(frame, FEATURES, num_boost_round=50)

    novel = _toy_frame(n=10, seed=99)
    novel["city"] = "mumbai"   # never seen in training at all
    novel["city"] = novel["city"].astype("category")

    pred = predict_quantiles(models, novel, FEATURES)   # must not raise
    assert len(pred) == 10
    assert pred.notna().all().all()


def test_ceiling_baseline_fits_without_error():
    frame = _toy_frame()
    reg = train_ceiling_baseline(frame, ["lag_0", "wind_ms"])
    pred = reg.predict(frame[["lag_0", "wind_ms"]])
    assert len(pred) == len(frame)


def test_ceiling_baseline_regularization_bounds_extrapolation():
    # Regression test for a real production bug: train_ceiling_baseline
    # used to force alpha=0.0 (no regularization), which let the fit
    # produce unstable coefficients that exploded when applied OOF to real
    # feature combinations outside the training sample's range -- on the
    # real 8-city run, ceiling_skill_vs_linear jumped from a genuine 6.9%
    # (v1) to a suspiciously exact 100.0 (v2), meaning the "ceiling"
    # baseline's own RMSE had exploded to ~2000x the model's, not that the
    # model actually got that much better. Near-collinear columns are the
    # classic condition that exposes this: an unregularized fit can't
    # distinguish how to split weight between them, so its coefficients
    # (and therefore its extrapolated predictions) become arbitrarily
    # unstable, while a regularized fit stays bounded.
    rng = np.random.default_rng(0)
    n = 60
    a = rng.uniform(0, 10, n)
    frame = pd.DataFrame({
        "a": a,
        "b": a + rng.normal(0, 1e-6, n),   # near-perfectly collinear with "a"
        "y": 2 * a + rng.normal(0, 1.0, n),
    })
    from sklearn.linear_model import QuantileRegressor
    unregularized = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")
    unregularized.fit(frame[["a", "b"]], frame["y"])
    regularized = train_ceiling_baseline(frame, ["a", "b"])

    # Far outside the training range (a, b in [0, 10]) -- the shape of a
    # real OOF row whose feature combination the small training sample
    # never saw, which is exactly where an unregularized fit's instability
    # shows up.
    extreme = pd.DataFrame({"a": [1000.0], "b": [1000.0]})
    pred_unreg = abs(unregularized.predict(extreme)[0])
    pred_reg = abs(regularized.predict(extreme)[0])
    assert pred_reg < pred_unreg
