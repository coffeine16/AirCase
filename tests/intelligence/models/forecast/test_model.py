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
