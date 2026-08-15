import numpy as np
import pandas as pd

from intelligence.models.forecast.validation import (
    walk_forward_folds, event_weights, city_loso_splits,
)


def test_walk_forward_folds_expanding_window():
    frame = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=24 * 300, freq="h", tz="UTC")})
    folds = walk_forward_folds(frame, min_train_days=180, test_days=21, step_days=21)

    assert len(folds) >= 2
    for train_end, test_start, test_end in folds:
        assert train_end == test_start
        assert (test_end - test_start).days == 21
    # expanding, not sliding: every fold's train_end is later than the last
    assert folds[1][0] > folds[0][0]


def test_walk_forward_folds_empty_when_too_little_history():
    frame = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="UTC")})
    assert walk_forward_folds(frame, min_train_days=180, test_days=21) == []


def test_event_weights_boosts_fire_rows_only():
    frame = pd.DataFrame({"fires_6h": [0, 0, 3, 0, 5]})
    w = event_weights(frame, boost=4.0)
    assert list(w) == [1.0, 1.0, 4.0, 1.0, 4.0]


def test_city_loso_splits_covers_every_city_once():
    splits = city_loso_splits(["bengaluru", "delhi", "chennai"])
    held_out = [h for h, _ in splits]
    assert sorted(held_out) == ["bengaluru", "chennai", "delhi"]
    for held, train in splits:
        assert held not in train
        assert len(train) == 2
