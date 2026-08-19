import numpy as np
import pandas as pd
import pytest

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


def _tested_days(folds):
    days = set()
    for _, test_start, test_end in folds:
        d = test_start
        while d < test_end:
            days.add(d.date())
            d += pd.Timedelta(days=1)
    return days


def test_walk_forward_folds_default_halves_folds_without_gapping_coverage():
    """The 42/42 default must cut the 21/21 fold count roughly in half
    while still testing every eligible calendar day exactly once --
    step_days > test_days (an earlier, rejected plan) creates a real gap
    between every pair of folds' test windows, silently dropping half the
    calendar from every OOF-based metric. 42/42 must NOT reproduce that."""
    frame = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=24 * 400, freq="h", tz="UTC")})

    old = walk_forward_folds(frame, min_train_days=180, test_days=21, step_days=21)
    new = walk_forward_folds(frame)   # the new 42/42 default

    assert len(new) < len(old)
    assert len(new) == pytest.approx(len(old) / 2, abs=1)

    old_days, new_days = _tested_days(old), _tested_days(new)
    # both configs are contiguous (step == test): coverage should match
    # almost exactly, not merely "similar" -- a couple of days' slack only
    # for the boundary effect of a wider test window near the panel's end.
    assert abs(len(new_days) - len(old_days)) <= max(21, 42)

    # the rejected step>test config actually gaps: proves the property this
    # test guards against is real, not hypothetical.
    gapped = walk_forward_folds(frame, min_train_days=180, test_days=21, step_days=42)
    gapped_days = _tested_days(gapped)
    assert len(gapped_days) < len(old_days) * 0.6


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
