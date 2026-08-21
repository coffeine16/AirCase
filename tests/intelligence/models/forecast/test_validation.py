import numpy as np
import pandas as pd
import pytest

from shared.grid import city_cells
from intelligence.models.forecast.features import build_features, FEATURE_COLUMNS
from intelligence.models.forecast.validation import (
    walk_forward_folds, event_weights, city_loso_splits, _run_one_walk_forward_fold,
    run_walk_forward, _align_city,
)
from intelligence.models.forecast.model import UNKNOWN_CITY


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


def _panel_for_walk_forward():
    # 5 station cells + 1 blank, 300 hours -- enough real history for a
    # train slice and a disjoint test slice with real, non-NaN y labels.
    cells = city_cells()[:6]
    hours = pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    rows = []
    for i, c in enumerate(cells):
        level = None if i == 5 else 40.0 + i * 3
        for h in hours:
            rows.append({
                "cell": c, "ts": h, "ward_id": "W1", "ward_name": "Ward 1", "city": "bengaluru",
                "pm25_station": np.nan if level is None else float(level + rng.normal(0, 2)),
                "wind_from_deg": 90.0, "wind_ms": 2.0, "blh_m": 400.0, "temp_c": 27.0,
                "fires_6h": 0, "frp_6h": 0.0, "lu_industrial": 0, "lu_construction": 0,
                "lu_waste_burning": 0, "lu_traffic": 0, "lu_road": 1, "lu_sensitive": 0,
                "hour": h.hour, "dow": h.dayofweek,
            })
    return pd.DataFrame(rows), hours


def test_walk_forward_fold_never_masks_test_frame_city():
    # Regression test for the Finding-2 fix: mask_unknown_city used to run
    # on the WHOLE pooled frame before any fold existed, so a walk-forward
    # fold's TEST rows could carry an artificially withheld "unknown" city
    # label -- a real serving call always knows its own city. After the
    # fix, only the TRAIN slice is masked, per fold, after the split.
    panel, hours = _panel_for_walk_forward()
    frame = build_features(panel, horizons=[3], restrict_to_station_cells=True)
    num_cols = [c for c in FEATURE_COLUMNS if c != "city"]
    train_end, test_start, test_end = hours[199], hours[199], hours[279]

    result = _run_one_walk_forward_fold(panel, frame, FEATURE_COLUMNS, num_cols,
                                         train_end, test_start, test_end, num_threads=None)
    assert result is not None
    # oof_frame's "city" column (validation.py's oof_frame) is built from
    # `te` (str-cast) -- "unknown" appearing there would mean a test row
    # got relabelled, exactly the bug this test guards against.
    assert "unknown" not in result["oof"]["city"].unique()


def test_run_walk_forward_resumes_from_checkpoint_without_recomputing(tmp_path):
    panel, hours = _panel_for_walk_forward()
    frame = build_features(panel, horizons=[3], restrict_to_station_cells=True)
    num_cols = [c for c in FEATURE_COLUMNS if c != "city"]
    folds = [
        (hours[149], hours[149], hours[199]),
        (hours[199], hours[199], hours[279]),
    ]
    ckpt = str(tmp_path / "ckpt")

    first = run_walk_forward(panel, frame, FEATURE_COLUMNS, num_cols, folds, checkpoint_dir=ckpt)
    ckpt_files = list((tmp_path / "ckpt" / "walk_forward").glob("*.pkl"))
    assert len(ckpt_files) == 2   # one per fold

    # Rebuild the frame from a 1000x-scaled panel -- a real recomputation
    # on this frame would produce very different skill/OOF numbers. If the
    # second call still matches `first` exactly, the checkpoint was used.
    scaled_panel = panel.assign(pm25_station=panel.pm25_station * 1000.0)
    scaled_frame = build_features(scaled_panel, horizons=[3], restrict_to_station_cells=True)
    second = run_walk_forward(scaled_panel, scaled_frame, FEATURE_COLUMNS, num_cols, folds,
                               checkpoint_dir=ckpt)
    assert second["fold_skills"] == first["fold_skills"]
    pd.testing.assert_frame_equal(second["oof"], first["oof"])


def test_align_city_relabels_unseen_city_to_unknown_not_nan():
    # Regression test for a real bug caught by an 8-city sanity run (real
    # data, staggered per-city start dates), not a unit test: a city in
    # test that train never saw used to silently become NaN via
    # pd.Categorical's default unmatched-value handling, which crashed
    # eval.py's event_by_outcome trying to np.unique/sort a column mixing
    # NaN (float) with real city strings.
    test_frame = pd.DataFrame({"city": ["bengaluru", "pune", "delhi"]})
    train_categories = pd.CategoricalDtype(["bengaluru", "delhi", UNKNOWN_CITY]).categories

    out = _align_city(test_frame, train_categories)
    assert list(out["city"]) == ["bengaluru", UNKNOWN_CITY, "delhi"]
    assert not out["city"].isna().any()


def test_align_city_adds_unknown_category_when_train_never_masked_a_row():
    # A tiny train slice can plausibly have mask_unknown_city's random 5%
    # draw select zero rows, leaving "unknown" absent from its own category
    # set entirely. Must still be usable as a relabel target.
    test_frame = pd.DataFrame({"city": ["bengaluru", "pune"]})
    train_categories = pd.CategoricalDtype(["bengaluru", "delhi"]).categories

    out = _align_city(test_frame, train_categories)
    assert list(out["city"]) == ["bengaluru", UNKNOWN_CITY]
    assert not out["city"].isna().any()


def test_align_city_relabel_unknown_true_blanket_relabels_every_row():
    # city_loso's own usage: the whole test frame is one held-out city.
    test_frame = pd.DataFrame({"city": ["delhi", "delhi", "delhi"]})
    train_categories = pd.CategoricalDtype(["bengaluru", "chennai", UNKNOWN_CITY]).categories

    out = _align_city(test_frame, train_categories, relabel_unknown=True)
    assert list(out["city"]) == [UNKNOWN_CITY] * 3
