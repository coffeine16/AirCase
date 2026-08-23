import numpy as np
import pandas as pd
import pytest

from shared.grid import city_cells
from intelligence.models.forecast.validation import spatial_loso, run_city_loso


def _nan_safe_equal(a, b):
    """Plain == treats NaN as never equal to itself, so a dict containing a
    legitimately-NaN value (e.g. a station whose baseline had no valid
    comparison rows) never compares equal via == even to an identical
    cached copy of itself. Used by the checkpoint-resume tests below, where
    "the second call returned exactly the cached result" is genuinely true
    even when one of the cached fields is NaN."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_nan_safe_equal(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def _panel_with_two_stations():
    cells = city_cells()[:4]
    hours = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    rows = []
    for i, c in enumerate(cells):
        is_station = i < 2
        for h in hours:
            rows.append({
                "cell": c, "ts": h, "ward_id": "W1", "ward_name": "Ward 1", "city": "bengaluru",
                "pm25_station": float(50 + i * 5 + rng.normal(0, 3)) if is_station else np.nan,
                "wind_from_deg": 90.0, "wind_ms": 2.0, "blh_m": 400.0, "temp_c": 27.0,
                "fires_6h": 0, "frp_6h": 0.0, "lu_industrial": 0, "lu_construction": 0,
                "lu_waste_burning": 0, "lu_traffic": 0, "lu_road": 1, "lu_sensitive": 0,
                "hour": h.hour, "dow": h.dayofweek,
            })
    return pd.DataFrame(rows)


def _panel_with_five_stations(held_out_level=200.0):
    """5 station cells + 1 blank. The held-out station reads at a level no
    other station comes near, so "did the composite/climatology just hand the
    model the answer" is directly checkable. A 2-3 station fixture cannot
    expose that — which is exactly why the bug survived Task 9's review."""
    cells = city_cells()[:6]
    hours = pd.date_range("2024-01-01", periods=240, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    rows = []
    for i, c in enumerate(cells):
        if i == 0:
            level = held_out_level
        elif i < 5:
            level = 40.0 + i * 3
        else:
            level = None
        for h in hours:
            rows.append({
                "cell": c, "ts": h, "ward_id": "W1", "ward_name": "Ward 1", "city": "bengaluru",
                "pm25_station": np.nan if level is None else float(level + rng.normal(0, 2)),
                "wind_from_deg": 90.0, "wind_ms": 2.0, "blh_m": 400.0, "temp_c": 27.0,
                "fires_6h": 0, "frp_6h": 0.0, "lu_industrial": 0, "lu_construction": 0,
                "lu_waste_burning": 0, "lu_traffic": 0, "lu_road": 1, "lu_sensitive": 0,
                "hour": h.hour, "dow": h.dayofweek,
            })
    return pd.DataFrame(rows)


def test_spatial_loso_runs_one_fold_per_station():
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    result = spatial_loso(_panel_with_two_stations(), horizons=[3], feature_cols=FEATURE_COLUMNS)

    assert result["n_stations"] == 2
    assert set(result["per_station"]) == set(city_cells()[:2])
    assert "overall_rmse" in result
    # not just "a key exists" -- the number has to be real
    assert np.isfinite(result["overall_rmse"])


def test_spatial_loso_resumes_from_checkpoint_without_recomputing(tmp_path):
    # Regression test for the checkpointing feature: a second call with the
    # SAME checkpoint_dir must return the cached result, not recompute.
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    panel = _panel_with_two_stations()
    ckpt = str(tmp_path / "ckpt")

    first = spatial_loso(panel, horizons=[3], feature_cols=FEATURE_COLUMNS, checkpoint_dir=ckpt)
    ckpt_files = list((tmp_path / "ckpt" / "spatial_loso").glob("*.pkl"))
    assert len(ckpt_files) == 2   # one per station, both real (non-empty) folds

    # Scale every station's readings 1000x -- same cells still report (so
    # the checkpoint-loading loop still runs, unlike an all-NaN panel which
    # would short-circuit before ever reaching it), but a REAL recomputation
    # on this panel would produce wildly different RMSE/per_station numbers.
    # If the second call still matches `first` exactly, it proves the
    # checkpoint was actually used, not just coincidentally reproduced.
    scaled = panel.copy()
    scaled["pm25_station"] = scaled["pm25_station"] * 1000.0
    second = spatial_loso(scaled, horizons=[3], feature_cols=FEATURE_COLUMNS, checkpoint_dir=ckpt)
    assert _nan_safe_equal(second, first)


def test_spatial_loso_parallel_resume_also_skips_completed_folds(tmp_path):
    # Same guarantee on the ProcessPoolExecutor path -- checkpointed
    # stations must be filtered out BEFORE submission, not after (see
    # spatial_loso's own comment on why), so this needs its own test: the
    # sequential test above never exercises that branch.
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    panel = _panel_with_five_stations()
    ckpt = str(tmp_path / "ckpt")

    first = spatial_loso(panel, horizons=[3], feature_cols=FEATURE_COLUMNS,
                          max_workers=2, checkpoint_dir=ckpt)
    scaled = panel.copy()
    scaled["pm25_station"] = scaled["pm25_station"] * 1000.0
    second = spatial_loso(scaled, horizons=[3], feature_cols=FEATURE_COLUMNS,
                           max_workers=2, checkpoint_dir=ckpt)
    assert _nan_safe_equal(second, first)


def test_spatial_loso_baseline_rmse_is_real_persistence_not_a_copy_of_overall():
    """spatial_loso_rmse had no comparator -- 40.51 on a real run couldn't be
    judged good or bad standalone. baseline_rmse is the persistence (lag_0)
    RMSE on the SAME held-out rows the model's own overall_rmse is scored
    on. Prove it's actually computed from lag_0, not accidentally aliased
    to the model's own prediction: on the five-station fixture the held-out
    station reads at a level (~200) far from lag_0's likely composite-filled
    value (~40-52, the other stations' level) at the very start of the
    panel, before the held-out station's own lag has had time to settle --
    so a persistence baseline and the model's real prediction should not
    coincide."""
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    result = spatial_loso(_panel_with_five_stations(held_out_level=200.0),
                           horizons=[3], feature_cols=FEATURE_COLUMNS)

    assert np.isfinite(result["baseline_rmse"])
    assert result["baseline_rmse"] > 0
    finite_per_station = [s["baseline_rmse"] for s in result["per_station"].values()
                          if np.isfinite(s["baseline_rmse"])]
    # every fold's key is present, but not every fold's VALUE has to be
    # finite -- composite_grid's wind-alignment weights can legitimately
    # sum to zero for a specific station's exact geometry (same edge case
    # noted elsewhere in this codebase), same as it always could for
    # lag_0 itself. The pooled baseline_rmse above already tolerates that
    # via nanmean, same as overall_rmse always has.
    assert all("baseline_rmse" in s for s in result["per_station"].values())
    assert len(finite_per_station) >= len(result["per_station"]) - 1
    # the two numbers must be free to differ -- if baseline_rmse were
    # secretly just overall_rmse under a new name, they would be bit-for-bit
    # identical on every fold, which a real persistence-vs-model comparison
    # has no reason to produce
    assert result["baseline_rmse"] != result["overall_rmse"]


def test_spatial_loso_parallel_path_matches_sequential():
    """The parallel (ProcessPoolExecutor) path exists purely as a faster way
    to compute the SAME thing the sequential loop does -- both call
    _run_one_loso_fold with identical arguments, just from different
    processes. This proves that equivalence directly rather than trusting
    it by construction: run both paths on the same panel and require the
    same stations, same per-station RMSEs, and the same overall RMSE. A
    small tolerance (not exact equality) allows for LightGBM histogram
    reduction order differing slightly across thread counts -- real,
    documented floating-point non-determinism, not a bug in the split
    itself."""
    from intelligence.models.forecast.features import FEATURE_COLUMNS

    panel = _panel_with_five_stations(held_out_level=200.0)
    sequential = spatial_loso(panel, horizons=[3], feature_cols=FEATURE_COLUMNS)
    parallel = spatial_loso(panel, horizons=[3], feature_cols=FEATURE_COLUMNS,
                             max_workers=2, threads_per_fold=1)

    assert sequential["n_stations"] == parallel["n_stations"] > 0
    assert set(sequential["per_station"]) == set(parallel["per_station"])
    for cell in sequential["per_station"]:
        seq_rmse = sequential["per_station"][cell]["rmse"]
        par_rmse = parallel["per_station"][cell]["rmse"]
        assert seq_rmse == pytest.approx(par_rmse, rel=0.05), (
            f"{cell}: sequential={seq_rmse} vs parallel={par_rmse} -- the two "
            "paths should compute the same fold, just on different processes")
    assert sequential["overall_rmse"] == pytest.approx(parallel["overall_rmse"], rel=0.05)


def test_spatial_loso_test_frame_sees_the_other_stations_not_its_own_answer():
    """Regression test for the composite/climatology leak.

    The old spatial_loso sliced the panel down to the held-out cell BEFORE
    calling build_features. The composite then had no other station to compose
    from (every spatial feature NaN) and the climatology, rebuilt from that one
    station, resolved to the held-out station's own value -- which is also the
    target. The "RMSE" measured nothing.
    """
    from intelligence.models.forecast.features import build_features

    panel = _panel_with_five_stations(held_out_level=200.0)
    held_out = city_cells()[0]

    frame = build_features(panel, horizons=[3], loso_exclude=held_out)
    test_frame = frame[frame.cell == held_out]
    assert not test_frame.empty

    # the OTHER stations' composite must actually reach the held-out cell
    for col in ("lag_0", "lag_24", "pos_0", "nearby_stations_delta"):
        assert test_frame[col].notna().any(), f"{col} is entirely NaN — nothing to compose from"

    # ...and it must not be the held-out station's own ~200 level (its own
    # readings are the only thing near 200 anywhere in this panel)
    assert test_frame["lag_0"].max() < 120, "held-out station leaked into its own composite"
    # climatology likewise: all three scopes must exclude it, not just `cell`
    assert test_frame["clim_dow_hour"].notna().any()
    assert test_frame["clim_dow_hour"].max() < 120, "held-out station leaked via ward/city climatology"
    # ...while the target it is scored against IS the ~200 level
    assert test_frame["y"].median() > 150


def test_loso_training_frames_carry_no_nan_labels_and_real_fire_pressure(monkeypatch):
    """build_features emits a row per (cell, hour, horizon); non-station cells
    have no label but survive the lag filter because the composite fills their
    lags. LightGBM neither raises nor warns on a NaN label — it just trains a
    useless model. Also asserts the `fires` argument actually reaches
    build_features, since fire_pressure_regional was 0.0 at every real call
    site."""
    import intelligence.models.forecast.validation as V
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    from shared.grid import cell_center

    panel = _panel_with_five_stations()
    lat, lon = cell_center(city_cells()[0])
    # due EAST of the cell, so the fire->cell bearing (270) matches where the
    # wind_from_deg=90 wind actually blows -> alignment 1.0, weight > 0
    fires = pd.DataFrame({"ts": panel.ts.unique()[:48],
                          "lat": lat, "lon": lon + 0.01, "frp": 100.0, "confidence": 90})

    seen = []
    real = V.train_quantile_models
    monkeypatch.setattr(V, "train_quantile_models",
                         lambda train, *a, **k: (seen.append(train), real(train, *a, **k))[1])

    V.spatial_loso(panel, horizons=[3], feature_cols=FEATURE_COLUMNS, fires=fires)

    assert seen, "spatial_loso trained nothing"
    for train_frame in seen:
        assert train_frame["y"].notna().all(), "training on NaN labels"
        assert train_frame["fire_pressure_regional"].max() > 0, "fires never reached build_features"


def test_spatial_loso_itself_builds_features_on_the_full_panel(monkeypatch):
    """Pins spatial_loso's own call shape, not just build_features in isolation.

    The earlier regression test above (`..._sees_the_other_stations_not_its_own_answer`)
    calls build_features directly, so it only proves build_features is capable of
    the right thing -- it says nothing about whether spatial_loso actually calls it
    that way. spatial_loso could be reverted to pre-slicing the panel to one cell
    before calling build_features (the original C1 bug) and every other loso test
    still passes, because none of them inspect the frame spatial_loso itself hands
    to predict_quantiles.
    """
    import intelligence.models.forecast.validation as V
    from intelligence.models.forecast.features import FEATURE_COLUMNS

    panel = _panel_with_five_stations(held_out_level=200.0)
    held_out = city_cells()[0]

    seen_test_frames = []
    real_predict = V.predict_quantiles
    monkeypatch.setattr(
        V, "predict_quantiles",
        lambda models, test, *a, **k: (seen_test_frames.append(test), real_predict(models, test, *a, **k))[1])

    V.spatial_loso(panel, horizons=[3], feature_cols=FEATURE_COLUMNS)

    frames = [f for f in seen_test_frames if held_out in set(f.get("cell", []))]
    assert frames, "held-out station's fold never reached predict_quantiles"
    test_frame = frames[0]

    # if spatial_loso pre-slices the panel to the held-out cell before calling
    # build_features (the original bug), lag_0 comes back entirely NaN -- there
    # is no other station left to compose from.
    assert test_frame["lag_0"].notna().any(), (
        "spatial_loso's own test frame has an all-NaN lag_0 -- it is pre-slicing "
        "the panel before build_features again")
    assert test_frame["lag_0"].max() < 120, (
        "held-out station's own ~200 level leaked into its test frame's composite")


def test_run_city_loso_covers_every_city():
    panels = {c: _panel_with_two_stations().assign(city=c) for c in ("bengaluru", "delhi")}
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    result = run_city_loso(panels, horizons=[3], feature_cols=FEATURE_COLUMNS)

    assert set(result["per_city"]) == {"bengaluru", "delhi"}


def test_run_city_loso_resumes_from_checkpoint_without_recomputing(tmp_path):
    panels = {c: _panel_with_two_stations().assign(city=c) for c in ("bengaluru", "delhi")}
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    ckpt = str(tmp_path / "ckpt")

    first = run_city_loso(panels, horizons=[3], feature_cols=FEATURE_COLUMNS, checkpoint_dir=ckpt)
    ckpt_files = list((tmp_path / "ckpt" / "city_loso").glob("*.pkl"))
    assert len(ckpt_files) == 2   # one per city

    scaled = {c: p.assign(pm25_station=p.pm25_station * 1000.0) for c, p in panels.items()}
    second = run_city_loso(scaled, horizons=[3], feature_cols=FEATURE_COLUMNS, checkpoint_dir=ckpt)
    assert _nan_safe_equal(second, first)
