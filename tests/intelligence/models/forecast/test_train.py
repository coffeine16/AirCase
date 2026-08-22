import json

import numpy as np
import pandas as pd
import pytest

from shared.grid import city_cells
from intelligence.models.forecast.train import train_and_promote, _mature_oof, _resolve_workers
import intelligence.models.forecast.train as train_module


def test_mature_oof_keeps_only_the_later_better_trained_folds():
    """Early walk-forward folds train on a fraction of the final model's
    data (measured on the real run: fold 1's 318K rows vs fold 38's
    12.69M) -- pooling them equally with late folds when calibrating the
    serve-time interval correction would calibrate against the average
    fold's immaturity, not what the actual final model does. _mature_oof
    must actually drop the small-n_train fold's rows, not just pass
    everything through unchanged."""
    oof = pd.DataFrame({
        "y": [1.0] * 3 + [2.0] * 5,
        "p10": [0.0] * 8, "p50": [0.0] * 8, "p90": [0.0] * 8,
        "n_train": [1_000] * 3 + [50_000] * 5,   # small early fold, big late fold
    })
    mature = _mature_oof(oof)

    assert set(mature["n_train"]) == {50_000}
    assert len(mature) == 5
    assert (mature["y"] == 2.0).all()


def test_mature_oof_keeps_everything_with_a_single_fold():
    oof = pd.DataFrame({"y": [1.0, 2.0], "p10": [0.0, 0.0], "p50": [0.0, 0.0],
                        "p90": [0.0, 0.0], "n_train": [10_000, 10_000]})
    assert len(_mature_oof(oof)) == len(oof)


def _panel(city, n_hours=400, seed=0):
    # Offset by seed so two different "cities" in one test get DISTINCT real
    # H3 cells -- every multi-city test call here already passes seed=1 for
    # the second city. Real cities never share a cell (H3 IDs are geographic),
    # but city_cells()[:3] for both fixture cities would fake that collision,
    # which now matters: build_features' met_lookup merges on (cell, ts), and
    # two "cities" sharing a cell ID would fan the merge out across both.
    cells = city_cells()[seed * 3: seed * 3 + 3]
    hours = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    rows = []
    for i, c in enumerate(cells):
        is_station = i < 2
        for h in hours:
            rows.append({
                "cell": c, "ts": h, "ward_id": "W1", "ward_name": "Ward 1", "city": city,
                "pm25_station": float(50 + i * 5 + rng.normal(0, 3)) if is_station else np.nan,
                "wind_from_deg": 90.0, "wind_ms": 2.0, "blh_m": 400.0, "temp_c": 27.0,
                "fires_6h": 0, "frp_6h": 0.0, "lu_industrial": 0, "lu_construction": 0,
                "lu_waste_burning": 0, "lu_traffic": 0, "lu_road": 1, "lu_sensitive": 0,
                "hour": h.hour, "dow": h.dayofweek,
            })
    return pd.DataFrame(rows)


# 400 hours is ~17 days, far short of walk_forward_folds' 180-day default, so
# a default-configured run produces ZERO folds -- and the three out-of-sample
# eval metrics are then honestly None. Shrink the fold geometry instead of
# faking the metrics, so the assertions below exercise the real OOF path.
_SHORT_FOLDS = {"min_train_days": 8, "test_days": 3, "step_days": 3}


def test_train_and_promote_writes_versioned_artifacts(tmp_path):
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    panels = {"bengaluru": _panel("bengaluru"), "delhi": _panel("delhi", seed=1)}

    manifest = train_and_promote(panels, horizons=[3, 6], feature_cols=FEATURE_COLUMNS,
                                  out_dir=tmp_path, walk_forward_kwargs=_SHORT_FOLDS)

    assert (tmp_path / manifest["version"] / "model_p10.txt").exists()
    assert (tmp_path / manifest["version"] / "model_p50.txt").exists()
    assert (tmp_path / manifest["version"] / "model_p90.txt").exists()
    assert (tmp_path / "manifest.json").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["version"] == manifest["version"]
    assert "spatial_loso_rmse" in manifest["eval"]
    assert "city_loso" in manifest["eval"]
    assert "quiet_vs_event" in manifest["eval"]
    qve = manifest["eval"]["quiet_vs_event"]
    assert set(qve) == {"quiet_rmse", "event_rmse", "n_quiet", "n_event", "per_city"}
    # per_city stratification exists specifically so a pooled event_rmse <
    # quiet_rmse can be told apart from a city-mix artifact -- it has to
    # actually be there, not just an empty dict alongside the pooled keys.
    assert qve["per_city"], "quiet_vs_event lost its per-city stratification"
    for city_stats in qve["per_city"].values():
        assert set(city_stats) == {"quiet_rmse", "event_rmse", "n_quiet", "n_event"}
    # the three formerly-in-sample metrics must come from the walk-forward
    # folds' held-out predictions, not from the frame the final model was fit on
    assert manifest["eval"]["eval_basis"] == "walk_forward_out_of_sample"
    assert manifest["eval"]["walk_forward_skill_folds"] >= 1
    assert manifest["eval"]["quantile_coverage"] is not None


def test_train_and_promote_parallel_matches_sequential(tmp_path):
    """train_and_promote's max_workers now parallelizes all three CV stages
    (walk-forward, spatial-LOSO, city-LOSO) via ProcessPoolExecutor. Prove
    the parallel path produces the same eval numbers as sequential on the
    same panels -- not just "should" because the per-fold logic was
    factored into shared helpers, actually run both and compare. A small
    tolerance (not exact equality) allows for LightGBM histogram reduction
    order differing across thread counts, the same real non-determinism
    spatial_loso's own equivalence test already accounts for."""
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    panels = {"bengaluru": _panel("bengaluru"), "delhi": _panel("delhi", seed=1)}

    sequential = train_and_promote(panels, horizons=[3, 6], feature_cols=FEATURE_COLUMNS,
                                    out_dir=tmp_path / "seq", walk_forward_kwargs=_SHORT_FOLDS,
                                    max_workers=1)
    parallel = train_and_promote(panels, horizons=[3, 6], feature_cols=FEATURE_COLUMNS,
                                  out_dir=tmp_path / "par", walk_forward_kwargs=_SHORT_FOLDS,
                                  max_workers=2, threads_per_fold=1)

    assert sequential["eval"]["walk_forward_skill_folds"] == parallel["eval"]["walk_forward_skill_folds"] > 0
    assert sequential["eval"]["spatial_loso_n_stations"] == parallel["eval"]["spatial_loso_n_stations"] > 0
    assert set(sequential["eval"]["city_loso"]) == set(parallel["eval"]["city_loso"]) != set()

    assert sequential["eval"]["spatial_loso_rmse"] == pytest.approx(
        parallel["eval"]["spatial_loso_rmse"], rel=0.1)
    assert sequential["eval"]["walk_forward_skill_median"] == pytest.approx(
        parallel["eval"]["walk_forward_skill_median"], abs=15)
    for city in sequential["eval"]["city_loso"]:
        assert sequential["eval"]["city_loso"][city]["rmse"] == pytest.approx(
            parallel["eval"]["city_loso"][city]["rmse"], rel=0.1)


def test_out_of_sample_metrics_are_none_without_folds(tmp_path):
    # no folds -> the honest answer is "not measured", never an in-sample stand-in
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    panels = {"bengaluru": _panel("bengaluru")}

    manifest = train_and_promote(panels, horizons=[3], feature_cols=FEATURE_COLUMNS,
                                  out_dir=tmp_path)

    assert manifest["eval"]["eval_basis"] == "no_walk_forward_folds"
    for k in ("quantile_coverage", "ceiling_skill_vs_linear", "quiet_vs_event"):
        assert manifest["eval"][k] is None


def test_regression_tolerance_band_works_for_negative_priors():
    # walk-forward skill is legitimately negative at short horizons in this
    # project (persistence wins). prior*(1-tol) flips the band's direction for
    # a negative prior and demands an IMPROVEMENT to pass; prior-abs(prior)*tol
    # does not.
    from intelligence.models.forecast.train import _regressed

    assert _regressed(-10.5, -10.0, higher_is_better=True, tolerance_pct=5.0) is False
    assert _regressed(-11.0, -10.0, higher_is_better=True, tolerance_pct=5.0) is True
    # positive priors keep their existing behaviour
    assert _regressed(9.6, 10.0, higher_is_better=True, tolerance_pct=5.0) is False
    assert _regressed(9.4, 10.0, higher_is_better=True, tolerance_pct=5.0) is True


def test_resolve_workers_grows_threads_into_memory_bound_idle_cpu(monkeypatch):
    # Regression test for a real finding: worker count used to be memory-
    # bound (by_mem, independent of thread count) while threads_per_fold
    # stayed fixed at whatever the caller passed -- on a real run, memory
    # capped workers at 8 on a 32-core box, but threads_per_fold=2 was
    # never grown, leaving 16 cores idle for the whole run. by_mem doesn't
    # depend on thread count, so once workers are memory-bound, growing
    # threads_per_fold costs no extra memory and just uses the idle cores.
    monkeypatch.setattr(train_module, "_available_cpus", lambda: 32)
    monkeypatch.setattr(train_module, "_available_memory_bytes", lambda: 275_000_000_000)

    # payload/multiplier chosen so by_mem lands at 8 (matching the real run):
    # spendable = (275e9 - payload) * 0.5; by_mem = spendable // (payload * 2.5) == 8
    payload_bytes = 6_240_000_000   # ~6.24GB, the real run's own measured payload
    workers, threads = _resolve_workers(threads_per_fold=2, payload_bytes=payload_bytes)
    assert workers == 8
    # 32 cpus // 8 workers = 4 -- grown well past the caller's floor of 2,
    # using every core the memory-bound worker count leaves idle.
    assert threads == 4
    assert workers * threads <= 32


def test_resolve_workers_never_shrinks_below_the_callers_floor(monkeypatch):
    # A tiny container (few cores) must never come back with FEWER threads
    # than the caller asked for, even if the arithmetic would suggest it.
    monkeypatch.setattr(train_module, "_available_cpus", lambda: 4)
    monkeypatch.setattr(train_module, "_available_memory_bytes", lambda: 275_000_000_000)
    workers, threads = _resolve_workers(threads_per_fold=2, payload_bytes=6_240_000_000)
    assert threads >= 2


def test_train_and_promote_refuses_regression(tmp_path):
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    panels = {"bengaluru": _panel("bengaluru"), "delhi": _panel("delhi", seed=1)}

    first = train_and_promote(panels, horizons=[3], feature_cols=FEATURE_COLUMNS, out_dir=tmp_path)

    # a deliberately bad prior manifest (impossibly good recorded skill) must block promotion
    fake_prior = {**first, "eval": {**first["eval"], "spatial_loso_rmse": 0.0}}
    second = train_and_promote(panels, horizons=[3], feature_cols=FEATURE_COLUMNS,
                                out_dir=tmp_path, prior_manifest=fake_prior)

    assert second["promoted"] is False
    assert second["version"] != first["version"]   # trained, just not promoted


def test_walk_forward_regression_still_blocks_promotion_when_geometry_matches(tmp_path):
    """The gate must keep its teeth: a genuine walk_forward_skill_median
    regression against a prior measured under the SAME fold geometry has to
    still refuse promotion, same as before the geometry check existed."""
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    panels = {"bengaluru": _panel("bengaluru"), "delhi": _panel("delhi", seed=1)}

    first = train_and_promote(panels, horizons=[3], feature_cols=FEATURE_COLUMNS,
                               out_dir=tmp_path, walk_forward_kwargs=_SHORT_FOLDS)
    assert first["eval"]["walk_forward_geometry"] == {"test_days": 3, "step_days": 3}

    fake_prior = {**first, "eval": {**first["eval"], "walk_forward_skill_median": 999.0}}
    second = train_and_promote(panels, horizons=[3], feature_cols=FEATURE_COLUMNS,
                                out_dir=tmp_path, walk_forward_kwargs=_SHORT_FOLDS,
                                prior_manifest=fake_prior)

    assert second["promoted"] is False


def test_walk_forward_regression_ignored_when_geometry_differs(tmp_path):
    """The actual fix: a prior manifest scored under a DIFFERENT fold
    geometry (e.g. trained before walk_forward_folds' defaults changed)
    must not let a walk_forward_skill_median 'regression' block promotion
    on its own -- that comparison is apples-to-oranges, not evidence the
    new model is worse. spatial_loso_rmse and city_loso are untouched by
    this and still gate normally."""
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    panels = {"bengaluru": _panel("bengaluru"), "delhi": _panel("delhi", seed=1)}

    first = train_and_promote(panels, horizons=[3], feature_cols=FEATURE_COLUMNS,
                               out_dir=tmp_path, walk_forward_kwargs=_SHORT_FOLDS)

    # same impossibly-good fake skill as the geometry-matches test, but this
    # prior's geometry is DIFFERENT (test_days=6 vs the real run's 3) --
    # simulating exactly the transition this run is meant to survive.
    fake_prior = {**first, "eval": {**first["eval"], "walk_forward_skill_median": 999.0,
                                     "walk_forward_geometry": {"test_days": 6, "step_days": 6}}}
    second = train_and_promote(panels, horizons=[3], feature_cols=FEATURE_COLUMNS,
                                out_dir=tmp_path, walk_forward_kwargs=_SHORT_FOLDS,
                                prior_manifest=fake_prior)

    # would have been refused under the old (pre-fix) comparison -- proves
    # the geometry mismatch is what's protecting promotion here, not that
    # the check silently stopped mattering.
    assert second["promoted"] is True


def test_walk_forward_regression_ignored_against_a_manifest_with_no_geometry_recorded(tmp_path):
    """A prior manifest written before walk_forward_geometry existed has no
    such key at all -- must be treated the same as a mismatch (unknown,
    don't compare), not as a false match that would re-enable the strict
    comparison via a None == None coincidence."""
    from intelligence.models.forecast.features import FEATURE_COLUMNS
    panels = {"bengaluru": _panel("bengaluru"), "delhi": _panel("delhi", seed=1)}

    first = train_and_promote(panels, horizons=[3], feature_cols=FEATURE_COLUMNS,
                               out_dir=tmp_path, walk_forward_kwargs=_SHORT_FOLDS)

    old_style_eval = {k: v for k, v in first["eval"].items() if k != "walk_forward_geometry"}
    old_style_eval["walk_forward_skill_median"] = 999.0
    fake_prior = {**first, "eval": old_style_eval}
    second = train_and_promote(panels, horizons=[3], feature_cols=FEATURE_COLUMNS,
                                out_dir=tmp_path, walk_forward_kwargs=_SHORT_FOLDS,
                                prior_manifest=fake_prior)

    assert second["promoted"] is True


def test_train_and_promote_default_engine_is_pandas():
    import inspect
    from intelligence.models.forecast.train import train_and_promote
    sig = inspect.signature(train_and_promote)
    assert sig.parameters["engine"].default == "pandas"


def test_train_and_promote_docstring_mentions_spatial_loso_native_support():
    from intelligence.models.forecast.train import train_and_promote
    assert "spatial-LOSO" in train_and_promote.__doc__
    assert "walk-forward" in train_and_promote.__doc__.lower() or "walk_forward" in train_and_promote.__doc__
