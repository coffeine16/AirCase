import numpy as np

from intelligence.models.forecast.eval import (
    skill_vs_baseline, quantile_coverage, quiet_vs_event_breakdown, event_by_outcome,
)


def test_skill_vs_baseline_positive_when_model_better():
    y_true = np.array([50.0, 60.0, 70.0])
    y_pred = np.array([51.0, 59.0, 71.0])       # small error
    y_base = np.array([80.0, 20.0, 40.0])        # large error
    assert skill_vs_baseline(y_true, y_pred, y_base) > 0


def test_skill_vs_baseline_negative_when_baseline_better():
    y_true = np.array([50.0, 60.0, 70.0])
    y_pred = np.array([10.0, 100.0, 10.0])
    y_base = np.array([51.0, 59.0, 71.0])
    assert skill_vs_baseline(y_true, y_pred, y_base) < 0


def test_quantile_coverage_all_inside():
    y_true = np.array([50.0, 60.0])
    p10 = np.array([40.0, 50.0])
    p90 = np.array([60.0, 70.0])
    assert quantile_coverage(y_true, p10, p90) == 1.0


def test_quantile_coverage_half_outside():
    y_true = np.array([50.0, 100.0])
    p10 = np.array([40.0, 40.0])
    p90 = np.array([60.0, 60.0])
    assert quantile_coverage(y_true, p10, p90) == 0.5


def test_quiet_vs_event_breakdown_reports_both():
    y_true = np.array([50.0, 60.0, 200.0, 210.0])
    y_pred = np.array([51.0, 59.0, 100.0, 220.0])
    is_event = np.array([False, False, True, True])

    out = quiet_vs_event_breakdown(y_true, y_pred, is_event)

    assert set(out) == {"quiet_rmse", "event_rmse", "n_quiet", "n_event"}
    assert out["n_quiet"] == 2
    assert out["n_event"] == 2
    assert out["event_rmse"] > out["quiet_rmse"]   # the model is worse during the event here


def test_event_by_outcome_is_relative_to_each_citys_own_level():
    """A "clean" city (baseline ~20) and a "dirty" city (baseline ~150) --
    an absolute PM2.5 threshold would flag almost none of the clean city's
    rows and almost all of the dirty city's, which is exactly the kind of
    city-mix confound this function exists to avoid (the real fires_6h
    definition did this by proxy: 64% of flagged rows came from just two
    cities). event_by_outcome must flag roughly the same FRACTION of each
    city's own rows, not the same absolute level."""
    rng = np.random.default_rng(0)
    clean = rng.normal(20, 3, 1000)
    dirty = rng.normal(150, 15, 1000)
    y = np.concatenate([clean, dirty])
    city = np.array(["clean"] * 1000 + ["dirty"] * 1000)

    is_event = event_by_outcome(y, city, percentile=80.0)

    clean_event_frac = is_event[city == "clean"].mean()
    dirty_event_frac = is_event[city == "dirty"].mean()
    # both cities contribute ~20% of their OWN rows as events
    assert 0.15 < clean_event_frac < 0.25
    assert 0.15 < dirty_event_frac < 0.25
    # and the clean city's event rows are still far below the dirty city's
    # ABSOLUTE level -- proving the threshold really is per-city, not global
    assert y[is_event & (city == "clean")].max() < y[~is_event & (city == "dirty")].min()


def test_quiet_vs_event_per_city_reveals_a_city_mix_artifact():
    """The exact confound this project measured for real: a pooled
    event_rmse < quiet_rmse that is ENTIRELY explained by which city
    contributes more event rows, not by the model behaving differently
    during events. City A is all "quiet" and has a large model error; city
    B is all "event" and has a small model error (a different city, simply
    easier to forecast, quiet or not). Pooled, this reads as "model is
    better during events" -- exactly backwards, and exactly what per_city
    has to catch by showing the SAME (large) error inside city A and the
    SAME (small) error inside city B regardless of quiet/event."""
    y_true = np.array([100.0, 100.0, 100.0, 100.0,   # city A, all quiet, big error
                       50.0, 50.0, 50.0, 50.0])       # city B, all event, small error
    y_pred = np.array([70.0, 70.0, 70.0, 70.0,        # error 30 throughout A
                       48.0, 48.0, 48.0, 48.0])        # error 2 throughout B
    is_event = np.array([False, False, False, False, True, True, True, True])
    city = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    out = quiet_vs_event_breakdown(y_true, y_pred, is_event, city=city)

    # pooled numbers alone tell exactly the wrong story
    assert out["event_rmse"] < out["quiet_rmse"]
    # but per-city, there is NO quiet-vs-event gap at all in either city --
    # city A is uniformly bad, city B is uniformly good, regardless of the
    # event flag. This is what proves the pooled gap was a city-mix
    # artifact rather than a real event-time capability.
    assert out["per_city"]["A"]["n_event"] == 0
    assert out["per_city"]["B"]["n_quiet"] == 0
    assert out["per_city"]["A"]["quiet_rmse"] == 30.0
    assert out["per_city"]["B"]["event_rmse"] == 2.0
