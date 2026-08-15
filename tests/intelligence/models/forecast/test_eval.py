import numpy as np

from intelligence.models.forecast.eval import (
    skill_vs_baseline, quantile_coverage, quiet_vs_event_breakdown,
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
