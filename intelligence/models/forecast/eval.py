"""Eval metrics for the forecaster (spec section 6). Extends what
forecast.py's old evaluate() already reported (skill vs. baseline, now as
building blocks a caller assembles into a fold distribution) with
quantile-interval coverage and a quiet-vs-event breakdown so an average
can't hide the model being worse exactly when it matters most."""
import numpy as np


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def skill_vs_baseline(y_true: np.ndarray, y_pred: np.ndarray, y_baseline: np.ndarray) -> float:
    """100 * (1 - rmse_model / rmse_baseline). Positive = model beats the
    baseline. Same formula forecast.py already used for persistence/diurnal."""
    r_model, r_base = _rmse(y_true, y_pred), _rmse(y_true, y_baseline)
    if r_base < 1e-9:
        return 0.0
    return round(100 * (1 - r_model / r_base), 1)


def quantile_coverage(y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
    """Fraction of true values inside [p10, p90]. Should land near 0.80 for
    a well-calibrated interval — spec section 6 requires this be checked,
    not assumed."""
    y_true, p10, p90 = np.asarray(y_true), np.asarray(p10), np.asarray(p90)
    return float(np.mean((y_true >= p10) & (y_true <= p90)))


def quiet_vs_event_breakdown(y_true: np.ndarray, y_pred: np.ndarray, is_event: np.ndarray) -> dict:
    """RMSE computed separately for real-event windows vs. normal periods."""
    y_true, y_pred, is_event = np.asarray(y_true), np.asarray(y_pred), np.asarray(is_event)
    quiet, event = ~is_event, is_event
    return {
        "quiet_rmse": _rmse(y_true[quiet], y_pred[quiet]) if quiet.any() else float("nan"),
        "event_rmse": _rmse(y_true[event], y_pred[event]) if event.any() else float("nan"),
        "n_quiet": int(quiet.sum()), "n_event": int(event.sum()),
    }


if __name__ == "__main__":
    y_true = np.array([50.0, 60.0, 200.0])
    y_pred = np.array([52.0, 58.0, 180.0])
    y_base = np.array([80.0, 20.0, 100.0])
    print("skill:", skill_vs_baseline(y_true, y_pred, y_base))
    print("coverage:", quantile_coverage(y_true, y_true - 10, y_true + 10))
