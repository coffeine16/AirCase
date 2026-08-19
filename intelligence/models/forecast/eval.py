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


def interval_scale_for_coverage(y_true: np.ndarray, p10: np.ndarray, p50: np.ndarray,
                                 p90: np.ndarray, target: float = 0.80,
                                 lo: float = 0.25, hi: float = 6.0) -> float:
    """Multiplier on each side's half-width that makes OUT-OF-SAMPLE coverage
    hit `target`. 1.0 means the raw quantiles were already calibrated.

    Measured on the first real 8-city run, raw coverage was 0.684 against a
    0.80 target -- the bands were too narrow, so roughly one value in three
    fell outside an interval that should miss one in five. Quantile boosting
    optimises pinball loss per quantile independently and has no mechanism
    that forces the resulting INTERVAL to cover; this rescales it after the
    fact against held-out residuals, which is the only place coverage can
    honestly be measured.

    Scaled per side, not as one symmetric band: PM2.5 residuals have a heavy
    upper tail, and collapsing both sides into one number would widen the
    floor to buy headroom it doesn't need.
    """
    y, p10, p50, p90 = (np.asarray(a, dtype=float) for a in (y_true, p10, p50, p90))
    ok = np.isfinite(y) & np.isfinite(p10) & np.isfinite(p50) & np.isfinite(p90)
    if not ok.any():
        return 1.0
    y, p10, p50, p90 = y[ok], p10[ok], p50[ok], p90[ok]
    down, up = p50 - p10, p90 - p50

    def coverage(s: float) -> float:
        return float(np.mean((y >= p50 - s * down) & (y <= p50 + s * up)))

    if coverage(hi) < target:      # even the widest band can't reach it
        return hi
    if coverage(lo) > target:
        return lo
    for _ in range(50):            # coverage is monotone in s, so bisect
        mid = (lo + hi) / 2
        if coverage(mid) < target:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


def apply_interval_scale(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray,
                          scale: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Widen (or narrow) a predicted interval by `scale` about the median.
    Returns the adjusted (p10, p90); p50 is never moved -- calibration is a
    statement about uncertainty, not about the central estimate."""
    p10, p50, p90 = (np.asarray(a, dtype=float) for a in (p10, p50, p90))
    if scale is None or not np.isfinite(scale) or scale == 1.0:
        return p10, p90
    return p50 - scale * (p50 - p10), p50 + scale * (p90 - p50)


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
