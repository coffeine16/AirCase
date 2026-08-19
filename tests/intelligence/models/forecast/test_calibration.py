import numpy as np

from intelligence.models.forecast.eval import (
    apply_interval_scale, interval_scale_for_coverage, quantile_coverage,
)


def _narrow_bands(n=20_000, seed=0):
    """Bands deliberately too tight: residuals are N(0, 1) but the interval is
    only +/-1 sigma, which covers ~68% -- close to the 0.684 the first real
    8-city run actually produced."""
    rng = np.random.default_rng(seed)
    p50 = rng.uniform(20, 200, n)
    y = p50 + rng.normal(0, 1, n) * 20
    return y, p50 - 20, p50, p50 + 20


def test_scale_lifts_undercovered_bands_to_target():
    y, p10, p50, p90 = _narrow_bands()
    before = quantile_coverage(y, p10, p90)
    assert before < 0.75, f"fixture must start UNDER-covered, got {before}"

    scale = interval_scale_for_coverage(y, p10, p50, p90, target=0.80)
    assert scale > 1.0, "an under-covered interval must be widened, not narrowed"

    lo, hi = apply_interval_scale(p10, p50, p90, scale)
    after = quantile_coverage(y, lo, hi)
    assert abs(after - 0.80) < 0.01, f"expected ~0.80 coverage after scaling, got {after}"


def test_already_calibrated_bands_are_left_alone():
    rng = np.random.default_rng(1)
    p50 = rng.uniform(20, 200, 20_000)
    resid = rng.normal(0, 20, 20_000)
    y = p50 + resid
    # +/-1.2816 sigma is exactly the 80% interval for a normal
    p10, p90 = p50 - 1.2816 * 20, p50 + 1.2816 * 20
    scale = interval_scale_for_coverage(y, p10, p50, p90, target=0.80)
    assert 0.95 < scale < 1.05, f"already-calibrated bands should stay put, got {scale}"


def test_scale_never_moves_the_median():
    y, p10, p50, p90 = _narrow_bands()
    scale = interval_scale_for_coverage(y, p10, p50, p90)
    lo, hi = apply_interval_scale(p10, p50, p90, scale)
    # the point forecast is untouched: calibration is about uncertainty only
    np.testing.assert_allclose((lo + hi) / 2, p50, rtol=1e-9)


def test_none_scale_is_a_passthrough():
    y, p10, p50, p90 = _narrow_bands()
    lo, hi = apply_interval_scale(p10, p50, p90, None)
    np.testing.assert_array_equal(lo, p10)
    np.testing.assert_array_equal(hi, p90)
