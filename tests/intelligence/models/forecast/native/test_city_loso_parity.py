import pandas as pd
import pytest

from intelligence.models.forecast import HORIZONS
from intelligence.models.forecast.features import FEATURE_COLUMNS, station_cells_only
from intelligence.models.forecast.validation import run_city_loso
from intelligence.models.forecast.native.streaming import run_city_loso_native

REAL_CITIES = ["chennai", "hyderabad", "ahmedabad"]


def _load_panels(cities):
    panels = {}
    for c in cities:
        p = pd.read_parquet(f"data/historical/{c}/panel.parquet")
        p["city"] = c
        panels[c] = station_cells_only(p)
    return panels


def test_city_loso_native_matches_pandas_on_a_real_3_city_slice():
    """A real (not synthetic) 3-city slice -- small enough to run BOTH
    the pandas and native paths in one test, per the spec's per-stage
    parity requirement. Full 8-city parity is Task 7's acceptance check,
    not a routine test."""
    panels = _load_panels(REAL_CITIES)
    pandas_result = run_city_loso(panels, HORIZONS, FEATURE_COLUMNS)
    native_result = run_city_loso_native(panels, HORIZONS, FEATURE_COLUMNS)

    assert set(native_result["per_city"]) == set(pandas_result["per_city"])
    for city in REAL_CITIES:
        pandas_rmse = pandas_result["per_city"][city]["rmse"]
        native_rmse = native_result["per_city"][city]["rmse"]
        # Tolerance rationale, from a REAL diagnostic run (not the plan's
        # placeholder rationale -- that guess turned out to be wrong, see
        # below): real 3-city run (chennai/hyderabad/ahmedabad),
        # PYTHONPATH=. python scratch_out/diag_city_loso_parity_timed.py.
        #
        # Observed: pandas=[chennai 23.61, hyderabad 35.20, ahmedabad
        # 45.47], native=[23.71, 35.83, 45.02], delta=[0.10, 0.63, 0.45].
        # Max observed delta: 0.63 (hyderabad).
        #
        # The plan's placeholder comment blamed "LightGBM's histogram-
        # building order is not deterministic". MEASURED AND FALSIFIED: two
        # back-to-back pandas-only run_city_loso() calls on the identical
        # data gave IDENTICAL RMSE to 2dp on all 3 cities (delta 0.0 -- see
        # the task report's diagnostic log). LightGBM's bagging/feature-
        # fraction seeds default to fixed constants when PARAMS sets no
        # explicit seed, so the pandas path alone is perfectly reproducible
        # here; the divergence is NOT run-to-run LightGBM noise.
        #
        # The real source is the native path's two INTENTIONAL design
        # differences from pandas, both already documented as acceptable:
        # (1) mask_unknown_city's ~5% random-unknown-city draw is applied
        # per-city here (seed=i) instead of once over the pooled frame (see
        # this test file's own module docstring / the spec's SS4) -- a
        # different set of rows gets the synthetic "unknown" label, which
        # is a real (if small) change to what the model actually trains on,
        # not just float noise; (2) stream_unit_to_disk downcasts every
        # feature to float32 (Task 3's whole memory-saving point), which
        # shifts LightGBM's histogram bin edges slightly relative to
        # pandas' float64 path. Both are consequences of this task's own
        # design, not a bug -- hence a tolerance rather than a fix.
        #
        # 0.8 is chosen deliberately above the observed 0.63 ceiling (not
        # the placeholder 0.5, which real data exceeds) -- comfortably
        # clear of the measured, explained divergence, still tight enough
        # (~2% of a ~35-45 RMSE) to catch an actual regression.
        assert abs(native_rmse - pandas_rmse) < 0.8, (
            f"{city}: pandas={pandas_rmse}, native={native_rmse}, "
            f"delta={abs(native_rmse - pandas_rmse)}")
