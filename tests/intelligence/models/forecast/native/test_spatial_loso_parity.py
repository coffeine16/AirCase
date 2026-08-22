import pandas as pd
import pytest

from intelligence.models.forecast import HORIZONS
from intelligence.models.forecast.features import FEATURE_COLUMNS, station_cells_only, downcast_panel
from intelligence.models.forecast.validation import spatial_loso
from intelligence.models.forecast.native.streaming import run_spatial_loso_native

REAL_CITIES = ["chennai", "hyderabad", "ahmedabad"]


def _load_pooled_panel(cities):
    panels = []
    for c in cities:
        p = pd.read_parquet(f"data/historical/{c}/panel.parquet")
        p["city"] = c
        panels.append(station_cells_only(p))
    return downcast_panel(pd.concat(panels, ignore_index=True))


def test_spatial_loso_native_matches_pandas_on_a_real_station_subset():
    """A real (not synthetic) subset of stations across 3 cities -- small
    enough to run BOTH the pandas and native paths in one test, per the
    spec's per-stage parity requirement. The full-scale acceptance check
    (all real stations this phase can reasonably cover) is Task 3, not a
    routine test -- spatial-LOSO's per-fold cost makes a full run too slow
    for CI."""
    panel = _load_pooled_panel(REAL_CITIES)
    station_cells = sorted(panel[panel.pm25_station.notna()].cell.unique())
    # First 2 stations per city keeps this test's wall-clock bounded while
    # still covering every city and both same-city and cross-city composite
    # behavior.
    subset = []
    for city in REAL_CITIES:
        city_cells = panel[(panel.city == city) & (panel.pm25_station.notna())].cell.unique()
        subset.extend(sorted(city_cells)[:2])
    panel_subset_stations = panel[panel.cell.isin(subset) | ~panel.pm25_station.notna()]

    def _rmse_for(fn):
        result = fn(panel_subset_stations, HORIZONS, FEATURE_COLUMNS)
        return {s: result["per_station"][s]["rmse"] for s in subset if s in result["per_station"]}

    # Both paths run the FULL spatial_loso/run_spatial_loso_native station
    # loop internally (they derive station_cells from the panel themselves),
    # but restricting the input panel's real station rows to `subset` means
    # only those stations have any pm25_station data to hold out, so only
    # they produce real per_station entries -- the rest of the panel is
    # still present to keep composite/positional fidelity real, exactly
    # matching this project's own station_cells_only discipline.
    pandas_rmse = _rmse_for(spatial_loso)
    native_rmse = _rmse_for(run_spatial_loso_native)

    assert set(native_rmse) == set(pandas_rmse)
    for station, pandas_r in pandas_rmse.items():
        native_r = native_rmse[station]
        # Tolerance rationale, from a REAL diagnostic run (not the plan's
        # placeholder guess -- that turned out to be too tight, see below):
        # real 3-city run (chennai/hyderabad/ahmedabad), ALL 14 real
        # stations, PYTHONPATH=. python <diagnostic script running
        # spatial_loso + run_spatial_loso_native on the full pooled panel>.
        #
        # Observed per-station deltas (pandas vs native rmse): 0.03-1.49
        # across all 14 real stations; max 1.49 (station 8860a24b6dfffff,
        # pandas=21.68, native=23.17). overall_rmse delta 0.23 (25.43 vs
        # 25.66).
        #
        # The two causes are the SAME ones Phase 0 already identified for
        # city-LOSO (per-city mask_unknown_city seeding draws a different
        # ~5% "unknown" subset than the pandas path's single pooled draw;
        # stream_unit_to_disk's float32 downcast shifts LightGBM's
        # histogram bin edges vs pandas' float64 path) -- confirmed by a
        # prior bug hunt on this same function that ruled out a THIRD,
        # much larger cause first: an earlier version of the native
        # training loop forgot to drop the held-out station's own rows
        # from its training frame (loso_exclude only self-excludes a cell
        # from its OWN composite feature, it does not remove that cell's
        # rows from build_features' output -- the pandas reference does
        # this explicitly via `frame[frame.cell != held_out]`). That bug
        # gave deltas of 6-10 RMSE points on a real single-city sanity
        # check -- an order of magnitude past what float32/seeding noise
        # can explain -- and was fixed before this tolerance was set.
        #
        # 2.0 is chosen comfortably above the observed 1.49 ceiling (a
        # ~34% margin, matching Phase 0's own ~27% margin on city-LOSO)
        # -- tight enough to catch a real regression, clear of the
        # measured, explained divergence.
        assert abs(native_r - pandas_r) < 2.0, (
            f"{station}: pandas={pandas_r}, native={native_r}, delta={abs(native_r - pandas_r)}")
