import numpy as np
import pandas as pd
import pytest

from shared.grid import city_cells
from intelligence.models.forecast import HORIZONS
from intelligence.models.forecast.features import FEATURE_COLUMNS, station_cells_only, downcast_panel
from intelligence.models.forecast.validation import spatial_loso
from intelligence.models.forecast.native.streaming import run_spatial_loso_native
import intelligence.models.forecast.native.streaming as streaming

REAL_CITIES = ["chennai", "hyderabad", "ahmedabad"]


def _tiny_panel_with_two_stations():
    """Minimal synthetic fixture for the structural regression test below --
    NOT a real-data run, deliberately small so the test runs in seconds.
    Same shape as test_loso.py's own `_panel_with_two_stations` (kept as a
    local copy rather than a cross-test-file import -- no shared test-fixture
    module exists in this repo yet and one fixture doesn't justify adding
    one)."""
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


def test_run_spatial_loso_native_never_streams_the_held_out_cells_own_rows(monkeypatch):
    """Structural regression test for the leakage bug found while building
    this function: an earlier version of the training loop never dropped
    the held-out station's own rows before streaming a fold's training data
    to disk -- `loso_exclude` only self-excludes a cell from its OWN
    composite feature (spec 3.1's self-exclusion rule), it does not remove
    that cell's rows from build_features' output. Left unfixed, the model
    trains on and is scored against the same station's real labels (see
    this function's own `built.cell != held_out` comment).

    The 2.0-tolerance real-data parity test above WOULD catch a leak this
    size (it moved RMSE 6-10 points), but a narrower future leak could slip
    under that tolerance silently. This test instead asserts the structural
    invariant directly -- monkeypatches stream_unit_to_disk to capture every
    frame written for training, keyed by which station that fold is holding
    out (recovered from `path.parent.name`, since run_spatial_loso_native's
    own work_dir is `scratch_out/native_spatial_loso/<held_out>/<city>.npy`)
    -- and fails immediately if that cell's own rows are ever in it, with no
    dependency on a real multi-minute training run or an RMSE threshold."""
    panel = _tiny_panel_with_two_stations()
    captured = []
    real_stream = streaming.stream_unit_to_disk

    def _spy(frame, path, feature_columns, label_col="y", city_codes=None):
        captured.append((path.parent.name, set(frame["cell"].astype(str).unique())))
        return real_stream(frame, path, feature_columns, label_col=label_col, city_codes=city_codes)

    monkeypatch.setattr(streaming, "stream_unit_to_disk", _spy)

    streaming.run_spatial_loso_native(panel, horizons=[3], feature_cols=FEATURE_COLUMNS)

    assert captured, "run_spatial_loso_native never streamed a training frame"
    for held_out, cells_in_frame in captured:
        assert held_out not in cells_in_frame, (
            f"leak: held-out station {held_out}'s own rows were streamed into "
            f"its own fold's training data ({cells_in_frame})")


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
        # stations, PYTHONPATH=. python scratch_out/diag_spatial_loso_parity.py
        # -- script and full captured output (incl. the per-station numbers
        # below) committed at scratch_out/diag_spatial_loso_parity.py /
        # .log, same precedent as Phase 0's diag_city_loso_parity_timed.py.
        #
        # Observed per-station deltas (pandas vs native rmse): 0.03-1.49
        # across all 14 real stations; max 1.49 (station 8860a24b6dfffff,
        # pandas=21.68, native=23.17). overall_rmse delta 0.23 (25.43 vs
        # 25.66).
        #
        # The causes are the SAME ones Phase 0 already identified for
        # city-LOSO (per-city mask_unknown_city seeding draws a different
        # ~5% "unknown" subset than the pandas path's single pooled draw;
        # stream_unit_to_disk's float32 downcast shifts LightGBM's
        # histogram bin edges vs pandas' float64 path), PLUS a third cause
        # structural to this function specifically: model.py's PARAMS sets
        # bagging_fraction=0.8, bagging_freq=1, so which rows land in each
        # boosting round's bagging sample depends on ROW ORDER, and this
        # function's combine_streamed_units concatenates per-city arrays in
        # `cities` (sorted) order -- a different row order than the pandas
        # path's single pooled build_features call produces. Same class of
        # divergence as the other two (a training-set construction detail
        # that shifts which exact rows/bins LightGBM sees), not a fourth,
        # unexplained one. -- confirmed by a prior bug hunt on this same
        # function that ruled out a much larger, FOURTH cause first: an
        # earlier version of the native training loop forgot to drop the
        # held-out station's own rows from its training frame (loso_exclude
        # only self-excludes a cell from its OWN composite feature, it does
        # not remove that cell's rows from build_features' output -- the
        # pandas reference does this explicitly via
        # `frame[frame.cell != held_out]`). That bug gave deltas of 6-10
        # RMSE points on a real single-city sanity check -- an order of
        # magnitude past what float32/seeding/row-order noise can explain
        # -- and was fixed before this tolerance was set.
        #
        # 2.0 is chosen comfortably above the observed 1.49 ceiling (a
        # ~34% margin, matching Phase 0's own ~27% margin on city-LOSO)
        # -- tight enough to catch a real regression, clear of the
        # measured, explained divergence.
        #
        # This diagnostic (like the routine test above it) passes
        # fires=None to both paths -- production's engine="native"
        # dispatch (train.py) passes real fires instead, and real fires
        # exercise a THIRD divergence cause fires=None never touches: the
        # "citywide representative wind" build_features averages over
        # (features.py's wind_by_hour/wind_ms_by_hour, fed into
        # fire_pressure's wind-alignment term) is a circular mean over
        # whatever cells are in the panel passed to it -- pooled across
        # ALL cities in the pandas path, but per-CITY here (this function
        # calls build_features once per city). First checked with a real
        # single-city fast check (ahmedabad, 2 real stations, real fires:
        # scratch_out/_fast_check_fires_ahmedabad.py -- deltas 0.18/0.20,
        # inside the 0.03-1.49 no-fires range, but a single city can't
        # exercise the cross-city wind-population difference itself).
        # CONFIRMED at the full real 3-city/14-station scale that DOES
        # exercise it (real fires, same cities as this diagnostic:
        # scratch_out/diag_spatial_loso_parity_with_fires.py/.log): max
        # per-station delta 1.56, overall_rmse delta 0.27 (25.75 vs
        # 25.48) -- comfortably under 2.0 and the same order of
        # magnitude as the 1.49 no-fires ceiling above, not a larger,
        # separate divergence. See run_spatial_loso_native's own
        # docstring (streaming.py) for the full writeup of this third
        # cause.
        assert abs(native_r - pandas_r) < 2.0, (
            f"{station}: pandas={pandas_r}, native={native_r}, delta={abs(native_r - pandas_r)}")
