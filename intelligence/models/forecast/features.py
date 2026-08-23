# intelligence/models/forecast/features.py
"""Builds the full training/prediction feature frame (spec section 3.2).
Station cells use their own real history; non-station cells (and a
spatial-LOSO held-out station's own rows) use the real-stations-only
composite from spatial.py at every lag depth — not just the current hour.

STRUCTURE: the expensive work (composites, positional blocks, lags, rolling
medians, fire pressure) does NOT depend on the forecast horizon, so it is
computed ONCE per (cell, ts) and then replicated across horizons. Only
`horizon`, the target clock columns, the climatology lookup (keyed on the
TARGET time), the weather block (also keyed on the TARGET time — see below)
and `y` vary with the horizon. Doing the whole build inside the horizon loop
made one call ~24x more expensive than it needs to be — with 24 horizons on
a 1210-cell x 60-day panel that is the difference between a pipeline stage
that finishes and one that does not. The deleted single-file forecast.py hit
exactly this wall (88 s of a 101 s Cloud Run request) and solved it the same
way.

WEATHER AT TARGET TIME, not issue time: `_MET` (blh_m, wind_ms,
wind_from_deg, temp_c) used to be read at each row's own `ts` — the model
had zero information about weather at the time it was actually forecasting
for, a hard ceiling on 48-72h skill regardless of model quality (a shallow
boundary layer 3 days out matters far more than the boundary layer right
now). It is now looked up at `ts + horizon` instead. TRAINING reads this
from the archived/observed value at that historical hour — genuine
issue-time forecast archives are not available for free at this project's
scale (checked: NOAA GFS's AWS archive is a 30-day rolling window, not a
real archive; Open-Meteo's Single Runs API has true issue-time snapshots
but is paid with no coverage before April 2026; ERA5 doesn't carry BLH at
all via its free GEE mirror) — so this is a deliberate, documented oracle_met
gap between train and serve, not hidden leakage. It is accepted because the
gap was MEASURED, not assumed: Open-Meteo's free Previous Runs API (Delhi,
Nov 2025) put 24h/72h-ahead temperature RMSE at 0.68C/0.76C against a 4.65C
std (excellent) and wind speed at 22%/26% of its mean (real but usable);
boundary layer height cannot be measured this way at all (not in that API),
and is the single strongest predictor this model has — the honest
residual risk in this design, not swept under the rug. SERVING reads the
live forecast: ingestion/collectors/pollers.py's fetch_weather() already
requests forecast_days=3 (72h, this model's max horizon) from Open-Meteo's
live endpoint whenever the window is recent, so the panel handed to
_predict_field already carries genuine forecast values at the target hours
with no separate wiring needed here.

WEATHER PER CELL for blh_m/temp_c/wind, by default: a live multi-point probe
showed real, not noise-level, spatial wind/BLH variation across a city (see
shared.grid.WEATHER_GRID_RES). blh_m/temp_c ARE ALWAYS looked up per (cell,
target ts) -- both are ambient atmospheric-column state, not advected, so a
cell's own local reading is physically correct regardless of `wind_scope`.
wind_ms/wind_from_deg follow `wind_scope` (see build_features' own
docstring): default "per_cell" makes them per-cell too, consistently with
composite_grid/positional_block's spatial weighting -- consistency is the
part that matters. An earlier attempt changed ONLY wind_u/wind_v to
per-cell while leaving composite_grid etc. on citywide wind (two different
wind readings, different scopes, no shared key) and measured worse than
citywide (pooled mean -0.55pp) -- but that measured the inconsistency, not
per-cell wind's real value. Once made consistent and tested on the real
pooled 8-city architecture (not per-city-isolated models), no city was
robustly harmed and several were robustly helped -- see scratchfile_notes
for the full validation history. `wind_scope="citywide"` is kept for
comparison/rollback. A genuinely propagation-aware wind feature (what's
upwind, what's downwind, wind speed setting how far) is real follow-on
work, not this fix -- see scratchfile_notes.
"""
import numpy as np
import pandas as pd

from intelligence.models.forecast.spatial import composite_grid, positional_block, fire_pressure
from intelligence.models.forecast.climatology import build_climatology, SCOPES
from shared.grid import cell_center, haversine_km, circular_mean_deg, latlng_to_cell, cell_to_weather_cell

LAGS = [0, 1, 3, 6, 24]
# roll_med_720 (30d) / roll_med_2160 (90d) fill the memory gap between
# roll_med_168 (7 days) and clim_month (the stationary all-time seasonal
# average) -- "is THIS October running hotter than a typical October" has
# no other feature answering it. Lead-in cost verified directly against
# real data/historical/<city>/panel.parquet ts min/max (not the manifest's
# requested-days field) for all 8 cities: 30d costs 4.1-5.5% of history,
# 90d costs 12.3-16.6%, worst case both on Pune/Ahmedabad (543d, the
# shortest real span) -- safe everywhere. A 365-day window was also
# checked and REJECTED: 50-67% lead-in would leave Pune/Ahmedabad non-NaN
# for barely a third of their rows, at which sparsity the feature trains
# as a disguised city/history-length proxy rather than genuine annual
# memory, compounding the walk-forward city-mix confound already
# documented in scratchfile_notes/forecast-data-scale-and-coverage.md.
# See the computation itself, below the horizon-independent lag/roll_med_24/
# 168 block, for why these two are built from a daily-resampled series
# rather than a naive hourly rolling window, and for the climatology-
# shrinkage fallback that fills the lead-in population (the 4.1-16.6%
# above) with a confidence-weighted blend instead of leaving it NaN.
_MET = ["blh_m", "wind_ms", "wind_from_deg", "temp_c"]
_STATIC = ["lu_road", "lu_industrial", "lu_traffic"]

# Pure-periodic quantities, fed to the model as sin/cos pairs -- never as a
# raw ordinal/degree value. A raw encoding forces the model to DISCOVER the
# wrap-adjacency (23:00 next to 00:00, December next to January, 359 deg
# next to 1 deg) from training examples that happen to land near the
# boundary; if those are sparse, boosting may never place a split there at
# all. sin/cos makes the adjacency true by construction.
_CYCLICAL = {"target_hour": 24.0, "target_dow": 7.0,
             "target_month": 12.0, "target_doy": 365.25}
_MET_MODEL_COLS = [c for c in _MET if c not in _CYCLICAL and c not in ("wind_ms", "wind_from_deg")]
_CYCLICAL_MODEL_COLS = [f"{c}_{trig}" for c in _CYCLICAL for trig in ("sin", "cos")]

# Wind is a VECTOR (speed and direction together), not two independently
# useful scalars -- feeding the model wind_ms and wind_from_deg_sin/cos as
# three SEPARATE features (the pre-existing design) forces it to
# rediscover their interaction from splits, on a signal this data-starved
# (real fire/event rows are a tiny fraction of the panel) that is not a
# reasonable thing to ask of it. Same class of fix as spatial.py's
# composite_grid/fire_pressure wind-vector decomposition (see that
# module's docstring), one layer up: applied to the model's own raw
# inputs instead of the spatial weighting kernel. Standard meteorological
# u/v convention (u=eastward component, v=northward component) computed
# from wind_from_deg ("blows FROM" compass bearing) and wind_ms, at
# TARGET time (see the module docstring's A1 note) since both are already
# looked up there via met_lookup.
_WIND_VECTOR_COLS = ["wind_u", "wind_v"]

FEATURE_COLUMNS = (
    [f"lag_{k}" for k in LAGS] + ["roll_med_24", "roll_med_168", "roll_med_720", "roll_med_2160",
    "has_station", "distance_to_nearest_station_km", "nearby_stations_delta"]
    # pos_0 excluded deliberately: it is an exact algebraic duplicate of
    # nearby_stations_delta + lag_0 (diffed real code output, max
    # discrepancy 3e-8) -- both are composite_grid evaluated at the cell's
    # own center with the same inputs, computed twice. positional_block
    # still computes it (index 0 of its fixed 7-column shape; see that
    # function's docstring), so this doesn't touch the spatial computation
    # at all, only what the model actually trains on. pos_1..pos_6 (the
    # up-to-6 real neighbor cells) are genuinely distinct spatial info.
    + [f"pos_{i}" for i in range(1, 7)] + ["fire_pressure_regional", "fires_6h", "frp_6h"]
    + _MET_MODEL_COLS + _WIND_VECTOR_COLS + _STATIC + ["clim_dow_hour", "clim_month"]
    + _CYCLICAL_MODEL_COLS + ["horizon", "city"]
)


def station_cells_only(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows for cells that carry a real station reading somewhere in
    `panel` (the cell's FULL time series, not just its non-NaN rows --
    build_features' own lag/rolling groupby needs that whole history).

    Every call site that runs build_features with restrict_to_station_
    cells=True only ever keeps station-cell rows in its output (see that
    flag's docstring), so building composite/positional features for a
    city's other ~95-99% of grid cells first, only to discard them, is
    pure waste -- both in compute and, on a real multi-city panel, in
    memory. Filter here, before that expensive work runs, and every
    downstream result is byte-identical to building on the full grid."""
    station_cells = panel.loc[panel.pm25_station.notna(), "cell"].unique()
    return panel[panel.cell.isin(station_cells)]


def downcast_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Shrinks a panel's memory footprint. Measured on Bengaluru's real
    2-year historical panel: 3.28GB -> ~1.1GB. The low-cardinality string
    columns account for most of it -- `city` is the SAME string repeated
    16.75M times, but arrow-backed string columns don't dedupe the way a
    categorical does.

    Call this again after `pd.concat`-ing panels from different cities, not
    just once at load time: `pd.concat` silently reverts a categorical
    column back to plain string dtype whenever the pieces being concatenated
    don't already share an identical category set (confirmed empirically —
    two single-city frames, each with its OWN one-city categorical, concat
    into a `str`-dtype column). A pooled multi-city panel's `full_panel =
    pd.concat(panels_by_city.values())` is exactly that case, and it is also
    exactly the line that failed with an ArrowMemoryError on a real 3-city
    run: loading all 3 cities' panels, concatenating them, and only THEN
    re-widening the string columns back out used more memory than was free.

    Numeric columns stay at their original width (float64/int64), not
    downcast to float32/int32: build_features fills lag/roll_med/composite
    columns via `X.loc[mask, col] = <float64 array from composite_grid>`,
    and pandas refuses to assign a float64 value into a float32 column
    losslessly (raises, doesn't silently widen) — confirmed by trying it and
    watching the existing, already-reviewed LOSO/train_and_promote tests
    fail with exactly that TypeError. The string columns are where the
    memory actually was (city/cell/ward_id/ward_name = 39% of a real panel's
    footprint from 4 of ~19 columns), so this alone is the safe, high-value
    part of the fix.

    Mutates `df` in place rather than defensively copying it first. Every
    call site (ingestion's per-city load, train.py's full_panel, run_city_
    loso's train_panel) passes the fresh, single-owner result of a
    pd.read_parquet or pd.concat call that no other name still references,
    so there is no aliasing to protect against — and on a real 4-city
    panel (~14.7GB post-concat) that defensive copy briefly held BOTH the
    string-typed original and its categorical copy at once, which is a
    measured, real contributor to a training Job's OOM."""
    for col in ("cell", "ward_id", "ward_name", "city"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def _station_matrix(panel: pd.DataFrame):
    st = panel[panel.pm25_station.notna()][["cell", "ts", "pm25_station"]].drop_duplicates(["cell", "ts"])
    if st.empty:
        return np.array([]), np.array([]), np.zeros((panel.ts.nunique(), 0)), []
    cells = sorted(st.cell.unique())
    centers = {c: cell_center(c) for c in cells}
    lat = np.array([centers[c][0] for c in cells])
    lon = np.array([centers[c][1] for c in cells])
    hours = pd.DatetimeIndex(sorted(panel.ts.unique()))
    val = (st.pivot(index="ts", columns="cell", values="pm25_station")
             .reindex(index=hours, columns=cells).values)
    return lat, lon, val, cells


def _lookup_scale(keys: pd.DataFrame, key_col: str, tables: dict[str, pd.Series], scale: str) -> np.ndarray:
    """cell -> ward -> city fallback merge for ONE climatology scale, most-
    specific-scope-wins (same semantics as climatology.lookup_climatology).
    `keys` must carry cell/ward_id/city (as str) plus an int64 `key_col` --
    the caller decides what time reference produced that key (climatology_
    columns uses target time; climatology_at_obs_time below uses raw
    observation time), this function only does the fallback merge."""
    vals = None
    for scope, ident_col in zip(SCOPES, ("cell", "ward_id", "city")):
        table = tables.get(f"{scope}_{scale}")
        if table is None or len(table) == 0:
            continue
        td = table.rename("_v").reset_index()
        td.columns = [ident_col, key_col, "_v"]
        td[ident_col] = td[ident_col].astype(str)
        td[key_col] = td[key_col].astype("int64")
        merged = keys[[ident_col, key_col]].merge(td, on=[ident_col, key_col], how="left")["_v"]
        vals = merged if vals is None else vals.fillna(merged)
    return np.full(len(keys), np.nan) if vals is None else vals.to_numpy(dtype=float)


def climatology_columns(frame: pd.DataFrame, tables: dict[str, pd.Series]):
    """Vectorised cell -> ward -> city climatology lookup, keyed on each row's
    TARGET time (ts + horizon). Returns (clim_dow_hour, clim_month) arrays.

    Same fallback semantics as climatology.lookup_climatology (most specific
    scope that has a value wins), but as three merges per scale instead of a
    per-row Python `.loc` scan — that scan was three full itertuples() passes
    over the frame and, on a real panel, the single slowest thing in here
    after the composites.
    """
    tgt = frame["ts"] + pd.to_timedelta(frame["horizon"], unit="h")
    keys = pd.DataFrame({
        "cell": frame["cell"].astype(str).values,
        "ward_id": frame["ward_id"].astype(str).values,
        "city": frame["city"].astype(str).values,
        "how": (tgt.dt.dayofweek * 24 + tgt.dt.hour).astype("int64").values,
        # "month" scale keys on day-of-year now, matching climatology.py's
        # _smoothed_doy_table (fixes the calendar-month hard-boundary
        # discontinuity — see that module's docstring). clip, not the
        # scalar min() lookup_climatology uses, since this is a Series.
        "month": tgt.dt.dayofyear.clip(upper=365).astype("int64").values,
    })
    return (_lookup_scale(keys, "how", tables, "dow_hour"),
            _lookup_scale(keys, "month", tables, "month"))


def climatology_at_obs_time(cell: pd.Series, ward_id: pd.Series, city: pd.Series,
                             ts: pd.Series, tables: dict[str, pd.Series]) -> np.ndarray:
    """The "month"-scale climatology (cell -> ward -> city fallback), keyed
    on `ts` directly -- NOT target time (ts + horizon) like climatology_
    columns above. roll_med_720/roll_med_2160 are horizon-INDEPENDENT
    (computed once per (cell, ts), before horizon expansion -- see the
    horizon-independent block in build_features), so their climatology-
    shrinkage fallback needs "what's typical for this cell RIGHT NOW", not
    "what's typical at the time we're forecasting for" -- a different
    question from what clim_month answers, hence its own lookup rather than
    reusing that column."""
    doy = ts.dt.dayofyear.clip(upper=365).astype("int64")
    keys = pd.DataFrame({"cell": cell.astype(str).values, "ward_id": ward_id.astype(str).values,
                          "city": city.astype(str).values, "month": doy.values})
    return _lookup_scale(keys, "month", tables, "month")


def attach_climatology(frame: pd.DataFrame, tables: dict[str, pd.Series]) -> pd.DataFrame:
    """Returns a COPY of `frame` with clim_dow_hour/clim_month recomputed from
    `tables`. Exists so a caller with a time-based train/test split can rebuild
    the climatology from TRAIN-ONLY data and re-attach it, without paying for a
    second full build_features pass (see train.py's walk-forward loop)."""
    out = frame.copy()
    out["clim_dow_hour"], out["clim_month"] = climatology_columns(out, tables)
    return out


def _compose_masks(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Fold a mask taken AFTER `first` was applied back into `first`'s frame.

    `keep_mask` is indexed against the original panel row order (the `y` shift
    below is computed from a groupby over it), so a second filter applied to the
    already-filtered X cannot simply replace it.
    """
    out = first.copy()
    out[np.flatnonzero(first)] = second
    return out


def build_features(panel: pd.DataFrame, horizons: list[int],
                    loso_exclude: str | None = None,
                    fires: pd.DataFrame | None = None,
                    clim_tables: dict[str, pd.Series] | None = None,
                    restrict_to_station_cells: bool = False,
                    wind_scope: str = "per_cell",
                    weather: pd.DataFrame | None = None,
                    only_last_ts: bool = False) -> pd.DataFrame:
    """`clim_tables`, when given, is used verbatim instead of building the
    climatology from `panel` — the hook a time-split caller needs so that a
    test row's climatology is never computed from that row's own future. A
    caller that passes BOTH `clim_tables` and `loso_exclude` is responsible
    for having built those tables with `build_climatology(..., exclude_cell=)`;
    the default path does it automatically.

    `restrict_to_station_cells`: skip horizon-expanding any cell that could
    never carry a labeled `y` (every non-station cell — `y` is
    `pm25_station.shift(-h)`, which is NaN everywhere there is no station).
    The horizon-independent block `X` is still built from the FULL panel
    first, so composite/positional fidelity is untouched; this only decides
    which (cell, ts) rows get replicated across horizons. Every existing
    training call site ends in `.dropna(subset=["y"])` anyway, so those rows
    were always going to be discarded — the only difference is discarding
    them ONCE instead of building and then dropping len(horizons) copies of
    them. On a real multi-city 2-year panel (~50M base rows, <40 station
    cells) that is the difference between a training run that fits in memory
    and one that needs tens of GB to hold rows nobody was going to use.
    Leave False for any prediction path (e.g. _predict_field) that needs
    every cell, station or not.

    `wind_scope`: "per_cell" (default, SHIPPED) or "citywide" (kept for
    comparison/rollback, no longer the default). "per_cell" feeds wind_u/
    wind_v AND composite_grid/positional_block's spatial weighting from each
    cell's OWN local weather-grid value consistently (fire_pressure stays
    citywide-representative regardless of this flag — its natural physical
    anchor is the FIRE's location, not the receiving cell's, a different
    question; see fire_pressure's own per-fire-location wind lookup instead).
    "citywide" feeds all of those from ONE representative city-average value.

    History: an earlier experiment changed ONLY wind_u/wind_v to per-cell
    while leaving composite_grid etc. on citywide wind — two different wind
    readings, at different scopes, feeding the same model with no shared
    key, which was a confounded test (it measured "does an internally
    inconsistent wind design hurt", not "does per-cell wind data help").
    Once fixed to be consistent, a seed+thread-pinned sweep across the real
    pooled 8-city architecture (not per-city-isolated models) found no city
    robustly harmed by per_cell wind, and several robustly helped
    (Chennai/Kolkata/Pune/Ahmedabad consistent-positive across 3 seeds) —
    see scratchfile_notes for the full validation history before changing
    this default again.

    `weather`: optional raw weather data (ts, weather_cell, wind_from_deg,
    wind_ms -- the shape ingestion/collectors/pollers.py::fetch_weather()
    returns, e.g. data/historical/<city>/weather.parquet). When given,
    fire_pressure_regional is computed with wind sampled AT EACH FIRE'S OWN
    LOCATION (its lat/lon resolved to a weather-grid cell) rather than one
    citywide value -- the fire is the physical anchor for "which way is this
    smoke plume going", not the receiving cell or the city average. This is
    independent of `wind_scope` above: it improves fire_pressure's own
    accuracy in EITHER wind_scope arm, it does not make fire_pressure
    per-cell in the same sense wind_u/wind_v or composite_grid can be.
    Deliberately NOT derived from `panel`/`cells` alone -- a training call
    typically pre-filters `panel` to station cells only (a few dozen), which
    would leave most of a city's fires with no matching weather-grid
    coverage to look up; `weather` carries the full city's spatial coverage
    independent of whatever rows survived that filtering. None (default,
    backward compatible) reproduces the existing citywide-only behaviour
    exactly for every caller that hasn't been updated to pass it."""
    # reset_index is required, not cosmetic: the fire_pressure merge below
    # produces a fresh 0..n-1 RangeIndex on X, and later code re-joins X
    # against THIS frame (p) positionally. Any caller that passes a sliced
    # or otherwise non-0-based panel would silently break that alignment.
    p = panel.sort_values(["cell", "ts"]).reset_index(drop=True).copy()
    if fires is None:
        fires = pd.DataFrame(columns=["ts", "lat", "lon", "frp", "confidence"])
    station_lat, station_lon, station_val, station_cells = _station_matrix(p)
    hours = pd.DatetimeIndex(sorted(p.ts.unique()))
    # wind_by_hour is ISSUE-time wind, used below only to weight composite_grid/
    # positional_block's spatial fill (which real stations are upwind of a
    # cell RIGHT NOW) -- unrelated to _MET's TARGET-time lookup further down.
    #
    # Weather now varies by cell (see shared.grid.WEATHER_GRID_RES): `p` is one
    # row per (cell, ts), so a naive drop_duplicates("ts") would keep whichever
    # cell happens to sort first -- an arbitrary pick, not "the citywide value".
    # composite_grid/positional_block still take one representative wind value
    # per hour (their query-point-specific wind is a separate, not-yet-built
    # extension), so aggregate properly instead: wind_from_deg is circular (350
    # and 10 degrees are 20 degrees apart, not 340), hence sin/cos averaging,
    # not a plain mean. Row-weighted by H3 cell (not by weather grid cell) on
    # purpose -- a weather cell covering more of the city's area should count
    # for more of "what's the city's wind right now". Inlined rather than
    # calling circular_mean_deg per group: a per-group Python call over
    # thousands of groups is the exact pattern this file vectorises elsewhere
    # (see _fire_features).
    _wind_rad = np.radians(p["wind_from_deg"].values)
    # index=p["ts"] (the tz-aware Series), not p["ts"].values -- grouping on
    # the bare numpy array loses the tz-aware dtype `hours` is built from, so
    # the reindex below would silently mismatch and come back all-NaN.
    _sin_by_ts = pd.Series(np.sin(_wind_rad), index=p["ts"]).groupby(level=0).mean()
    _cos_by_ts = pd.Series(np.cos(_wind_rad), index=p["ts"]).groupby(level=0).mean()
    wind_by_hour = (np.degrees(np.arctan2(_sin_by_ts, _cos_by_ts)) % 360.0).reindex(hours).values
    # Same issue-time semantics, for spatial.py's wind-speed-scaled decay --
    # a calm hour and a gale should not spread a station's/fire's influence
    # by the same fixed distance. Median (not mean) across cells per this
    # project's own "never the mean" rule. None (not present in `p`) falls
    # back to composite_grid/fire_pressure's original fixed-DIST_DECAY_KM
    # behaviour.
    wind_ms_by_hour = (p.groupby("ts").wind_ms.median().reindex(hours).values
                        if "wind_ms" in p else None)
    # blh_m/temp_c: per (cell, ts). Both are ambient atmospheric-column state
    # -- set by local surface heating and regional synoptic conditions, not
    # carried in from elsewhere -- so a cell's own local reading is physically
    # the right value to look up at target time (see notekeeper physics audit,
    # 2026-08-20).
    #
    # wind_ms/wind_from_deg: per-cell by default (wind_scope="per_cell") --
    # see this function's own docstring for the validation history. Every
    # wind-derived quantity in this function (wind_u/wind_v here, and
    # composite_grid/positional_block's spatial weighting at the call sites
    # further down) reads the SAME scope consistently -- that consistency is
    # what an earlier, confounded attempt got wrong (per-cell wind_u/wind_v
    # paired against citywide composite features, two different wind
    # readings with no shared key), not per-cell wind itself.
    _met_cols_percell = [m for m in ("blh_m", "temp_c") if m in p]
    if wind_scope == "per_cell":
        _met_cols_percell = _met_cols_percell + [m for m in ("wind_ms", "wind_from_deg") if m in p]
    elif wind_scope != "citywide":
        raise ValueError(f"wind_scope must be 'citywide' or 'per_cell', got {wind_scope!r}")
    met_lookup = p[["cell", "ts"] + _met_cols_percell].rename(columns={"ts": "_met_ts"})
    _wind_dir_by_ts = pd.Series(wind_by_hour, index=hours)
    _wind_ms_by_ts = pd.Series(wind_ms_by_hour, index=hours) if wind_ms_by_hour is not None else None

    if clim_tables is None:
        clim_tables = build_climatology(p, exclude_cell=loso_exclude)

    cells = sorted(p.cell.unique())
    cell_centers = {c: cell_center(c) for c in cells}
    for s in station_cells:
        cell_centers.setdefault(s, cell_center(s))

    # Per-cell wind for composite_grid/positional_block's spatial weighting,
    # only built when wind_scope="per_cell" -- see the docstring and
    # met_lookup's construction comment above. Columns aligned to `cells`'
    # own order via reindex, so column i here is cells[i] everywhere below.
    # NaN where a cell's weather-grid parent has no data that hour (e.g. the
    # still-open boundary gap in shared.grid.weather_grid_cells' pre-fix
    # data) -- composite_grid already treats NaN wind as "no weight from any
    # station for this query point, this hour", isolated to that one column,
    # the same missing-value convention used everywhere else in this file.
    if wind_scope == "per_cell":
        wind_dir_kernel = (p.pivot(index="ts", columns="cell", values="wind_from_deg")
                            .reindex(index=hours, columns=cells).to_numpy())
        wind_ms_kernel = (p.pivot(index="ts", columns="cell", values="wind_ms")
                           .reindex(index=hours, columns=cells).to_numpy()
                           if "wind_ms" in p else None)
    else:
        wind_dir_kernel = wind_by_hour
        wind_ms_kernel = wind_ms_by_hour

    # loso_exclude masks the station's READINGS; it must mask its IDENTITY too.
    # A held-out cell that still reports has_station=True / distance 0.0 is a
    # train/serve mismatch — a genuinely station-less cell always reports
    # False and a positive distance, which is the regime LOSO is meant to
    # measure.
    station_set = set(station_cells) - ({loso_exclude} if loso_exclude is not None else set())
    nearest_station_km = {}
    for c in cells:
        cand = [s for s in station_cells if s != c] if c == loso_exclude else station_cells
        if len(cand) == 0:
            nearest_station_km[c] = float("nan")
            continue
        lat, lon = cell_centers[c]
        nearest_station_km[c] = min(
            haversine_km(lat, lon, cell_centers[s][0], cell_centers[s][1]) for s in cand)

    # ---- horizon-INDEPENDENT block: computed exactly once ----
    g = p.groupby("cell", group_keys=False)
    feats = {f"lag_{k}": g.pm25_station.shift(k) for k in LAGS}
    feats["roll_med_24"] = g.pm25_station.transform(lambda s: s.shift(1).rolling(24, min_periods=6).median())
    feats["roll_med_168"] = g.pm25_station.transform(lambda s: s.shift(1).rolling(168, min_periods=24).median())

    # roll_med_720 (30d) / roll_med_2160 (90d): fill the memory gap noted
    # above LAGS. Computed via daily median first, THEN a rolling median
    # over that short (~550-730 point) daily series -- not a naive
    # hourly-window rolling median (~13k-17.5k points). This is the vectorised
    # form of "share one buffer across window sizes instead of recomputing
    # each independently": one groupby-median pass builds the daily series
    # once, and BOTH windows roll over that same series. A hand-rolled
    # per-row Python sliding-window (deque + slice + median) was considered
    # and rejected -- pandas' own rolling().median() already maintains an
    # O(log w) order-statistic structure internally, in compiled code, so a
    # Python-level loop across ~1e4-1e7 rows would lose to it on constant
    # factor alone despite being "more clever" in big-O. Daily-first is the
    # actual win because it shrinks n (the number of points rolled over),
    # not because it reimplements the rolling primitive.
    #
    # This also isn't merely a compute shortcut: a true hourly rolling
    # median over 720/2160 raw readings implicitly weights a day with more
    # station uptime more heavily than a sparse day. Medianing per day first
    # gives every day equal say before the second median runs over days --
    # more consistent with this project's own robust-statistics principle
    # (never let sampling density masquerade as signal).
    _date = p["ts"].dt.floor("D")
    _daily = (p.assign(_date=_date).groupby(["cell", "_date"])["pm25_station"]
              .median().sort_index())
    _dg = _daily.groupby(level="cell", group_keys=False)
    # min_periods=1, not 10/30 -- the OLD hard cutoff just returned NaN for
    # every lead-in row (the first ~30d/90d of each cell's own history,
    # ~5.5-16.6% of the panel). Shrinkage below replaces that cliff: use
    # whatever real trailing days exist, down-weighted by how few there
    # are, blended with this cell's own climatology instead of thrown away.
    _roll_720 = _dg.transform(lambda s: s.shift(1).rolling(30, min_periods=1).median())
    _roll_2160 = _dg.transform(lambda s: s.shift(1).rolling(90, min_periods=1).median())
    # Real trailing-day COUNT within the window, same shift/rolling shape --
    # the confidence weight below needs "how much real data is actually in
    # this window", which .median() alone doesn't expose.
    _cnt_720 = _dg.transform(lambda s: s.shift(1).rolling(30, min_periods=1).count())
    _cnt_2160 = _dg.transform(lambda s: s.shift(1).rolling(90, min_periods=1).count())
    # Broadcast the daily-cadence values back onto every hourly row for that
    # (cell, date) via an explicit merge, NOT a MultiIndex built from
    # `.values` -- `Series.dt.floor("D").values` silently strips the tz off
    # a tz-aware datetime (documented gotcha, this project's own CLAUDE.md),
    # which desynced this exact merge key from `_daily`'s own tz-aware
    # groupby index and made every row NaN until caught by the test below.
    _roll_lookup = pd.DataFrame({
        "cell": _roll_720.index.get_level_values("cell"),
        "_date": _roll_720.index.get_level_values("_date"),
        "roll_med_720": _roll_720.to_numpy(),
        "roll_med_2160": _roll_2160.to_numpy(),
        "cnt_720": _cnt_720.to_numpy(),
        "cnt_2160": _cnt_2160.to_numpy(),
    })
    _merged = p[["cell"]].assign(_date=_date).merge(_roll_lookup, on=["cell", "_date"], how="left")

    # Climatology-shrinkage fallback for the lead-in population (pre-flight
    # audit, 2026-08-20; see the earlier LAGS comment for the underlying
    # gap and scratchfile_notes for the sanity-check discussion this rests
    # on). w = fraction of the FULL window that's real data (1.0 once a
    # cell has 30/90 real trailing days -- the mature ~83-88% of the panel,
    # byte-identical to the old hard-cutoff behaviour there). Below that,
    # blend the raw (noisy, few-point) trailing median with this cell's own
    # obs-time climatology, weighted by how little real data backs the raw
    # side -- textbook empirical-Bayes/cold-start shrinkage, same family as
    # a Kalman filter's warm-up prior or a meteorological "climatological
    # normal" fallback for a too-short station record. Deliberately NOT a
    # bigger window for cities with more history: the window's own target
    # size (30d/90d) stays IDENTICAL for every cell everywhere -- varying
    # it by how much history a city happens to have is exactly the
    # disguised city/history-length proxy that got the naive 365d window
    # rejected. Only how the SAME window degrades when it isn't yet full
    # changes here.
    #
    # GATED to real station cells only (`_is_station_cell`, via the same
    # `station_cells` this function already resolved above) -- without this,
    # a non-station cell's ALL-NaN raw series (count=0 everywhere) still
    # falls to the `_has_raw=False` branch below and picks up its WARD's
    # climatology, silently giving every non-station cell in a ward that
    # contains any real station a borrowed trend value it never earned.
    # roll_med_720/2160 mean "this cell's own recent trend" -- a non-station
    # cell has none, and must stay NaN, exactly as before this fix (LAGS'
    # composite-grid spatial fill already covers that population a
    # different, deliberately spatial way -- see the fill block above).
    # `_has_raw=True` can only occur for a real station cell in the first
    # place (a non-station cell's raw rolling median is always NaN, never
    # partially populated), so only the fallback branch needs the guard.
    _clim_at_obs = climatology_at_obs_time(p["cell"], p["ward_id"], p["city"], p["ts"], clim_tables)
    _is_station_cell = p["cell"].isin(station_cells).to_numpy()
    for _w, _raw_col, _cnt_col, _out_col in (
        (30, "roll_med_720", "cnt_720", "roll_med_720"),
        (90, "roll_med_2160", "cnt_2160", "roll_med_2160"),
    ):
        _raw = _merged[_raw_col].to_numpy()
        _weight = np.clip(_merged[_cnt_col].to_numpy() / _w, 0.0, 1.0)
        _has_clim = ~np.isnan(_clim_at_obs)
        _has_raw = ~np.isnan(_raw)
        _blended = np.where(_has_clim & _has_raw, _raw * _weight + _clim_at_obs * (1 - _weight),
                             np.where(_has_clim & _is_station_cell, _clim_at_obs, _raw))
        feats[_out_col] = _blended

    for m in _STATIC:
        if m in p:
            feats[m] = p[m]
    X = pd.DataFrame(feats, index=p.index)
    X["cell"] = p.cell
    X["ts"] = p.ts
    X["city"] = p.city
    X["ward_id"] = p.ward_id
    X["has_station"] = p.cell.isin(station_set)
    X["distance_to_nearest_station_km"] = p.cell.map(nearest_station_km).fillna(0.0)
    X.loc[X.has_station, "distance_to_nearest_station_km"] = 0.0

    # roll_med_24/168/720/2160 are computed from the RAW per-cell groupby
    # above, which is correct for every station cell and already NaN for
    # every non-station cell (no history to roll over) -- except the ONE
    # case that matters: the loso_exclude cell IS a station, so its rolling
    # medians are its own real history unless explicitly nulled here.
    if loso_exclude is not None:
        X.loc[X.cell == loso_exclude,
              ["roll_med_24", "roll_med_168", "roll_med_720", "roll_med_2160"]] = np.nan

    # composite-based fill for non-station cells (and a LOSO station's own rows)
    if len(station_cells) > 0:
        q_lat = np.array([cell_centers[c][0] for c in cells])
        q_lon = np.array([cell_centers[c][1] for c in cells])
        excl = None
        if loso_exclude is not None and loso_exclude in station_cells:
            s_idx = station_cells.index(loso_exclude)
            excl = np.zeros((len(cells), len(station_cells)), dtype=bool)
            cell_idx = {c: i for i, c in enumerate(cells)}
            excl[cell_idx[loso_exclude], s_idx] = True

        needs_fill = (~X.has_station.values) | (X.cell == loso_exclude).values
        for k in LAGS:
            shifted_val = np.roll(station_val, k, axis=0)
            shifted_val[:k, :] = np.nan
            comp = composite_grid(q_lat, q_lon, station_lat, station_lon,
                                   shifted_val, wind_dir_kernel, wind_ms_kernel, exclude=excl)
            merged = _grid_to_rows(comp, hours, cells, X, f"_comp_{k}")
            X.loc[needs_fill, f"lag_{k}"] = merged[needs_fill]

        comp_now = composite_grid(q_lat, q_lon, station_lat, station_lon,
                                   station_val, wind_dir_kernel, wind_ms_kernel, exclude=excl)
        X["nearby_stations_delta"] = _grid_to_rows(comp_now, hours, cells, X, "_comp_now") - X["lag_0"].values

        # ONE merge for all 7 positional columns instead of 7 masked
        # assignments per cell (that inner loop was O(n_cells) full-frame
        # boolean writes -- 8k passes over a 1.7M-row frame on a real panel).
        # wind_dir_kernel/wind_ms_kernel: (n_t,) citywide broadcasts across
        # this cell's own positional block unchanged; (n_t, n_cells) per_cell
        # passes THIS cell's own column, applied uniformly to its 7-point
        # block (its own centre + neighbours) -- neighbours are ~460m away,
        # almost always inside the same ~3.2km weather-grid parent anyway.
        pos = np.full((len(hours), len(cells), 7), np.nan)
        for ci, c in enumerate(cells):
            cell_wind_dir = wind_dir_kernel[:, ci] if wind_scope == "per_cell" else wind_dir_kernel
            cell_wind_ms = (wind_ms_kernel[:, ci] if wind_scope == "per_cell" else wind_ms_kernel)
            pos[:, ci, :] = positional_block(
                c, station_lat, station_lon, station_val, cell_wind_dir, cell_wind_ms,
                exclude=(excl[[ci]] if excl is not None else None))
        pos_df = pd.DataFrame(pos.reshape(len(hours) * len(cells), 7),
                               columns=[f"pos_{i}" for i in range(7)])
        # .repeat on the DatetimeIndex itself, NOT .values -- .values strips
        # the tz and silently breaks the merge (project-wide gotcha).
        pos_df["ts"] = hours.repeat(len(cells))
        pos_df["cell"] = np.tile(np.asarray(cells), len(hours))
        X = X.merge(pos_df, on=["cell", "ts"], how="left")
    else:
        # no stations at all: lags stay NaN, LightGBM treats them as missing
        X["nearby_stations_delta"] = np.nan
        for i in range(7):
            X[f"pos_{i}"] = np.nan

    # Real per-fire-location wind, when `weather` (raw, weather_cell-keyed)
    # is available -- see build_features' own docstring for why this can't
    # be derived from `p`/`cells` alone. A merge, not a per-row Python loop:
    # fire counts are small (hundreds-thousands per city per window, not
    # panel-sized), but a merge is exactly as fast and stays consistent with
    # this project's vectorisation discipline elsewhere.
    fire_wind_from_deg = fire_wind_ms = None
    if weather is not None and len(fires) > 0:
        wx_cols = ["ts", "weather_cell", "wind_from_deg"] + (["wind_ms"] if "wind_ms" in weather else [])
        wx = weather[wx_cols].copy()
        wx["ts"] = pd.to_datetime(wx.ts, utc=True).dt.floor("h")
        # weather_grid_cells() is a UNIFORM, complete grid by construction
        # (shared.grid.weather_grid_cells's own docstring) -- gaps here are
        # in the FETCHED data, not the grid definition (e.g. weather.parquet
        # pulled before a coverage fix, or a one-off fetch failure), and
        # coverage is complete-or-absent PER CELL, never partial-by-hour
        # (measured: every weather cell checked had 100% hourly coverage
        # over its full history -- see scratchfile_notes/wind coherence
        # measurement). So "which cells have data at all" is a static set,
        # computed once, not a per-hour search.
        present_wc = set(wx.weather_cell.unique())
        fire_wc_raw = [cell_to_weather_cell(latlng_to_cell(lat, lon))
                        for lat, lon in zip(fires.lat, fires.lon)]
        if present_wc:
            # A real neighbouring cell's actual reading is a much closer
            # approximation than a citywide average -- wind rarely changes
            # much across one ~3.2km weather-grid cell, and the gap orphans
            # SPECIFIC cells, not their surroundings. Citywide (in
            # fire_pressure's own NaN fallback below) is the last resort,
            # only when NO weather cell has data at all.
            _present_centers = {wc: cell_center(wc) for wc in present_wc}
            def _nearest_present(wc):
                if wc in present_wc:
                    return wc
                lat0, lon0 = cell_center(wc)
                return min(present_wc, key=lambda c: haversine_km(lat0, lon0, *_present_centers[c]))
            fire_wc = [_nearest_present(wc) for wc in fire_wc_raw]
        else:
            fire_wc = fire_wc_raw   # nothing fetched at all; every lookup below
                                     # misses, fire_pressure's own NaN handling
                                     # falls back to citywide as the true last resort
        fire_lookup = pd.DataFrame({
            "ts": pd.to_datetime(fires.ts, utc=True).dt.floor("h"),
            "weather_cell": fire_wc,
        })
        merged = fire_lookup.merge(wx, on=["ts", "weather_cell"], how="left")
        fire_wind_from_deg = merged["wind_from_deg"].to_numpy()
        fire_wind_ms = merged["wind_ms"].to_numpy() if "wind_ms" in merged else None

    fp = fire_pressure(cells, fires, hours, wind_by_hour, wind_ms_by_hour,
                        fire_wind_from_deg=fire_wind_from_deg, fire_wind_ms=fire_wind_ms)
    X = X.merge(fp, on=["cell", "ts"], how="left")
    if "fires_6h" in p:
        X["fires_6h"] = p.fires_6h.values
        X["frp_6h"] = p.frp_6h.values

    # Drop rows that can never be y-labeled BEFORE the horizon expansion below,
    # not after -- see the docstring. Safe superset reduction: every row kept
    # here was going to survive expansion unchanged in every horizon anyway.
    # `keep_mask` also has to gate the `y` shift below: that's computed from
    # `g`, the groupby over the ORIGINAL (unfiltered) `p`, so it stays aligned
    # to p's row order -- X's own index gets reset by the filter and can't be
    # used to re-select from `g` afterward.
    keep_mask = None
    if restrict_to_station_cells:
        allowed = set(station_cells) | ({loso_exclude} if loso_exclude is not None else set())
        keep_mask = X.cell.isin(allowed).to_numpy()
        X = X[keep_mask].reset_index(drop=True)

    # `only_last_ts` is the same trick one level over: SERVING keeps only the
    # final timestamp's rows, so replicating every other hour across every
    # horizon builds tens of millions of rows to throw away.
    #
    # It has to happen HERE, after the horizon-independent block is built from
    # the whole panel, because that block is where the long rolling windows live
    # (roll_med_2160 is a 90-day median). Trimming the panel on the way IN to get
    # the same memory saving is what starved those features: measured on Delhi,
    # a 10-day input window left roll_med_2160 28.66 ug/m3 wrong. Full history in,
    # one timestamp out.
    if only_last_ts and len(X):
        last_ts = X.ts.max()
        last_mask = (X.ts == last_ts).to_numpy()
        X = X[last_mask].reset_index(drop=True)
        keep_mask = last_mask if keep_mask is None else _compose_masks(keep_mask, last_mask)

    # ---- horizon-DEPENDENT block: replicate the base, vary 6 columns ----
    n = len(X)
    out = pd.concat([X] * len(horizons), ignore_index=True)
    out["horizon"] = np.repeat(np.asarray(horizons, dtype="int64"), n)
    tgt = out["ts"] + pd.to_timedelta(out["horizon"], unit="h")
    # Weather AT TARGET TIME, not issue time -- see the module docstring.
    # blh_m/temp_c via met_lookup (per-cell, one row per (cell, hour), so this
    # left-merge on both keys cannot duplicate or reorder `out`'s rows).
    # wind_scope="per_cell" (default): wind_ms/wind_from_deg already came
    # through the met_lookup merge itself (added to _met_cols_percell
    # above), consistent with blh_m/temp_c and with composite_grid/
    # positional_block's per-cell wind further up. A
    # target hour past the panel's own coverage (the training tail, or a live
    # run whose forecast fetch didn't reach this horizon) comes back NaN
    # either way, same as `y` already does near a panel's tail -- LightGBM
    # treats it as a native missing value, not a new failure mode.
    out["_met_ts"] = tgt
    out = out.merge(met_lookup, on=["cell", "_met_ts"], how="left")
    if wind_scope == "citywide":
        out["wind_from_deg"] = out["_met_ts"].map(_wind_dir_by_ts)
        out["wind_ms"] = out["_met_ts"].map(_wind_ms_by_ts) if _wind_ms_by_ts is not None else np.nan
    out = out.drop(columns="_met_ts")
    out["target_hour"] = tgt.dt.hour
    out["target_dow"] = tgt.dt.dayofweek
    out["target_month"] = tgt.dt.month
    # Finer than target_month, which can't separate early- from late-November
    # -- exactly when burning season turns. Deliberately NOT an absolute date:
    # that would let the model fit a year-over-year trend it can only
    # extrapolate wrongly at serve time.
    out["target_doy"] = tgt.dt.dayofyear
    # sin/cos pairs for every periodic quantity -- see _CYCLICAL's docstring
    # note above. Computed from the raw columns just assigned (and from
    # wind_from_deg, already in `out` via the met_lookup merge); the raw
    # columns themselves stay in `out` for readability/debugging, they are
    # just not in FEATURE_COLUMNS, so nothing downstream trains on them.
    for _col, _period in _CYCLICAL.items():
        _rad = 2 * np.pi * out[_col].astype(float) / _period
        out[f"{_col}_sin"] = np.sin(_rad)
        out[f"{_col}_cos"] = np.cos(_rad)
    # wind_u/wind_v: standard meteorological u/v (eastward/northward
    # components) from wind_from_deg ("blows FROM" bearing) and wind_ms,
    # both already in `out` via the met_lookup merge above -- see
    # _WIND_VECTOR_COLS' docstring note. Raw wind_ms/wind_from_deg stay in
    # `out` for readability/debugging, same convention as the cyclical
    # columns above; only wind_u/wind_v are in FEATURE_COLUMNS.
    _wind_from_rad = np.radians(out["wind_from_deg"].astype(float))
    out["wind_u"] = -out["wind_ms"].astype(float) * np.sin(_wind_from_rad)
    out["wind_v"] = -out["wind_ms"].astype(float) * np.cos(_wind_from_rad)
    if keep_mask is not None:
        out["y"] = np.concatenate(
            [g.pm25_station.shift(-h).to_numpy(dtype=float)[keep_mask] for h in horizons])
    else:
        out["y"] = np.concatenate([g.pm25_station.shift(-h).to_numpy(dtype=float) for h in horizons])
    out["clim_dow_hour"], out["clim_month"] = climatology_columns(out, clim_tables)
    out["city"] = out["city"].astype("category")

    if loso_exclude is not None:
        # The loso_exclude cell's own rows are the whole point of the
        # parameter: spatial-LOSO evaluates predictions for exactly this cell
        # against its real (masked-out) y. Dropping them whenever the
        # self-excluded composite comes back NaN would silently empty out the
        # one cell LOSO exists to test.
        is_excluded_cell = out.cell == loso_exclude
        has_signal = out[["lag_0", "lag_24"]].notna().all(axis=1)
        return out[is_excluded_cell | has_signal].reset_index(drop=True)
    return out.dropna(subset=["lag_0", "lag_24"]).reset_index(drop=True)


def _grid_to_rows(grid: np.ndarray, hours: pd.DatetimeIndex, cells: list[str],
                   X: pd.DataFrame, name: str) -> np.ndarray:
    """(n_hours, n_cells) composite grid -> one value per row of X, matched on
    (cell, ts)."""
    lookup = pd.DataFrame({
        "ts": hours.repeat(len(cells)),
        "cell": np.tile(np.asarray(cells), len(hours)),
        name: grid.ravel(),
    })
    return X[["cell", "ts"]].merge(lookup, on=["cell", "ts"], how="left")[name].to_numpy(dtype=float)


if __name__ == "__main__":
    from shared.grid import city_cells
    cells = city_cells()[:3]
    hours = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    rows = [{"cell": c, "ts": h, "ward_id": "W1", "ward_name": "Ward 1", "city": "bengaluru",
             "pm25_station": 50.0 if i == 0 else np.nan, "wind_from_deg": 90.0, "wind_ms": 2.0,
             "blh_m": 400.0, "temp_c": 27.0, "fires_6h": 0, "frp_6h": 0.0, "lu_industrial": 0,
             "lu_construction": 0, "lu_waste_burning": 0, "lu_traffic": 0, "lu_road": 1,
             "lu_sensitive": 0, "hour": h.hour, "dow": h.dayofweek}
            for i, c in enumerate(cells) for h in hours]
    demo = pd.DataFrame(rows)
    print(build_features(demo, horizons=[3]).shape)
