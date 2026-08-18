# intelligence/models/forecast/features.py
"""Builds the full training/prediction feature frame (spec section 3.2).
Station cells use their own real history; non-station cells (and a
spatial-LOSO held-out station's own rows) use the real-stations-only
composite from spatial.py at every lag depth — not just the current hour.

STRUCTURE: the expensive work (composites, positional blocks, lags, rolling
medians, fire pressure) does NOT depend on the forecast horizon, so it is
computed ONCE per (cell, ts) and then replicated across horizons. Only
`horizon`, the target clock columns, the climatology lookup (keyed on the
TARGET time) and `y` vary with the horizon. Doing the whole build inside the
horizon loop made one call ~24x more expensive than it needs to be — with 24
horizons on a 1210-cell x 60-day panel that is the difference between a
pipeline stage that finishes and one that does not. The deleted single-file
forecast.py hit exactly this wall (88 s of a 101 s Cloud Run request) and
solved it the same way.
"""
import numpy as np
import pandas as pd

from intelligence.models.forecast.spatial import composite_grid, positional_block, fire_pressure
from intelligence.models.forecast.climatology import build_climatology, SCOPES
from shared.grid import cell_center, haversine_km

LAGS = [0, 1, 3, 6, 24]
_MET = ["blh_m", "wind_ms", "wind_from_deg", "temp_c"]
_STATIC = ["lu_road", "lu_industrial", "lu_traffic"]

FEATURE_COLUMNS = (
    [f"lag_{k}" for k in LAGS] + ["roll_med_24", "roll_med_168",
    "has_station", "distance_to_nearest_station_km", "nearby_stations_delta"]
    + [f"pos_{i}" for i in range(7)] + ["fire_pressure_regional", "fires_6h", "frp_6h"]
    + _MET + _STATIC + ["clim_dow_hour", "clim_month",
    "target_hour", "target_dow", "target_month", "target_doy", "horizon", "city"]
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
        "month": tgt.dt.month.astype("int64").values,
    })
    out = []
    for scale, key_col in (("dow_hour", "how"), ("month", "month")):
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
        out.append(np.full(len(keys), np.nan) if vals is None else vals.to_numpy(dtype=float))
    return out[0], out[1]


def attach_climatology(frame: pd.DataFrame, tables: dict[str, pd.Series]) -> pd.DataFrame:
    """Returns a COPY of `frame` with clim_dow_hour/clim_month recomputed from
    `tables`. Exists so a caller with a time-based train/test split can rebuild
    the climatology from TRAIN-ONLY data and re-attach it, without paying for a
    second full build_features pass (see train.py's walk-forward loop)."""
    out = frame.copy()
    out["clim_dow_hour"], out["clim_month"] = climatology_columns(out, tables)
    return out


def build_features(panel: pd.DataFrame, horizons: list[int],
                    loso_exclude: str | None = None,
                    fires: pd.DataFrame | None = None,
                    clim_tables: dict[str, pd.Series] | None = None,
                    restrict_to_station_cells: bool = False) -> pd.DataFrame:
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
    every cell, station or not."""
    # reset_index is required, not cosmetic: the fire_pressure merge below
    # produces a fresh 0..n-1 RangeIndex on X, and later code re-joins X
    # against THIS frame (p) positionally. Any caller that passes a sliced
    # or otherwise non-0-based panel would silently break that alignment.
    p = panel.sort_values(["cell", "ts"]).reset_index(drop=True).copy()
    if fires is None:
        fires = pd.DataFrame(columns=["ts", "lat", "lon", "frp", "confidence"])
    station_lat, station_lon, station_val, station_cells = _station_matrix(p)
    hours = pd.DatetimeIndex(sorted(p.ts.unique()))
    wind_by_hour = p.drop_duplicates("ts").set_index("ts")["wind_from_deg"].reindex(hours).values

    if clim_tables is None:
        clim_tables = build_climatology(p, exclude_cell=loso_exclude)

    cells = sorted(p.cell.unique())
    cell_centers = {c: cell_center(c) for c in cells}
    for s in station_cells:
        cell_centers.setdefault(s, cell_center(s))

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
    for m in _MET + _STATIC:
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

    # roll_med_24/168 are computed from the RAW per-cell groupby above,
    # which is correct for every station cell and already NaN for every
    # non-station cell (no history to roll over) -- except the ONE case
    # that matters: the loso_exclude cell IS a station, so its rolling
    # medians are its own real history unless explicitly nulled here.
    if loso_exclude is not None:
        X.loc[X.cell == loso_exclude, ["roll_med_24", "roll_med_168"]] = np.nan

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
                                   shifted_val, wind_by_hour, exclude=excl)
            merged = _grid_to_rows(comp, hours, cells, X, f"_comp_{k}")
            X.loc[needs_fill, f"lag_{k}"] = merged[needs_fill]

        comp_now = composite_grid(q_lat, q_lon, station_lat, station_lon,
                                   station_val, wind_by_hour, exclude=excl)
        X["nearby_stations_delta"] = _grid_to_rows(comp_now, hours, cells, X, "_comp_now") - X["lag_0"].values

        # ONE merge for all 7 positional columns instead of 7 masked
        # assignments per cell (that inner loop was O(n_cells) full-frame
        # boolean writes -- 8k passes over a 1.7M-row frame on a real panel).
        pos = np.full((len(hours), len(cells), 7), np.nan)
        for ci, c in enumerate(cells):
            pos[:, ci, :] = positional_block(
                c, station_lat, station_lon, station_val, wind_by_hour,
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

    fp = fire_pressure(cells, fires, hours, wind_by_hour)
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

    # ---- horizon-DEPENDENT block: replicate the base, vary 6 columns ----
    n = len(X)
    out = pd.concat([X] * len(horizons), ignore_index=True)
    out["horizon"] = np.repeat(np.asarray(horizons, dtype="int64"), n)
    tgt = out["ts"] + pd.to_timedelta(out["horizon"], unit="h")
    out["target_hour"] = tgt.dt.hour
    out["target_dow"] = tgt.dt.dayofweek
    out["target_month"] = tgt.dt.month
    # Finer than target_month, which can't separate early- from late-November
    # -- exactly when burning season turns. Deliberately NOT an absolute date:
    # that would let the model fit a year-over-year trend it can only
    # extrapolate wrongly at serve time.
    out["target_doy"] = tgt.dt.dayofyear
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
