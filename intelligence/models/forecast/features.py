# intelligence/models/forecast/features.py
"""Builds the full training/prediction feature frame (spec section 3.2).
Station cells use their own real history; non-station cells (and a
spatial-LOSO held-out station's own rows) use the real-stations-only
composite from spatial.py at every lag depth — not just the current hour."""
import numpy as np
import pandas as pd

from intelligence.models.forecast.spatial import composite_grid, positional_block, fire_pressure
from intelligence.models.forecast.climatology import build_climatology, lookup_climatology
from shared.grid import cell_center, haversine_km

LAGS = [0, 1, 3, 6, 24]
_MET = ["blh_m", "wind_ms", "wind_from_deg", "temp_c"]
_STATIC = ["lu_road", "lu_industrial", "lu_traffic"]

FEATURE_COLUMNS = (
    [f"lag_{k}" for k in LAGS] + ["roll_med_24", "roll_med_168",
    "has_station", "distance_to_nearest_station_km", "nearby_stations_delta"]
    + [f"pos_{i}" for i in range(7)] + ["fire_pressure_regional", "fires_6h", "frp_6h"]
    + _MET + _STATIC + ["clim_dow_hour", "clim_month",
    "target_hour", "target_dow", "target_month", "horizon", "city"]
)


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


def build_features(panel: pd.DataFrame, horizons: list[int],
                    loso_exclude: str | None = None,
                    fires: pd.DataFrame | None = None) -> pd.DataFrame:
    # reset_index is required, not cosmetic: the fire_pressure merge below
    # produces a fresh 0..n-1 RangeIndex on X, and later code re-joins X
    # against THIS frame (p) by index label (p.loc[row.Index, "ward_id"]).
    # Any caller that passes a sliced or otherwise non-0-based panel (e.g.
    # panel[panel.cell != some_cell], as Task 9's spatial-LOSO does) would
    # silently break that alignment and raise KeyError deep inside the
    # climatology lookup -- found the hard way once already; fixed at the
    # source so no future caller has to remember this precondition.
    p = panel.sort_values(["cell", "ts"]).reset_index(drop=True).copy()
    if fires is None:
        fires = pd.DataFrame(columns=["ts", "lat", "lon", "frp", "confidence"])
    station_lat, station_lon, station_val, station_cells = _station_matrix(p)
    hours = pd.DatetimeIndex(sorted(p.ts.unique()))
    hour_idx = {h: i for i, h in enumerate(hours)}
    wind_by_hour = p.drop_duplicates("ts").set_index("ts")["wind_from_deg"].reindex(hours).values

    clim_tables = build_climatology(p)

    cells = sorted(p.cell.unique())
    cell_centers = {c: cell_center(c) for c in cells}
    nearest_station_km = {}
    for c in cells:
        if len(station_cells) == 0:
            nearest_station_km[c] = float("nan")
            continue
        lat, lon = cell_centers[c]
        nearest_station_km[c] = min(haversine_km(lat, lon, cell_centers.get(s, cell_center(s))[0],
                                                   cell_centers.get(s, cell_center(s))[1])
                                     for s in station_cells)

    frames = []
    for horizon in horizons:
        g = p.groupby("cell", group_keys=False)
        feats = {}
        for k in LAGS:
            feats[f"lag_{k}"] = g.pm25_station.shift(k)
        feats["roll_med_24"] = g.pm25_station.transform(lambda s: s.shift(1).rolling(24, min_periods=6).median())
        feats["roll_med_168"] = g.pm25_station.transform(lambda s: s.shift(1).rolling(168, min_periods=24).median())
        for m in _MET + _STATIC:
            if m in p:
                feats[m] = p[m]
        X = pd.DataFrame(feats, index=p.index)
        X["cell"] = p.cell
        X["ts"] = p.ts
        X["city"] = p.city
        X["has_station"] = p.cell.isin(station_cells)
        X["distance_to_nearest_station_km"] = p.cell.map(nearest_station_km).fillna(0.0)
        X.loc[X.has_station, "distance_to_nearest_station_km"] = 0.0

        # roll_med_24/168 are computed from the RAW per-cell groupby above,
        # which is correct for every station cell and already NaN for every
        # non-station cell (no history to roll over) -- except the ONE case
        # that matters: the loso_exclude cell IS a station, so its rolling
        # medians are its own real history unless explicitly nulled here.
        # Leaving them in place would let a "held-out" station see the
        # median of its own last 24/168 real hours -- exactly the leak
        # spec 3.1's self-exclusion rule exists to prevent, just via a
        # feature the composite-based lag fill below never touches.
        if loso_exclude is not None:
            X.loc[X.cell == loso_exclude, ["roll_med_24", "roll_med_168"]] = np.nan

        # composite-based fill for non-station cells (and a LOSO station's own rows)
        if len(station_cells) > 0:
            excl = None
            if loso_exclude is not None and loso_exclude in station_cells:
                s_idx = station_cells.index(loso_exclude)
                excl = np.zeros((len(cells), len(station_cells)), dtype=bool)
                cell_idx = {c: i for i, c in enumerate(cells)}
                excl[cell_idx[loso_exclude], s_idx] = True

            for k in LAGS:
                shifted_val = np.roll(station_val, k, axis=0)
                shifted_val[:k, :] = np.nan
                comp = composite_grid(np.array([cell_centers[c][0] for c in cells]),
                                       np.array([cell_centers[c][1] for c in cells]),
                                       station_lat, station_lon, shifted_val, wind_by_hour,
                                       exclude=excl)
                comp_df = pd.DataFrame(comp, index=hours, columns=cells).stack().rename(f"_comp_{k}")
                comp_lookup = comp_df.reset_index()
                comp_lookup.columns = ["ts", "cell", f"_comp_{k}"]
                merged = X[["cell", "ts"]].merge(comp_lookup, on=["cell", "ts"], how="left")[f"_comp_{k}"]
                needs_fill = (~X.has_station.values) | (X.cell == loso_exclude).values
                X.loc[needs_fill, f"lag_{k}"] = merged[needs_fill].values

            comp_now = composite_grid(np.array([cell_centers[c][0] for c in cells]),
                                       np.array([cell_centers[c][1] for c in cells]),
                                       station_lat, station_lon, station_val, wind_by_hour, exclude=excl)
            comp_now_df = pd.DataFrame(comp_now, index=hours, columns=cells).stack().rename("_comp_now")
            comp_now_lookup = comp_now_df.reset_index()
            comp_now_lookup.columns = ["ts", "cell", "_comp_now"]
            comp_now_merged = X[["cell", "ts"]].merge(comp_now_lookup, on=["cell", "ts"], how="left")["_comp_now"]
            X["nearby_stations_delta"] = comp_now_merged.values - X["lag_0"].values

            for i in range(7):
                X[f"pos_{i}"] = np.nan
            for c in cells:
                pb = positional_block(c, station_lat, station_lon, station_val, wind_by_hour,
                                       exclude=(excl[[cells.index(c)]] if excl is not None else None))
                pb_hours = hours
                mask = X.cell == c
                for i in range(7):
                    series = pd.Series(pb[:, i], index=pb_hours)
                    X.loc[mask, f"pos_{i}"] = X.loc[mask, "ts"].map(series).values
        else:
            for k in LAGS:
                pass  # no stations at all: lags stay NaN, LightGBM treats as missing
            X["nearby_stations_delta"] = np.nan
            for i in range(7):
                X[f"pos_{i}"] = np.nan

        fp = fire_pressure(cells, fires, hours, wind_by_hour)
        X = X.merge(fp, on=["cell", "ts"], how="left")
        if "fires_6h" in p:
            X["fires_6h"] = p.fires_6h
            X["frp_6h"] = p.frp_6h

        tgt = X["ts"] + pd.Timedelta(hours=horizon)
        X["target_hour"] = tgt.dt.hour
        X["target_dow"] = tgt.dt.dayofweek
        X["target_month"] = tgt.dt.month
        X["horizon"] = horizon

        # For the loso_exclude cell's own rows, skip the "cell" scope and let
        # lookup_climatology fall through to ward/city -- exactly what a
        # genuinely non-station cell would see. climatology.py has no
        # exclude parameter at all, so the "cell" scope for loso_exclude is
        # that station's own real history -- passing a cell value that can
        # never match the table's index (None) is what forces the
        # fallback, without touching climatology.py or discarding the
        # ward/city signal the way a blanket NaN would.
        clim_cell = [None if row.cell == loso_exclude else row.cell for row in X.itertuples()]
        X["clim_dow_hour"] = [
            lookup_climatology(clim_tables, cc, p.loc[row.Index, "ward_id"], row.city, t, "dow_hour")
            for cc, row, t in zip(clim_cell, X.itertuples(), tgt)
        ]
        X["clim_month"] = [
            lookup_climatology(clim_tables, cc, p.loc[row.Index, "ward_id"], row.city, t, "month")
            for cc, row, t in zip(clim_cell, X.itertuples(), tgt)
        ]

        X["y"] = g.pm25_station.shift(-horizon)
        frames.append(X)

    out = pd.concat(frames, ignore_index=True)
    out["city"] = out["city"].astype("category")
    if loso_exclude is not None:
        # The loso_exclude cell's own rows are the whole point of the
        # loso_exclude parameter: Task 9's spatial-LOSO harness evaluates
        # predictions for exactly this cell against its real (masked-out)
        # y. Dropping them here whenever the self-excluded composite comes
        # back NaN (which happens whenever too few other stations remain
        # to compose from — not just in this test's single-station panel)
        # would silently empty out the one cell LOSO exists to test. Every
        # other cell keeps the ordinary warm-up filter; NaN features on the
        # loso_exclude cell reach the model as native missing values, same
        # as everywhere else in this codebase (see climatology.py).
        is_excluded_cell = out.cell == loso_exclude
        has_signal = out[["lag_0", "lag_24"]].notna().all(axis=1)
        return out[is_excluded_cell | has_signal].reset_index(drop=True)
    return out.dropna(subset=["lag_0", "lag_24"])


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
