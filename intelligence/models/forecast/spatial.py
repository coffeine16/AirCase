"""Real-station-grounded spatial primitives.

Every value here is a direct function of REAL station readings only — never
another cell's or another model's estimate. No propagation order, no error
accumulation. See docs/superpowers/specs/2026-08-15-forecast-rework-design.md
section 3.1 for the full reasoning, including why this uses a weighted MEAN
(a deliberate, documented exception to this project's "never the mean"
principle) and why there is no nearest-k truncation.
"""
import numpy as np
import pandas as pd

from shared.grid import cell_center, neighbors

DIST_DECAY_KM = 2.0   # same exp(-d/2) kernel as attribution.py::category_scores
FIRE_PRESSURE_RADIUS_KM = 6.0   # wider than detect.py's 1.5km direct-hit radius —
                                 # this feature is regional pressure, not "is this
                                 # exact cell burning" (that's the existing fires_6h)
FIRE_PRESSURE_WINDOW_H = 6

# A calm wind and a gale get the SAME distance decay below unless wind_ms is
# passed in and this scale is applied. decay_km(wind_ms) = DIST_DECAY_KM *
# (1 + WIND_DECAY_SCALE * wind_ms): a floor at DIST_DECAY_KM for calm air
# (turbulent diffusion still happens at wind_ms=0, so this never collapses
# to zero reach) plus a wind-proportional extension, not a literal
# advection-distance calculation (wind_speed * FIRE_PRESSURE_WINDOW_H would
# blow past FIRE_PRESSURE_RADIUS_KM at any real wind speed and say nothing
# about within-radius concentration gradient, which is what this decay
# actually represents).
#
# CRITICAL: this scale applies ONLY along the downwind axis, never
# isotropically. A first version of this fix stretched decay_km identically
# in every direction regardless of bearing -- physically backwards, since
# wind speed governs how far a DIRECTED transport process reaches, not how
# far influence spreads sideways or against the wind. That version was
# swept (seed-controlled, real Delhi data) and found NO reliable
# improvement over WIND_DECAY_SCALE=0 at any tested value 0.0-3.0: a fast
# wind was inflating the reach of crosswind and near-upwind sources too
# (everything short of the align term's 90-degree cutoff still got the
# isotropic stretch), injecting noise proportional to the very variable the
# feature was supposed to extract signal from. See composite_grid/
# fire_pressure below: distance is decomposed into a downwind component
# (stretched by wind_ms) and a crosswind component (fixed at DIST_DECAY_KM,
# never speed-scaled) -- an ellipse in wind-rotated coordinates, not a
# circle in raw ones. Matches the standard Gaussian-plume treatment, where
# wind speed enters the dilution/along-axis terms and never widens the
# crosswind spread.
#
# The exact constant is empirically swept against real held-out data, not
# asserted. Multiple sweeps were run before this value was trusted, and
# the process itself is worth recording: a Delhi-only, seed-only-pinned
# sweep first suggested 1.0 as a clean win at every horizon, but a
# separate-process rerun of the same scale gave materially different
# numbers -- LightGBM's histogram reduction order is not fully
# deterministic across thread counts even with a fixed seed (this
# project's own test suite documents this same caveat elsewhere), so
# "seed-controlled" alone was not sufficient for real reproducibility.
# The trustworthy version: seed AND thread count (num_threads=4) both
# pinned, POOLED across three cities (Delhi/Bengaluru/Chennai, matching
# the A1 finding's own lesson that single-city results can reverse), using
# the actual wind_u/wind_v feature set this model trains on. Result:
# scale=0.5 is a flat loss at every horizon; scale=1.0 costs skill only at
# the shortest, noisiest horizon (h3 -- unreliable in every test this
# project has run) and gains it back at every longer horizon (h9/h24/h48/
# h72, the ones the forecast-scheduling use case actually depends on).
# Real, modest, mostly-positive -- not a large effect, and not asserted as
# one.
WIND_DECAY_SCALE = 1.0


def composite_grid(query_lat: np.ndarray, query_lon: np.ndarray,
                    station_lat: np.ndarray, station_lon: np.ndarray,
                    station_val: np.ndarray, wind_from_deg: np.ndarray,
                    wind_ms: np.ndarray | None = None,
                    exclude: np.ndarray | None = None) -> np.ndarray:
    """Wind/distance-weighted MEAN of every real station's value, evaluated
    at every query point, for every timestamp — vectorised, no per-row
    Python loop (same discipline as panel.py::_fire_features).

    query_lat/query_lon: (n_q,) query point locations (a cell center, or a
        cell's k=1 neighbor center — same function either way).
    station_lat/station_lon: (n_s,) real station locations, fixed for a city.
    station_val: (n_t, n_s) per-timestamp station readings; NaN = no reading
        that hour (a station's own gaps are handled, not treated as zero).
    wind_from_deg: (n_t,) citywide-representative wind bearing, OR (n_t, n_q)
        per-query-point wind bearing -- columns aligned to query_lat/
        query_lon's order, letting each query point use its own local wind
        (weather varies by cell, see shared.grid.WEATHER_GRID_RES) instead
        of one city-average value. A (n_t,) array broadcasts across every
        query point exactly as before -- pure shape upgrade, no behaviour
        change for existing callers.
    wind_ms: optional (n_t,) or (n_t, n_q), same broadcasting rule as
        wind_from_deg above. None (default)
        reproduces the original fixed-DIST_DECAY_KM behaviour exactly, for
        every existing call site that hasn't been updated to pass it. When
        given, distance decay widens with wind speed ONLY along the
        downwind axis (see WIND_DECAY_SCALE) -- an ellipse in wind-rotated
        coordinates, not a circle stretched the same amount in every
        direction.
    exclude: optional (n_q, n_s) bool mask — True = this station must not
        contribute to this query point's composite. Used ONLY for
        spatial-LOSO's self-exclusion rule (spec 3.1) — a held-out station
        excluded from its OWN query point, still included in every other.

    Returns (n_t, n_q) ndarray. NaN where no station has any weight
    (e.g. every candidate station masked, or no stations passed in at all).
    """
    n_q, n_s = len(query_lat), len(station_lat)
    n_t = station_val.shape[0]
    if n_s == 0:
        return np.full((n_t, n_q), np.nan)

    # equirectangular approximation, exact enough at city scale — same
    # approach signals.py::downwind_enhancement already uses.
    lat0 = np.radians(np.concatenate([query_lat, station_lat]).mean())
    dy = (station_lat[None, :] - query_lat[:, None]) * 111.32          # (n_q, n_s)
    dx = (station_lon[None, :] - query_lon[:, None]) * 111.32 * np.cos(lat0)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    bearing_to_query = (np.degrees(np.arctan2(-dx, -dy)) + 360.0) % 360.0   # station -> query

    # (n_t,) -> (n_t, 1) so it broadcasts identically over every query point
    # (old behaviour, unchanged); (n_t, n_q) passes straight through, giving
    # each query point its own wind column.
    wind_from_deg = np.asarray(wind_from_deg)
    _wind_q = wind_from_deg if wind_from_deg.ndim == 2 else wind_from_deg[:, None]
    wind_to = (_wind_q + 180.0) % 360.0                                   # (n_t, n_q) direction wind blows TOWARD
    off = np.abs(((bearing_to_query[None, :, :] - wind_to[:, :, None] + 180.0) % 360.0) - 180.0)
    align = np.clip(np.cos(np.radians(off)), 0.0, None)                  # (n_t, n_q, n_s) -- unchanged by wind_ms

    if wind_ms is None:
        decay = np.exp(-dist / DIST_DECAY_KM)[None, :, :]              # (1, n_q, n_s), broadcasts over n_t
    else:
        # Decompose into wind-rotated coordinates: x_down is the signed
        # projection of the source->query displacement onto the downwind
        # axis, y_cross is the projection onto the perpendicular axis --
        # dist*cos(off)/dist*sin(off) ARE those projections, off already
        # being the angle between the displacement bearing and the
        # downwind axis. Speed stretches ONLY the downwind reach; the
        # crosswind reach stays at the fixed calm-air scale regardless of
        # wind speed, matching a Gaussian plume's along-wind vs
        # across-wind spread. At off=0 (dead downwind) this reduces to the
        # exact isotropic-stretch formula it replaces; off's cosine/sine
        # split into two decay axes is where it stops matching.
        off_rad = np.radians(off)
        x_down = dist[None, :, :] * np.cos(off_rad)
        y_cross = dist[None, :, :] * np.sin(off_rad)
        wind_ms = np.asarray(wind_ms)
        _wind_ms_q = wind_ms if wind_ms.ndim == 2 else wind_ms[:, None]  # (n_t, n_q) or (n_t, 1)
        along_km = (DIST_DECAY_KM * (1.0 + WIND_DECAY_SCALE * _wind_ms_q))[:, :, None]  # (n_t,n_q,1)
        r_eff = np.sqrt((x_down / along_km) ** 2 + (y_cross / DIST_DECAY_KM) ** 2)
        decay = np.exp(-r_eff)                                          # (n_t, n_q, n_s)

    weight = align * decay                                              # (n_t, n_q, n_s)
    if exclude is not None:
        weight = weight * (~exclude)[None, :, :]

    has_val = ~np.isnan(station_val)                                    # (n_t, n_s)
    weight = weight * has_val[:, None, :]
    vals = np.where(has_val[:, None, :], station_val[:, None, :], 0.0)

    total = weight.sum(axis=2)                                          # (n_t, n_q)
    num = (weight * vals).sum(axis=2)
    return np.divide(num, total, out=np.full_like(total, np.nan), where=total > 1e-9)


def positional_block(cell: str, station_lat: np.ndarray, station_lon: np.ndarray,
                      station_val: np.ndarray, wind_from_deg: np.ndarray,
                      wind_ms: np.ndarray | None = None,
                      exclude: np.ndarray | None = None) -> np.ndarray:
    """composite_grid evaluated at the cell's own center + its up-to-6 k=1
    neighbor centers, in one pass (spec section 3.2's positional block).
    Always returns 7 columns; padded with NaN if the cell has fewer than 6
    neighbors (H3 pentagon distortion — a global edge case, won't occur
    inside a city bbox, but handled rather than assumed away). `wind_ms`
    passes straight through to composite_grid — see its docstring."""
    nbrs = neighbors(cell, k=1)[:6]
    pts = [cell] + nbrs
    lat = np.array([cell_center(c)[0] for c in pts])
    lon = np.array([cell_center(c)[1] for c in pts])
    out = composite_grid(lat, lon, station_lat, station_lon, station_val,
                          wind_from_deg, wind_ms, exclude)
    if len(pts) < 7:
        pad = np.full((out.shape[0], 7 - len(pts)), np.nan)
        out = np.concatenate([out, pad], axis=1)
    return out


def fire_pressure(cells: list[str], fires: pd.DataFrame,
                   hours: pd.DatetimeIndex, wind_from_deg: np.ndarray,
                   wind_ms: np.ndarray | None = None,
                   fire_wind_from_deg: np.ndarray | None = None,
                   fire_wind_ms: np.ndarray | None = None) -> pd.DataFrame:
    """Regional fire-pressure composite per (cell, hour): distance AND
    wind-decay-weighted sum of REAL FIRMS detections within the trailing
    FIRE_PRESSURE_WINDOW_H hours, out to FIRE_PRESSURE_RADIUS_KM — "the
    same distance/wind weighting" spec section 3.2's Fire bullet requires,
    matching composite_grid's convention (bearing FROM the source TO the
    destination, compared against that hour's wind). Not circular — FIRMS
    is a raw observation, never a model's own output.

    `wind_from_deg`/`wind_ms`: one value per entry in `hours` — citywide
    REPRESENTATIVE wind, used as-is when the fire_wind_* args below are not
    given, and as the per-fire FALLBACK when they are given but NaN for a
    particular fire (e.g. that fire's location falls outside current
    weather-grid coverage).

    `fire_wind_from_deg`/`fire_wind_ms`: optional, one value per row of
    `fires` (SAME length and order as `fires` itself, before this function's
    own internal ts-filtering) — the wind AT EACH FIRE'S OWN LOCATION and
    hour, resolved by the caller (see features.py's build_features, which
    looks each fire's lat/lon up against the real weather grid). The fire is
    the physical anchor for "which way is this smoke plume going" — using
    wind sampled at the RECEIVING cell (or a citywide average) answers a
    related but different question. NaN entries (fire outside weather-grid
    coverage) fall back to the citywide `wind_from_deg`/`wind_ms` for THAT
    fire only, never silently propagate -- one fire's missing location data
    must not poison every cell's accumulated pressure for that hour.
    """
    n_c, n_h = len(cells), len(hours)
    spine_cell = np.tile(np.asarray(cells), n_h)
    spine_ts = pd.DatetimeIndex(hours).repeat(n_c)

    if fires.empty:
        return pd.DataFrame({"cell": spine_cell, "ts": spine_ts, "fire_pressure_regional": 0.0})

    f = fires.copy()
    f["ts"] = pd.to_datetime(f.ts, utc=True).dt.floor("h")
    _keep = f.ts.isin(hours).to_numpy()
    if fire_wind_from_deg is not None:
        fire_wind_from_deg = np.asarray(fire_wind_from_deg, dtype=float)[_keep]
    if fire_wind_ms is not None:
        fire_wind_ms = np.asarray(fire_wind_ms, dtype=float)[_keep]
    f = f[_keep]
    if f.empty:
        return pd.DataFrame({"cell": spine_cell, "ts": spine_ts, "fire_pressure_regional": 0.0})

    centers = np.array([cell_center(c) for c in cells])       # (n_c, 2)
    # to_numpy(dtype=float), not .values: an empty per-city fires.parquet
    # carries object-dtype lat/lon, and concatenating one into a pooled
    # multi-city table turns the WHOLE column object -- np.radians then
    # tries to call .radians() on each element and dies. Measured: kolkata
    # has 0 fire rows, which poisoned all 8 cities' lat/lon.
    flat = f.lat.to_numpy(dtype=float)                         # (n_f,)
    flon = f.lon.to_numpy(dtype=float)
    p1 = np.radians(centers[:, 0])[:, None]
    p2 = np.radians(flat)[None, :]
    dp = p2 - p1
    dl = np.radians(flon)[None, :] - np.radians(centers[:, 1])[:, None]
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    dist_km = 2 * 6371.0 * np.arcsin(np.sqrt(a))                # (n_c, n_f)
    within = dist_km <= FIRE_PRESSURE_RADIUS_KM

    # bearing FROM the fire TO the cell (fire is the source, cell is the
    # destination) -- same "source -> destination" convention composite_grid
    # uses for stations.
    dy = (centers[:, 0][:, None] - flat[None, :]) * 111.32              # (n_c, n_f)
    dx = (centers[:, 1][:, None] - flon[None, :]) * 111.32 * np.cos(p1)
    bearing_fire_to_cell = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

    hour_of = pd.Index(hours).get_indexer(f.ts)                # (n_f,) fire -> hour index
    # Fire's own-location wind when given, falling back to citywide-at-that-
    # hour PER FIRE where the per-location value is NaN (missing weather-grid
    # coverage for that fire's spot) -- np.where evaluates both branches, so
    # this never raises even when wind_from_deg[hour_of] is the only source.
    if fire_wind_from_deg is not None:
        wind_at_fire_dir = np.where(np.isnan(fire_wind_from_deg),
                                     wind_from_deg[hour_of], fire_wind_from_deg)
    else:
        wind_at_fire_dir = wind_from_deg[hour_of]
    wind_to_per_fire = (wind_at_fire_dir + 180.0) % 360.0        # (n_f,) wind at each fire's own hour
    off = np.abs(((bearing_fire_to_cell - wind_to_per_fire[None, :] + 180.0) % 360.0) - 180.0)
    align = np.clip(np.cos(np.radians(off)), 0.0, None)          # (n_c, n_f)

    if wind_ms is None:
        decay = np.exp(-dist_km / DIST_DECAY_KM)
    else:
        # Same wind-rotated decomposition as composite_grid: speed
        # stretches reach only along the downwind axis, never crosswind.
        off_rad = np.radians(off)
        x_down = dist_km * np.cos(off_rad)
        y_cross = dist_km * np.sin(off_rad)
        if fire_wind_ms is not None:
            wind_at_fire_ms = np.where(np.isnan(fire_wind_ms), wind_ms[hour_of], fire_wind_ms)
        else:
            wind_at_fire_ms = wind_ms[hour_of]
        along_km = (DIST_DECAY_KM * (1.0 + WIND_DECAY_SCALE * wind_at_fire_ms))[None, :]  # (1, n_f)
        r_eff = np.sqrt((x_down / along_km) ** 2 + (y_cross / DIST_DECAY_KM) ** 2)
        decay = np.exp(-r_eff)
    weight = np.where(within, decay * align, 0.0)

    pressure = np.zeros((n_h, n_c))
    for fi, hi in enumerate(hour_of):
        pressure[hi, :] += weight[:, fi] * float(f.frp.values[fi])

    csum = np.cumsum(pressure, axis=0)
    lag = np.zeros_like(csum)
    lag[FIRE_PRESSURE_WINDOW_H:] = csum[:-FIRE_PRESSURE_WINDOW_H]
    trailing = csum - lag

    return pd.DataFrame({"cell": spine_cell, "ts": spine_ts,
                          "fire_pressure_regional": trailing.ravel()})


if __name__ == "__main__":
    demo_lat = np.array([12.97])
    demo_lon = np.array([77.60])
    s_lat = np.array([12.97, 12.98])
    s_lon = np.array([77.59, 77.61])
    s_val = np.array([[40.0, 70.0]])
    wind = np.array([0.0])
    print("composite:", composite_grid(demo_lat, demo_lon, s_lat, s_lon, s_val, wind))
