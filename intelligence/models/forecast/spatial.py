"""Real-station-grounded spatial primitives.

Every value here is a direct function of REAL station readings only — never
another cell's or another model's estimate. No propagation order, no error
accumulation. See docs/superpowers/specs/2026-08-15-forecast-rework-design.md
section 3.1 for the full reasoning, including why this uses a weighted MEAN
(a deliberate, documented exception to this project's "never the mean"
principle) and why there is no nearest-k truncation.
"""
import numpy as np

DIST_DECAY_KM = 2.0   # same exp(-d/2) kernel as attribution.py::category_scores


def composite_grid(query_lat: np.ndarray, query_lon: np.ndarray,
                    station_lat: np.ndarray, station_lon: np.ndarray,
                    station_val: np.ndarray, wind_from_deg: np.ndarray,
                    exclude: np.ndarray | None = None) -> np.ndarray:
    """Wind/distance-weighted MEAN of every real station's value, evaluated
    at every query point, for every timestamp — vectorised, no per-row
    Python loop (same discipline as panel.py::_fire_features).

    query_lat/query_lon: (n_q,) query point locations (a cell center, or a
        cell's k=1 neighbor center — same function either way).
    station_lat/station_lon: (n_s,) real station locations, fixed for a city.
    station_val: (n_t, n_s) per-timestamp station readings; NaN = no reading
        that hour (a station's own gaps are handled, not treated as zero).
    wind_from_deg: (n_t,) per-timestamp wind bearing (citywide single point,
        matching the operational panel's existing convention).
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

    decay = np.exp(-dist / DIST_DECAY_KM)                               # (n_q, n_s), static per city

    wind_to = (wind_from_deg + 180.0) % 360.0                            # (n_t,) direction wind blows TOWARD
    off = np.abs(((bearing_to_query[None, :, :] - wind_to[:, None, None] + 180.0) % 360.0) - 180.0)
    align = np.clip(np.sin(np.radians(off)), 0.0, None)                  # (n_t, n_q, n_s)

    weight = align * decay[None, :, :]                                  # (n_t, n_q, n_s)
    if exclude is not None:
        weight = weight * (~exclude)[None, :, :]

    has_val = ~np.isnan(station_val)                                    # (n_t, n_s)
    weight = weight * has_val[:, None, :]
    vals = np.where(has_val[:, None, :], station_val[:, None, :], 0.0)

    total = weight.sum(axis=2)                                          # (n_t, n_q)
    num = (weight * vals).sum(axis=2)
    return np.divide(num, total, out=np.full_like(total, np.nan), where=total > 1e-9)


if __name__ == "__main__":
    demo_lat = np.array([12.97])
    demo_lon = np.array([77.60])
    s_lat = np.array([12.97, 12.98])
    s_lon = np.array([77.59, 77.61])
    s_val = np.array([[40.0, 70.0]])
    wind = np.array([0.0])
    print("composite:", composite_grid(demo_lat, demo_lon, s_lat, s_lon, s_val, wind))
