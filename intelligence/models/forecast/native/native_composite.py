"""Numba-JIT reimplementation of spatial.py::composite_grid. Every
per-timestep quantity (align, decay, weight) is accumulated as a scalar
inside the loop instead of materialized as a full (n_t, n_q, n_s) NumPy
array -- that array is exactly what OOM'd a real 7-city pooled build on
this machine's 17GB (see
docs/superpowers/specs/2026-08-21-local-native-training-pipeline-design.md
section 1). dist/bearing_to_query never depend on time (only wind and
station values do), so they're precomputed once, matching the original
function's own broadcasting structure -- this is a direct translation of
that function's math, not a reimplementation from first principles, and
must stay byte-for-byte in sync with it (see the parity tests in this
package's test file, which are the actual proof this holds)."""
import numpy as np
from numba import njit, prange

from intelligence.models.forecast.spatial import DIST_DECAY_KM, WIND_DECAY_SCALE


@njit(cache=True, parallel=True)
def _composite_grid_kernel(query_lat, query_lon, station_lat, station_lon,
                            station_val, wind_from_deg_2d, wind_ms_2d,
                            exclude, has_wind_ms, has_exclude,
                            dist_decay_km, wind_decay_scale):
    n_t = station_val.shape[0]
    n_q = query_lat.shape[0]
    n_s = station_lat.shape[0]
    out = np.empty((n_t, n_q), dtype=np.float64)

    # lat0 = radians(mean(concat(query_lat, station_lat))) in the original --
    # written as a weighted sum/count here since Numba doesn't accelerate
    # np.concatenate well; mathematically identical to that mean.
    lat0 = np.radians((np.sum(query_lat) + np.sum(station_lat)) / (n_q + n_s))

    dist = np.empty((n_q, n_s), dtype=np.float64)
    bearing = np.empty((n_q, n_s), dtype=np.float64)
    for qi in range(n_q):
        for si in range(n_s):
            dy = (station_lat[si] - query_lat[qi]) * 111.32
            dx = (station_lon[si] - query_lon[qi]) * 111.32 * np.cos(lat0)
            dist[qi, si] = np.sqrt(dx * dx + dy * dy)
            bearing[qi, si] = (np.degrees(np.arctan2(-dx, -dy)) + 360.0) % 360.0

    for t in prange(n_t):
        for qi in range(n_q):
            total = 0.0
            num = 0.0
            wind_to = (wind_from_deg_2d[t, qi] + 180.0) % 360.0
            for si in range(n_s):
                if has_exclude and exclude[qi, si]:
                    continue
                v = station_val[t, si]
                if np.isnan(v):
                    continue
                off = abs(((bearing[qi, si] - wind_to + 180.0) % 360.0) - 180.0)
                align = np.cos(np.radians(off))
                if align < 0.0:
                    align = 0.0
                if has_wind_ms:
                    off_rad = np.radians(off)
                    x_down = dist[qi, si] * np.cos(off_rad)
                    y_cross = dist[qi, si] * np.sin(off_rad)
                    along_km = dist_decay_km * (1.0 + wind_decay_scale * wind_ms_2d[t, qi])
                    r_eff = np.sqrt((x_down / along_km) ** 2 + (y_cross / dist_decay_km) ** 2)
                    decay = np.exp(-r_eff)
                else:
                    decay = np.exp(-dist[qi, si] / dist_decay_km)
                w = align * decay
                total += w
                num += w * v
            if total > 1e-9:
                out[t, qi] = num / total
            else:
                out[t, qi] = np.nan
    return out


def composite_grid_native(query_lat, query_lon, station_lat, station_lon,
                           station_val, wind_from_deg, wind_ms=None,
                           exclude=None):
    """Drop-in, numerically-identical replacement for
    spatial.composite_grid -- identical signature, identical broadcasting
    rules for wind_from_deg/wind_ms (1-D (n_t,) or 2-D (n_t, n_q)),
    identical NaN/zero-weight handling. See that function's own docstring
    for the parameter contract this mirrors."""
    query_lat = np.asarray(query_lat, dtype=np.float64)
    query_lon = np.asarray(query_lon, dtype=np.float64)
    station_lat = np.asarray(station_lat, dtype=np.float64)
    station_lon = np.asarray(station_lon, dtype=np.float64)
    station_val = np.asarray(station_val, dtype=np.float64)
    n_q = len(query_lat)
    n_s = len(station_lat)
    n_t = station_val.shape[0]

    if n_s == 0:
        return np.full((n_t, n_q), np.nan)

    wind_from_deg = np.asarray(wind_from_deg, dtype=np.float64)
    wind_2d = (wind_from_deg if wind_from_deg.ndim == 2
               else np.broadcast_to(wind_from_deg[:, None], (n_t, n_q)).copy())

    has_wind_ms = wind_ms is not None
    if has_wind_ms:
        wind_ms_arr = np.asarray(wind_ms, dtype=np.float64)
        wind_ms_2d = (wind_ms_arr if wind_ms_arr.ndim == 2
                      else np.broadcast_to(wind_ms_arr[:, None], (n_t, n_q)).copy())
    else:
        wind_ms_2d = np.zeros((n_t, n_q), dtype=np.float64)

    has_exclude = exclude is not None
    exclude_arr = exclude if has_exclude else np.zeros((n_q, n_s), dtype=np.bool_)

    return _composite_grid_kernel(query_lat, query_lon, station_lat, station_lon,
                                   station_val, wind_2d, wind_ms_2d,
                                   exclude_arr, has_wind_ms, has_exclude,
                                   float(DIST_DECAY_KM), float(WIND_DECAY_SCALE))
