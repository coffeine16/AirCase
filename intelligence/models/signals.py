"""Robust signal conditioning — the statistical core of detection.

Three rules, all learned the hard way:

1. NEVER THE MEAN. A single spike hour inflates a mean and manufactures a
   "chronic source" that is really one fire. Every aggregate here is a median,
   and every spread is a MAD (median absolute deviation, scaled by 1.4826 to be
   comparable to a standard deviation on normal data). Robust to the outliers
   that ARE the phenomenon we are studying.

2. ONE WINDOW IS NOT A SIGNAL. A source is something that is still there when
   you zoom out. We aggregate over 24 h / 7 d / 30 d and require agreement:
     chronic  — elevated in the 7 d AND 30 d windows (a standing violator)
     emerging — elevated in 24 h/7 d but NOT 30 d (newly commissioned; act now)
     acute    — elevated right now, absent from every window (a fire; dispatch,
                but do not build a case file)
   Real-time alone is a noisy signal and cannot identify a source.

3. COMPARE A CELL TO ITS NEIGHBOURS, NOT TO THE CITY. The dense urban core is
   high everywhere; ranking against the city median just returns "the core is
   dirty", which is true, unactionable, and not a violator. Contrast against the
   surrounding ANNULUS instead: it asks "is this cell hot *for where it is*",
   which is the question an inspector actually has.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
import h3

from shared.config import DETECT_WINDOWS_H, CONTRAST_INNER_K, CONTRAST_OUTER_K
from shared.grid import circular_mean_deg

MAD_SCALE = 1.4826   # makes MAD comparable to sigma for normally-distributed data
EPS = 1e-9


@lru_cache(maxsize=None)
def _neighbourhood(cell: str, inner_k: int, lo: int, hi: int) -> tuple[tuple, tuple]:
    """(inner disk, outer annulus) cell ids. Cached: the contrast is evaluated
    once per pollutant per window over the same grid, so the ring geometry would
    otherwise be recomputed ~9x for nothing."""
    inner = tuple(h3.grid_disk(cell, inner_k))
    outer = tuple(c for r in range(lo, hi + 1) for c in h3.grid_ring(cell, r))
    return inner, outer


def robust_z(values: pd.Series | np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 * MAD). The outlier-resistant z-score.

    Falls back to a standard-deviation z only if MAD is exactly zero (a constant
    field), which otherwise divides by zero.
    """
    v = np.asarray(values, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    scale = MAD_SCALE * mad
    if scale < EPS:
        scale = np.nanstd(v)
    if scale < EPS:
        return np.zeros_like(v)
    return (v - med) / scale


def window_medians(df: pd.DataFrame, value_col: str, at: pd.Timestamp,
                   windows: dict[str, int] | None = None) -> pd.DataFrame:
    """Per-cell median of `value_col` over each trailing window ending at `at`.

    Returns one row per cell, one column per window (`w24h`, `w7d`, `w30d`), plus
    a per-window robust z against the citywide distribution of those medians.
    """
    windows = windows or DETECT_WINDOWS_H
    out = None
    for name, hours in windows.items():
        w = df[(df.ts > at - pd.Timedelta(hours=hours)) & (df.ts <= at)]
        med = w.groupby("cell")[value_col].median().rename(name)
        part = med.to_frame()
        part[f"z_{name}"] = robust_z(med.values)
        out = part if out is None else out.join(part, how="outer")
    return out.reset_index()


def neighbourhood_contrast(values: pd.Series, inner_k: int = CONTRAST_INNER_K,
                           outer_k: tuple[int, int] = CONTRAST_OUTER_K) -> pd.Series:
    """How far is each cell above its own surroundings, in robust z units?

    inner = the cell and its k<=inner_k disk (the candidate zone, ~1 km).
    outer = the annulus between outer_k[0] and outer_k[1] rings (~2-4 km).

    contrast = (median(inner) - median(outer)) / (1.4826 * MAD(outer))

    The MAD is taken over the OUTER ring, so the denominator is "how much does
    this neighbourhood normally vary" — a cell 3 MADs above a quiet suburb scores
    higher than one 3 ug/m3 above a churning industrial belt, which is right.
    """
    v = values.to_dict()
    lo, hi = outer_k
    scores = {}
    for cell in values.index:
        inner_c, outer_c = _neighbourhood(cell, inner_k, lo, hi)
        inner = [v[c] for c in inner_c if c in v]
        outer = [v[c] for c in outer_c if c in v]
        if len(inner) < 2 or len(outer) < 6:
            scores[cell] = 0.0                      # edge of the grid: no verdict
            continue
        o = np.asarray(outer, dtype=float)
        o_med = np.median(o)
        scale = MAD_SCALE * np.median(np.abs(o - o_med))
        if scale < EPS:
            scale = float(o.std())
        # Floor the denominator. A perfectly flat neighbourhood has MAD 0, and
        # dividing a real excess by that would either explode or (worse) get
        # silently clamped to a zero contrast, hiding the very spike we want.
        scale = max(scale, 0.05 * abs(o_med), EPS)
        scores[cell] = float((np.median(inner) - o_med) / scale)
    return pd.Series(scores, name="contrast")


def downwind_enhancement(daily: pd.DataFrame, wind: pd.DataFrame, cells: list[str],
                         value: str = "no2_col",
                         r_min: float = 2.0, r_max: float = 6.0,
                         half_angle: float = 40.0) -> pd.Series:
    """Wind-rotated plume detection: is this cell's DOWNWIND air dirtier than its
    CROSSWIND air?

    THE PROBLEM THIS SOLVES. Plain neighbourhood contrast fails on industry because
    the urban road network lifts NO2 across the entire core, so a factory cannot
    out-shout its own neighbourhood. Measured: 0/4 recall on the NO2-confounded tier.

    THE IDEA. A source's plume always points DOWNWIND. The diffuse background does
    not care which way the wind blows. So for each cell, on each day, compare the NO2
    in the downwind sector against the NO2 in the CROSSWIND sectors — same cell, same
    day, same distance band, same everything except direction:

        enhancement(cell, day) = median(NO2 downwind) - median(NO2 crosswind)

    A real point source produces a consistent positive enhancement, because the wind
    drags its plume into whichever sector is downwind THAT day. A dense-but-innocent
    district produces ~0, because it is dirty in every direction equally. Take the
    median over days and the background cancels while the plume accumulates.

    This is the standard technique in the TROPOMI point-source literature (wind
    rotation / oversampling, cf. Beirle et al.). It is not a trick — it is what the
    field does, and we already have every input: per-cell daily NO2, and hourly wind.

    Returns a robust z-score per cell. Cells near the grid edge (too few neighbours
    in a sector) return 0.0 rather than a guess.
    """
    n = len(cells)
    idx = {c: i for i, c in enumerate(cells)}
    pts = np.array([h3.cell_to_latlng(c) for c in cells])

    # pairwise distance + bearing, equirectangular (exact enough at city scale)
    lat0 = np.radians(pts[:, 0].mean())
    dy = (pts[None, :, 0] - pts[:, None, 0]) * 111.32
    dx = (pts[None, :, 1] - pts[:, None, 1]) * 111.32 * np.cos(lat0)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    bearing = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0   # from cell i to cell j
    ring = (dist >= r_min) & (dist <= r_max)

    # daily mean wind, on the circle
    w = wind.copy()
    w["date"] = pd.to_datetime(w.ts).dt.tz_localize(None).dt.normalize()
    daily_wind = w.groupby("date").wind_from_deg.apply(circular_mean_deg)

    d = daily.copy()
    d["date"] = pd.to_datetime(d.date).dt.tz_localize(None).dt.normalize() \
        if getattr(pd.to_datetime(d.date).dt, "tz", None) is not None \
        else pd.to_datetime(d.date).dt.normalize()

    per_day = []
    for date, g in d.groupby("date"):
        if date not in daily_wind.index:
            continue
        v = np.full(n, np.nan)
        for c, val in zip(g.cell, g[value]):
            if c in idx:
                v[idx[c]] = val
        if np.isnan(v).all():
            continue

        wind_to = (float(daily_wind[date]) + 180.0) % 360.0   # where the plume GOES
        off = np.abs(((bearing - wind_to + 180.0) % 360.0) - 180.0)
        down = ring & (off <= half_angle)
        cross = ring & (np.abs(off - 90.0) <= half_angle)

        md = np.nanmedian(np.where(down, v[None, :], np.nan), axis=1)
        mc = np.nanmedian(np.where(cross, v[None, :], np.nan), axis=1)
        per_day.append(md - mc)

    if not per_day:
        return pd.Series(0.0, index=cells, name="plume")

    enh = np.nanmedian(np.vstack(per_day), axis=0)     # median over days
    enh = np.nan_to_num(enh, nan=0.0)
    return pd.Series(robust_z(enh), index=cells, name="plume")


def classify_persistence(z24: float, z7: float, z30: float, thresh: float = 2.0) -> str:
    """chronic / emerging / acute — the enforcement-relevant distinction.

    A standing violator and a burning pile of rubbish need different responses:
    one gets a case file, the other gets a truck. The windows tell them apart.
    """
    hot24, hot7, hot30 = z24 >= thresh, z7 >= thresh, z30 >= thresh
    # The 30-day window ALONE establishes chronic. Requiring the 7-day window to
    # agree as well silently drops standing violators that happened to have a
    # quiet week — one week of unfavourable wind is not exoneration. (Measured:
    # this exact bug lost Bommasandra, z30d = 3.4, because its z7d was 1.7.)
    if hot30:
        return "chronic"        # standing violator: build the case file
    if hot7:
        return "emerging"       # sustained a week, absent over a month: act now
    if hot24:
        return "acute"          # loud today only: send a truck, not a notice
    return "none"
