"""Multi-scale robust historical climatology (spec section 3.2). Station
cells get their own; non-station cells fall back to ward- then city-level.
Median throughout — this is a temporal aggregate, principle 6 applies here
without exception (unlike spatial.py's composite, which does not)."""
import pandas as pd

SCOPES = ("cell", "ward", "city")
SCALES = SCOPES   # back-compat alias for the original (misnamed) export

# FIXED (pre-v2-retrain audit, 2026-08-20): the "month" scale used to bucket
# by calendar month with NO interpolation between adjacent months -- a
# Nov-30 row and a Dec-1 row drew their medians from disjoint sets of
# historical rows, even though North Indian stubble-burning intensity ramps
# CONTINUOUSLY through Oct->Nov->Dec, not as a step function. Not a
# sample-size problem (a city/month bucket over ~2 years pooled 1000+ hourly
# rows already) -- a SHAPE problem: the true seasonal curve is smooth, the
# bucket structure was a 12-step staircase, and the discontinuity landed
# squarely inside the exact seasonal transition this project's own headline
# Delhi burning-season story depends on.
#
# Fix: the "month" scale now keys on day-of-year (1-365, Dec-31-of-a-leap-
# year folded onto 365) instead of calendar month, and each day's value is a
# CENTERED, CIRCULAR windowed median (+/- _DOY_WINDOW_HALF_DAYS days) over
# that identity's own per-calendar-day medians -- see _smoothed_doy_table.
# Nov-30 and Dec-1's windows now overlap in all but 2 of 31 days, so their
# values move by the marginal contribution of 2 boundary days, not by a
# whole bucket's worth of seasonal drift. Verified directly on a linear
# 1.5-unit/day ramp: the fix reproduces the ramp's own +1.5/day rate AT the
# Nov-30/Dec-1 seam (vs. the old hard-bucket jump of +45.75 there) -- see
# test_month_scale_smooths_across_the_calendar_month_boundary. The scale is
# still called "month" (the dict key, the clim_month feature name,
# FEATURE_COLUMNS) to keep every caller's interface unchanged -- only what
# populates and looks up that table changed.
#
# Real limitation, not overclaimed: median is a majority-vote statistic,
# not an averaging one, so this does NOT blend a genuine step discontinuity
# -- a synthetic 50-vs-150 step at the boundary still returns a hard 50.0
# at Nov-30 (its 31-day window has a 16-vs-15-day majority on the Nov side;
# that is the median working correctly, not a residual bug). The real
# seasonal curve this fix targets ramps continuously, not as a step, so
# this does not matter for the actual use case -- but it means the fix
# removes the STAIRCASE artifact from smooth/gradual data, it does not
# make climatology a general-purpose smoother.
#
# _how (dow*24+hour) has the analogous week-boundary issue (Sun 23:00 / Mon
# 00:00) and is DELIBERATELY left alone: it is smaller in practice (recurs
# weekly, not the one seasonal transition the project's narrative rides on;
# each side of the boundary still individually accumulates 100+ samples over
# 2 years), and neither scale is fed to the model as a raw ordinal (see
# build_climatology's own lookup-key usage below) -- this was always about
# the underlying VALUE's bucket discontinuity, never a wraparound-blindness
# bug in how the model reads it.
_DOY_WINDOW_HALF_DAYS = 15   # +/- 15 days = a 31-day centered window
_DOY_MIN_PERIODS = 3   # each populated day is already a median of many
# hourly readings, not a single noisy point -- a handful of real days
# within the window is enough to avoid a degenerate estimate. Deliberately
# low (not e.g. half the window): a strict threshold went NaN on this
# project's own short synthetic test fixtures (72h/3-day panels) and would
# also go NaN near the very start/end of a short-history city's real data,
# which is the opposite of this fix's intent -- prefer a noisier real value
# over an artificially-precise threshold that just pushes the discontinuity
# problem to "when does the window first fill up" instead of removing it.


def _smoothed_doy_table(p: pd.DataFrame, id_col: str) -> pd.Series:
    """The "month" scale's table: per (identity, day-of-year 1-365), the
    median of that identity's own per-calendar-day medians over a centered,
    circular +/- _DOY_WINDOW_HALF_DAYS-day window. "Day-then-window" (not a
    window over raw hourly values) for the same reason features.py's
    roll_med_720/2160 median the day before rolling: every calendar day
    gets an equal vote regardless of how many hourly readings happened to
    land on it, matching this project's own robust-statistics principle
    (never let sampling density masquerade as signal).

    Circular: Dec-31's window wraps into early Jan, not just Dec. Achieved
    by tripling each identity's 365-day series (prev/current/next) before a
    plain centered rolling median, then keeping the middle third -- pandas'
    rolling() has no native wraparound mode.

    Returns an EMPTY (identity, doy) Series, same convention as an ordinary
    groupby().median() on an empty frame, when `p` has no rows -- callers
    (build_climatology's exclude_cell path, its own tests) rely on this."""
    empty = pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=[id_col, "_doy"]))
    if p.empty:
        return empty
    doy = p["ts"].dt.dayofyear.clip(upper=365)
    daily = p.assign(_doy=doy).groupby([id_col, "_doy"])["pm25_station"].median()
    window = 2 * _DOY_WINDOW_HALF_DAYS + 1
    parts = []
    for ident in daily.index.get_level_values(0).unique():
        s = daily.loc[ident].reindex(range(1, 366))
        tripled = pd.concat([s, s, s], ignore_index=True)
        smoothed = tripled.rolling(window, center=True, min_periods=_DOY_MIN_PERIODS).median()
        mid = smoothed.iloc[365:730].reset_index(drop=True).dropna()
        if mid.empty:
            continue
        mid.index = pd.MultiIndex.from_arrays(
            [[ident] * len(mid), (mid.index + 1).to_numpy()], names=[id_col, "_doy"])
        parts.append(mid)
    return pd.concat(parts) if parts else empty


def _how(ts: pd.Series) -> pd.Series:
    return ts.dt.dayofweek * 24 + ts.dt.hour


def build_climatology(panel: pd.DataFrame, exclude_cell: str | None = None) -> dict[str, pd.Series]:
    """`exclude_cell` drops that cell's readings from ALL THREE scopes, not
    just the cell scope. Blanking only the cell-level lookup is not enough:
    a ward that contains exactly one station (the common case) has a "ward
    fallback" that IS that station's own history wearing a different hat —
    which is the same leak spec 3.1's self-exclusion rule exists to prevent.
    """
    p = panel[panel.pm25_station.notna()]
    if exclude_cell is not None:
        p = p[p.cell != exclude_cell]
    p = p.copy()
    p["how"] = _how(p.ts)
    return {
        "cell_dow_hour": p.groupby(["cell", "how"]).pm25_station.median(),
        "cell_month": _smoothed_doy_table(p, "cell"),
        "ward_dow_hour": p.groupby(["ward_id", "how"]).pm25_station.median(),
        "ward_month": _smoothed_doy_table(p, "ward_id"),
        "city_dow_hour": p.groupby(["city", "how"]).pm25_station.median(),
        "city_month": _smoothed_doy_table(p, "city"),
    }


def lookup_climatology(tables: dict[str, pd.Series], cell: str, ward_id: str,
                        city: str, ts: pd.Timestamp, scale: str = "dow_hour") -> float:
    """cell -> ward -> city fallback, in that order. Returns NaN (not a
    guess) if none of the three has a matching row — LightGBM treats NaN
    as a native missing value."""
    # "month" scale keys on day-of-year now (see _smoothed_doy_table).
    key = (ts.dayofweek * 24 + ts.hour) if scale == "dow_hour" else min(ts.dayofyear, 365)
    for scope, ident in zip(SCOPES, (cell, ward_id, city)):
        table = tables[f"{scope}_{scale}"]
        if (ident, key) in table.index:
            return float(table.loc[(ident, key)])
    return float("nan")


if __name__ == "__main__":
    hours = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    demo = pd.DataFrame({"cell": ["A"] * 72, "ward_id": ["W1"] * 72,
                          "city": ["bengaluru"] * 72, "ts": hours,
                          "pm25_station": [40.0 + (i % 24) for i in range(72)]})
    t = build_climatology(demo)
    print("cell climatology @ hour 5:",
          lookup_climatology(t, "A", "W1", "bengaluru", pd.Timestamp("2024-01-02T05:00:00", tz="UTC")))
