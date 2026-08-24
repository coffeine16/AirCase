"""Prediction log and residuals — what we said, what actually happened.

WHY THIS EXISTS
    The model is a generalist. It can say "this cell is usually bad around
    3am", but it is blind to its OWN recent error: nothing tells it that the
    last few forecasts for a cell ran 40 ug/m3 low, so it cannot latch onto a
    developing episode or self-correct between them. `lag_0` carries what the
    air did; nothing carries what WE did.

    Fixing that properly means an error-feedback feature and a retrain. This
    module is the prerequisite either way, because that feature has no data
    source until predictions are persisted: forecast.json is overwritten on
    every run, so by the time ground truth arrives the prediction it should be
    scored against is already gone.

    ledger.py already does exactly this shape, just narrowly — it freezes the
    +48h forecast for a DISPATCHED zone and compares it to the realized outcome
    later. This is the same idea for every cell and every horizon.

WHAT IT STORES
    One row per (issued_at, cell, horizon): the p50 we published, the p10/p90
    band, and — filled in on a later run, once the panel has caught up — the
    observed value at the target hour and the signed error.

    Parquet keyed by city, appended and de-duplicated on (issued_at, cell,
    horizon). Small: one issue x ~1700 cells x 24 horizons is ~40k rows.

WHAT IT DELIBERATELY DOES NOT DO
    It does not touch the served prediction. Nothing here changes what the
    product outputs today — a bias correction built on these residuals is a
    separate, deliberate decision, and it should be measured against
    persistence before it ships, exactly like the model itself was.
"""
from __future__ import annotations

import pandas as pd

# Rows older than this are dropped on write. A residual's use is recent error;
# two weeks covers the longest horizon (72h) several times over, and keeps the
# file from growing without bound on an hourly refresh.
RETAIN_DAYS = 14

_SCHEMA = ["issued_at", "cell", "horizon_h", "pm25_p50", "pm25_p10", "pm25_p90",
           "target_ts", "observed", "error"]


def _path(city_out):
    return city_out / "forecast_residuals.parquet"


def log_predictions(city_out, field: list[dict], issued_at: pd.Timestamp) -> int:
    """Append this run's predictions. Returns rows written.

    `issued_at` is the last OBSERVED hour the forecast was made from, not
    wall-clock: that is what makes target_ts computable and what a later run
    joins ground truth against.
    """
    if not field:
        return 0
    df = pd.DataFrame([{
        "issued_at": issued_at,
        "cell": r["cell"],
        "horizon_h": int(r["horizon_h"]),
        "pm25_p50": float(r["pm25_hat"]),
        "pm25_p10": float(r.get("pm25_p10", float("nan"))),
        "pm25_p90": float(r.get("pm25_p90", float("nan"))),
    } for r in field])
    df["target_ts"] = df.issued_at + pd.to_timedelta(df.horizon_h, unit="h")
    df["observed"] = float("nan")
    df["error"] = float("nan")

    p = _path(city_out)
    if p.exists():
        try:
            prev = pd.read_parquet(p)
            df = pd.concat([prev, df], ignore_index=True)
        except Exception as e:      # noqa: BLE001 — a corrupt log must not stop serving
            print(f"[residuals] {city_out.name}: existing log unreadable "
                  f"({type(e).__name__}) — starting a new one")
    # Last write wins for a repeated (issue, cell, horizon): re-running the same
    # hour should replace its row, not double it.
    df = df.drop_duplicates(["issued_at", "cell", "horizon_h"], keep="last")
    cutoff = df.issued_at.max() - pd.Timedelta(days=RETAIN_DAYS)
    df = df[df.issued_at >= cutoff]
    city_out.mkdir(parents=True, exist_ok=True)
    df[_SCHEMA].to_parquet(p, index=False)
    return len(df)


def score_against(city_out, panel: pd.DataFrame) -> dict:
    """Fill `observed`/`error` for any logged prediction whose target hour the
    panel now covers, and report what that says.

    Called with the CURRENT panel, so a prediction made 3 days ago for a target
    2 days ago gets scored the next time the pipeline runs.
    """
    p = _path(city_out)
    if not p.exists():
        return {"scored": 0, "note": "no prediction log yet"}
    df = pd.read_parquet(p)
    truth = (panel[panel.pm25_station.notna()][["cell", "ts", "pm25_station"]]
             .rename(columns={"ts": "target_ts", "pm25_station": "obs"})
             .drop_duplicates(["cell", "target_ts"]))
    merged = df.merge(truth, on=["cell", "target_ts"], how="left")
    fill = merged.obs.notna()
    merged.loc[fill, "observed"] = merged.loc[fill, "obs"]
    merged.loc[fill, "error"] = merged.loc[fill, "pm25_p50"] - merged.loc[fill, "obs"]
    merged = merged.drop(columns=["obs"])
    merged[_SCHEMA].to_parquet(p, index=False)

    done = merged[merged.error.notna()]
    if done.empty:
        return {"scored": 0, "logged": len(merged),
                "note": "nothing has reached its target hour yet"}
    # MAD, not mean absolute error, and median bias rather than mean: the same
    # robust-statistics rule the rest of this project runs on.
    return {
        "scored": int(len(done)),
        "logged": int(len(merged)),
        "median_bias": round(float(done.error.median()), 2),
        "median_abs_error": round(float(done.error.abs().median()), 2),
        "by_horizon": {int(h): round(float(g.error.median()), 2)
                       for h, g in done.groupby("horizon_h") if len(g) >= 20},
    }
