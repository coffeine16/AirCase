"""Hotspot / zone detection — Node 1 of the agent pipeline.

WHY THIS RUNS ON THE SATELLITE, NOT ON THE FUSION FIELD
------------------------------------------------------
The fusion field is trained on station cells, and CPCB siting norms put stations
AWAY from sources. Measured on our own synthetic world, the 12 training stations
see a mean source contribution of 0.25 ug/m3 (p99 = 6.7) while the rest of the
city reaches 210. A model fit on those labels never observes a source, cannot
learn a source response, and (being a tree ensemble) cannot extrapolate one
either. Its field is background-dominated by construction.

So the fusion field's job here is EXPOSURE ("how much PM2.5 is a person in this
cell breathing"), which it does well and honestly. It is NOT the detector.

Detection runs on the SATELLITE, the only layer with genuinely uniform coverage:
every cell, no siting bias. That is where the coverage-debiasing claim actually
lives, and there it is real.

HOW
---
For each trailing window (24 h / 7 d / 30 d) we take the per-cell MEDIAN of each
satellite pollutant — never the mean, which one spike hour would inflate into a
fake chronic source — and score each cell by NEIGHBOURHOOD CONTRAST: how far it
sits above the annulus around it, in robust (MAD) units. Contrast, not a citywide
rank, because the dense urban core is high everywhere and "this district is
dense" is true, unactionable, and not a violator.

Agreement across windows then separates the three things an administrator must
respond to differently:

    chronic  — elevated over 7 d AND 30 d   -> a standing violator; build a case
    emerging — elevated over 7 d, not 30 d  -> newly commissioned; act now
    acute    — elevated in 24 h only, or an active fire -> send a truck today

Output: data/outputs/hotspots.json
"""
import json

import numpy as np
import pandas as pd

from shared.config import DATA_OUT, DETECT_WINDOWS_H, FIRE_FRAC_SCALE
from shared.wards import attach_wards
from intelligence.models.signals import neighbourhood_contrast, classify_persistence

POLLUTANTS = ["no2_col", "so2_col", "aai"]
CONTRAST_THRESH = 2.0    # robust-z above the surrounding annulus to count as hot
CITY_EPISODE_Z = 2.0     # citywide meteorological episode, reported separately


def _zone_scores(panel: pd.DataFrame, at: pd.Timestamp) -> pd.DataFrame:
    """Per-cell detection score per window: the stronger of two instruments.

    SATELLITE CONTRAST — how far the cell's window-median column sits above the
    annulus around it, in robust units. Sees industry (SO2/NO2) and, weakly,
    smoke (aerosol index).

    FIRE PERSISTENCE — what fraction of the window had a FIRMS detection within
    1.5 km. FIRMS observes burning DIRECTLY; it needs no inference and no
    contrast. Crucially this is evaluated PER WINDOW, so a landfill that burns
    every night for two months reads as chronic rather than as sixty unrelated
    acute events — which is the difference between a case file and a wild goose
    chase.

    Known blind spot, stated rather than hidden: construction dust is coarse PM
    with NO satellite tracer (S5P measures NO2/SO2/CO/aerosol index, none of
    which fingerprint it) and it does not burn. Neither instrument can see it.
    Construction sources are therefore undetectable from these data alone; they
    need the OSM permit layer and citizen reports, not this detector.
    """
    out = None
    for wname, hours in DETECT_WINDOWS_H.items():
        w = panel[(panel.ts > at - pd.Timedelta(hours=hours)) & (panel.ts <= at)]
        if w.empty:
            continue
        per_signal = []
        for p in POLLUTANTS:
            med = w.groupby("cell")[p].median()
            per_signal.append(neighbourhood_contrast(med).rename(p))

        fire_frac = w.groupby("cell").fires_6h.apply(lambda s: (s > 0).mean())
        per_signal.append((fire_frac / FIRE_FRAC_SCALE).rename("fire"))

        z = pd.concat(per_signal, axis=1).max(axis=1).rename(f"z_{wname}")
        out = z.to_frame() if out is None else out.join(z, how="outer")
    return out.fillna(0.0)


def detect(at: pd.Timestamp | None = None) -> pd.DataFrame:
    panel = pd.read_parquet(DATA_OUT / "panel.parquet")
    field = pd.read_parquet(DATA_OUT / "fusion_field.parquet")
    panel["ts"] = pd.to_datetime(panel.ts, utc=True)
    field["ts"] = pd.to_datetime(field.ts, utc=True)
    at = at or panel.ts.max()

    z = _zone_scores(panel, at)

    # Exposure (from the fusion field) and fire activity, both robust aggregates.
    wk = field[(field.ts > at - pd.Timedelta(days=7)) & (field.ts <= at)]
    pm_med = wk.groupby("cell").pm25_hat.median().rename("pm25_med")
    now = panel[panel.ts == at].set_index("cell")
    fires_now = now.fires_6h.reindex(z.index).fillna(0).astype(int).rename("fires_6h")

    df = z.join(pm_med).join(fires_now).reset_index().rename(columns={"index": "cell"})
    df["pm25_med"] = df.pm25_med.fillna(df.pm25_med.median())

    # ---- classify ----
    df["kind"] = [classify_persistence(r.z_w24h, r.z_w7d, r.z_w30d, CONTRAST_THRESH)
                  for r in df.itertuples()]
    hot = df[df.kind != "none"].copy()

    # ---- severity: deterministic, [0, 1] ----
    # Blend of (a) how far above its neighbourhood, (b) what people actually
    # breathe there, (c) how durable the signal is. No LLM anywhere near this.
    persistence_weight = {"chronic": 1.0, "emerging": 0.6, "acute": 0.35}
    if len(hot):
        zmax = hot[["z_w24h", "z_w7d", "z_w30d"]].max(axis=1)
        hot["severity"] = (
            0.45 * np.clip(zmax / 6.0, 0, 1)
            + 0.35 * np.clip(hot.pm25_med / 120.0, 0, 1)
            + 0.20 * hot.kind.map(persistence_weight)
        ).round(3)
        hot["detection_basis"] = [
            f"satellite contrast z={zm:.1f} vs surrounding 4-8 km"
            + (f"; {int(f)} FIRMS detections in last 6 h" if f else "")
            for zm, f in zip(zmax, hot.fires_6h)]
        hot = hot.sort_values("severity", ascending=False)
        hot = attach_wards(hot)

    # ---- citywide episode, reported separately from local hotspots ----
    hourly = field[field.ts == at].pm25_hat
    week_med = float(field[(field.ts > at - pd.Timedelta(days=7))].pm25_hat.median())
    city_now = float(hourly.median())
    if week_med > 0 and city_now / week_med > 1.25:
        print(f"[detect] citywide episode: city median {city_now:.0f} ug/m3 vs "
              f"7-day median {week_med:.0f} (meteorology-driven; not a local hotspot)")

    cols = ["cell", "ward_id", "ward_name", "kind", "severity", "pm25_med",
            "z_w24h", "z_w7d", "z_w30d", "fires_6h", "detection_basis"]
    out = hot[cols] if len(hot) else pd.DataFrame(columns=cols)
    payload = [{**r, "ts": str(at),
                **{k: round(float(r[k]), 2) for k in
                   ("pm25_med", "z_w24h", "z_w7d", "z_w30d")}}
               for r in out.to_dict("records")]
    (DATA_OUT / "hotspots.json").write_text(json.dumps(payload, indent=2))

    counts = out.kind.value_counts().to_dict() if len(out) else {}
    print(f"[detect] {at}: {len(out)} hotspots of {len(df)} cells  {counts}")
    return out


if __name__ == "__main__":
    detect()
