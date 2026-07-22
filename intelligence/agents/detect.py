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

from shared.config import DATA_RAW, DATA_OUT, DETECT_WINDOWS_H, FIRE_FRAC_SCALE
from shared.grid import cell_center, haversine_km
from shared.wards import attach_wards
from intelligence.models.signals import neighbourhood_contrast, classify_persistence

# DETECTION CHANNELS — cut down to the two that survived contact with real data.
#
# We used to score on max(contrast) across NO2, SO2 and AAI. Measured on real S5P
# over both Bengaluru and Delhi, SO2 and AAI have NO resolvable spatial structure:
#
#     channel   SNR (spatial signal / noise surviving a 60-day median)
#     NO2       2.6 - 2.8     real
#     SO2       0.66 - 0.87   NOISE
#     AAI       0.76 - 1.03   NOISE
#
# (Real S5P SO2 over a city is 49% NEGATIVE with a MAD 30x its median. TROPOMI SO2
# is built for volcanoes and mega point-sources; an urban industrial cluster sits
# far under its noise floor. AAI at a 5.5 km footprint sees regional aerosol, not a
# landfill.) Taking the max across one signal and two noise fields means the max is
# usually noise: on real Delhi, 87% of 470 detected cells were driven by SO2 (63%)
# or AAI (24%) — i.e. we were manufacturing enforcement targets out of retrieval
# error, and a real burning landfill was ranked BELOW them.
#
# The synthetic world gave both channels clean signatures they do not have. This is
# the cost of an instrument model that was kinder than the instrument.
POLLUTANTS = ["no2_col"]
NOISE_CHANNELS = ["so2_col", "aai"]   # kept in the panel for exposure/evidence, not for detection
CONTRAST_THRESH = 2.0    # robust-z above the surrounding annulus to count as hot

# Enforceability. `road` is excluded on purpose: you cannot serve a notice on a
# road, so road proximity never makes a hotspot attributable.
#
# `traffic` was excluded for the SAME reason, after measuring that this rule was
# not discriminating at all: on real Delhi, Chennai and Bengaluru it marked
# 70/70, 157/157 and 51/51 hotspots enforceable. Never once diffuse. Two causes,
# both fixed here:
#
#   1. `traffic` (1,240 features in Delhi, 1,286 in Bengaluru) is a corridor or a
#      junction — you can no more serve a notice on it than on a road. Counting
#      it re-admitted the exact road network the NO2-alone clause below exists to
#      keep out, and it is what this project's own architecture doc calls the
#      textbook diffuse source. Delhi's hotspots move from a median 0.73 km to a
#      named candidate to 1.03 km once corridors stop counting.
#
#   2. 3 km was not a threshold, it was a tautology. In a real Indian city SOME
#      industrial or construction feature is always within 3 km — the FURTHEST
#      hotspot in any of the three cities was 2.87 km — so the test could only
#      ever return True and the fire clause was never consulted.
#
# 500 m is the honest replacement because it is a claim the instrument can
# support: the H3 res-8 cell is ~460 m across, so it means "the candidate is in
# this hexagon or the one touching it". That is the tightest spatial coincidence
# the grid can express, and beyond it we are asserting a link the data does not
# carry. It is a deliberately conservative bar for putting an inspector in a van.
ENFORCEABLE_KINDS = ["industrial", "construction", "waste_burning"]
ATTRIBUTABLE_KM = 0.5    # a candidate site this close is something to go inspect
POINT_TRACER_Z = 2.0     # SO2 / aerosol index: emitted by point sources, not by roads
ZONE_LINK_KM = 2.0       # hotspot cells this close belong to the same source zone

# Hard cap on how wide a zone may get. A zone exists to be DISPATCHED to — one
# inspector, one visit — so its size is bounded by what a team can cover, not by
# how far a plume happens to reach. Without this, single-linkage chaining fused
# real Chennai into 2 zones of 18.3 km and 11.1 km.
#
# 5 km is the bound Delhi's own zones already respect (2.5-5.2 km across 6 zones),
# so it formalises a size the working case had reached on its own rather than
# imposing a number picked to make Chennai look better.
MAX_ZONE_KM = 5.0


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
    z_by_window, comp = {}, {}
    for wname, hours in DETECT_WINDOWS_H.items():
        w = panel[(panel.ts > at - pd.Timedelta(hours=hours)) & (panel.ts <= at)]
        if w.empty:
            continue
        sigs = {}
        for p in POLLUTANTS:
            sigs[p] = neighbourhood_contrast(w.groupby("cell")[p].median())
        sigs["fire"] = w.groupby("cell").fires_6h.apply(lambda s: (s > 0).mean()) / FIRE_FRAC_SCALE

        frame = pd.concat(sigs, axis=1)
        z_by_window[f"z_{wname}"] = frame.max(axis=1)
        # Keep each instrument's own score (max across windows), not just the
        # collapsed max: deciding whether a hotspot is ENFORCEABLE needs to know
        # WHICH instrument fired. SO2 and fire point at a place you can drive to;
        # NO2 over a dense road network points at the whole city.
        for k, v in sigs.items():
            comp[k] = v if k not in comp else pd.concat([comp[k], v], axis=1).max(axis=1)

    out = pd.DataFrame(z_by_window)
    for k, v in comp.items():
        out[f"c_{k}"] = v
    return out.fillna(0.0)


def _mark_attributable(hot: pd.DataFrame) -> pd.DataFrame:
    """Is this hotspot an ENFORCEMENT target or a POLICY target?

    Not the same question as "is it polluted". A cell in the dense core with a
    high NO2 column and nothing else is real pollution with no one to serve a
    notice on — that is the diffuse urban background (traffic, cooking,
    resuspension), and it is a policy lever, not an inspection.

    A hotspot is ATTRIBUTABLE only if some instrument points at a PLACE:

      * a named OSM candidate within 3 km            -> a site you can visit
      * FIRMS fire persistence                       -> a location you can drive to
      * SO2 or aerosol-index contrast                -> point-source tracers

    Note the deliberate omission: an NO2 contrast alone does NOT make a hotspot
    attributable, because the road network raises NO2 across the whole core.

    And note why the OSM test cannot stand alone: our two best results — the
    landfill and the kiln — appear on NO map. "Not on the map" is what makes them
    interesting, not what makes them diffuse. Fire evidence is what localises
    them, which is exactly the case this rule exists to handle.
    """
    osm = pd.read_parquet(DATA_RAW / "osm.parquet")
    cand = osm[osm.kind.isin(ENFORCEABLE_KINDS)]
    pts = list(zip(cand.lat, cand.lon))

    near = []
    for cell in hot.cell:
        la, lo = cell_center(cell)
        d = min((haversine_km(la, lo, sla, slo) for sla, slo in pts), default=999.0)
        near.append(round(d, 2))
    hot["nearest_candidate_km"] = near

    # The SO2/AAI "point-source tracer" clauses are GONE. They were the theoretical
    # justification for calling a zone enforceable without a named candidate — and
    # they were noise. Keeping them would mean promoting retrieval error into an
    # inspection order.
    #
    # What survives: a named site you can visit, or a fire you can drive to.
    hot["attributable"] = (
        (hot.nearest_candidate_km <= ATTRIBUTABLE_KM)
        | (hot.c_fire > 0)
    )
    return hot


_ZONE_RANK = {"chronic": 3, "emerging": 2, "acute": 1}


def _reconcile_zones(hot: pd.DataFrame) -> pd.DataFrame:
    """Cluster adjacent hotspot cells into zones, and let the ZONE decide the kind.

    We classify cells, but the thing we are classifying is a SOURCE. A chronic
    source's fringe cells only go hot when the wind points at them, so over 30
    days they look intermittent and get labelled `emerging` — which would tell an
    inspector to go find a newly commissioned facility that does not exist.
    (Measured: both the `emerging` and `acute` flags in the previous run were
    fringe cells 2.0 km from the chronic landfill.)

    So: connect hotspot cells within ZONE_LINK_KM, then every cell in a zone
    inherits its most persistent member. A genuine new source, far from anything
    chronic, forms its own zone and keeps its `emerging` label.
    """
    cells = hot.cell.tolist()
    centres = {c: cell_center(c) for c in cells}

    parent = {c: c for c in cells}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Single-linkage with a DIAMETER CAP. Plain single-linkage chains: A links to
    # B, B to C, and the zone grows without limit as long as consecutive cells are
    # within ZONE_LINK_KM of *each other*. On real Chennai that fused 157 cells
    # into 2 zones spanning 18.3 km and 11.1 km — and an 18 km "inspection zone"
    # is not a place you can send a van, which is the only thing a zone is for.
    #
    # So a merge is refused when it would make the zone wider than MAX_ZONE_KM.
    # Pairs are considered nearest-first, so the tightest, most defensible links
    # form before the budget is spent on a straggler.
    members = {c: [c] for c in cells}          # root -> its cells, for the span test

    def span_if_merged(ra: str, rb: str) -> float:
        grp = members[ra] + members[rb]
        return max(haversine_km(*centres[x], *centres[y])
                   for i, x in enumerate(grp) for y in grp[i + 1:]) if len(grp) > 1 else 0.0

    pairs = []
    for i, a in enumerate(cells):
        la, lo = centres[a]
        for b in cells[i + 1:]:
            lb, lob = centres[b]
            d = haversine_km(la, lo, lb, lob)
            if d <= ZONE_LINK_KM:
                pairs.append((d, a, b))
    pairs.sort()                                # nearest first

    for _, a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if span_if_merged(ra, rb) > MAX_ZONE_KM:
            continue                            # would exceed a dispatchable area
        merged = members[ra] + members[rb]
        union(a, b)
        root = find(a)
        members[root] = merged                  # the surviving root owns the union

    hot["zone_id"] = [find(c) for c in hot.cell]
    zone_kind = (hot.assign(rank=hot.kind.map(_ZONE_RANK))
                    .groupby("zone_id")["rank"].max()
                    .map({v: k for k, v in _ZONE_RANK.items()}))
    hot["cell_kind"] = hot.kind                      # keep the raw per-cell verdict
    hot["kind"] = hot.zone_id.map(zone_kind)
    # a zone is enforceable if ANY of its cells has localisable evidence
    zone_attr = hot.groupby("zone_id").attributable.any()
    hot["attributable"] = hot.zone_id.map(zone_attr)
    hot["zone_id"] = ["Z%02d" % i for i in pd.factorize(hot.zone_id)[0]]
    return hot


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
    if len(hot):
        hot = _mark_attributable(hot)
        hot = _reconcile_zones(hot)

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

    cols = ["cell", "ward_id", "ward_name", "zone_id", "kind", "cell_kind",
            "attributable", "severity", "pm25_med", "nearest_candidate_km",
            "z_w24h", "z_w7d", "z_w30d", "fires_6h", "detection_basis"]
    out = hot[cols] if len(hot) else pd.DataFrame(columns=cols)
    payload = [{**r, "ts": str(at), "attributable": bool(r["attributable"]),
                **{k: round(float(r[k]), 2) for k in
                   ("pm25_med", "z_w24h", "z_w7d", "z_w30d", "nearest_candidate_km")}}
               for r in out.to_dict("records")]
    (DATA_OUT / "hotspots.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if len(out):
        n_enf = int(out.attributable.sum())
        zones = out.groupby("zone_id").kind.first().value_counts().to_dict()
        print(f"[detect] {at}: {len(out)} hotspot cells in {out.zone_id.nunique()} zones "
              f"{zones}")
        print(f"[detect]   {n_enf} cells ENFORCEABLE (a source to inspect); "
              f"{len(out) - n_enf} diffuse urban background (a policy target, not a notice)")
    else:
        print(f"[detect] {at}: no hotspots of {len(df)} cells")
    return out


if __name__ == "__main__":
    detect()
