"""The headline evaluation: does the detector find the sources, and would the
station network have found them?

This replaces "does the fusion field recover hotspots" as the money stat, because
we now know it cannot, and we know why (see intelligence/agents/detect.py). The
honest question is not whether one model beats another on RMSE — it is:

    Of the real sources polluting this city, how many does the platform find,
    how many does a station-based dashboard find, and how many of the platform's
    accusations are wrong?

Three numbers, scored against the synthetic world's hidden source list:

  RECALL     — of the true sources, how many did we detect? Split by whether the
               source is physically OBSERVABLE with these instruments at all
               (industry -> SO2/NO2 columns; burning -> FIRMS) versus not
               (construction dust is coarse PM with no satellite tracer; traffic
               NO2 is confounded with the urban background). Claiming credit for
               finding what we cannot see would be dishonest; so would hiding the
               blind spot.
  PRECISION  — of the cells we flagged, how many sit near a real source? A false
               accusation costs an inspector a wasted trip and the platform its
               credibility.
  STATION BASELINE — how many sources sit near enough to a monitor for a
               station-only dashboard to have noticed? This is the coverage-bias
               number the whole platform exists to produce.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from shared.config import DATA_RAW, DATA_OUT
from shared.grid import cell_center, haversine_km
from ingestion.synthetic import SOURCES

NEAR_KM = 2.0        # a detection "finds" a source if it lands within this radius
STATION_KM = 2.0     # a station could plausibly notice a source within this radius

# WHAT THE INSTRUMENTS CAN ACTUALLY SEE. Not a tuning knob — a statement about what
# FIRMS thermal detection and the S5P NO2 column physically measure, revised after
# running on real data (see intelligence/agents/detect.py::POLLUTANTS).
#
# This used to say {industrial, waste_burning}, because our synthetic satellite gave
# SO2 a clean industrial signature. Real S5P SO2 over a city is NOISE — 49% negative,
# SNR 0.7. So industry lost its fingerprint, and the honest tiering is now:
TIERS = {
    # FIRMS sees burning DIRECTLY. No inference, no contrast, no confound.
    "direct (thermal fire)": {"waste_burning"},
    # NO2 is a real combustion tracer, but the road network raises it across the
    # whole urban core, so a point source must out-shout its own neighbourhood.
    # Hard, not impossible: on real Delhi this tier DID surface industrial zones.
    "NO2, confounded":       {"industrial", "traffic"},
    # Coarse PM. No satellite tracer of any kind, and it does not burn.
    "no tracer at all":      {"construction"},
}
OBSERVABLE = TIERS["direct (thermal fire)"]


def main():
    hotspots = json.loads((DATA_OUT / "hotspots.json").read_text())
    attrs = {a["cell"]: a for a in json.loads((DATA_OUT / "attributions.json").read_text())}
    stations = pd.read_parquet(DATA_RAW / "stations.parquet")
    station_cells = stations.cell.unique()
    station_pts = [cell_center(c) for c in station_cells]

    hot_pts = [(h["cell"], *cell_center(h["cell"])) for h in hotspots]
    by_cell = {h["cell"]: h for h in hotspots}

    rows = []
    for name, kind, lat, lon, strength, _act, registered, live_from in SOURCES:
        hits = [(c, haversine_km(lat, lon, la, lo)) for c, la, lo in hot_pts
                if haversine_km(lat, lon, la, lo) <= NEAR_KM]
        hits.sort(key=lambda x: x[1])
        found = bool(hits)
        named = attrs.get(hits[0][0], {}).get("primary_source") if found else None
        seen_by_station = any(haversine_km(lat, lon, sla, slo) <= STATION_KM
                              for sla, slo in station_pts)
        rows.append({"source": name, "kind": kind, "registered": registered,
                     "observable": kind in OBSERVABLE, "detected": found,
                     "attributed_as": named, "correct": named == kind if found else False,
                     "near_a_station": seen_by_station})
    df = pd.DataFrame(rows)

    print("=" * 78)
    print(f"{'SOURCE':<28}{'TYPE':<15}{'MAPPED':<8}{'VISIBLE':<9}{'FOUND':<7}{'NAMED AS'}")
    print("-" * 78)
    for r in df.itertuples():
        print(f"{r.source[:27]:<28}{r.kind:<15}{'yes' if r.registered else 'NO':<8}"
              f"{'yes' if r.observable else 'no':<9}{'YES' if r.detected else '-':<7}"
              f"{r.attributed_as or '-'}")
    print("=" * 78)

    print("RECALL, by what the instruments can physically see:")
    for tier, kinds in TIERS.items():
        t = df[df.kind.isin(kinds)]
        if t.empty:
            continue
        named = t.correct.sum()
        print(f"  {tier:<24} {t.detected.sum()}/{len(t)} found, "
              f"{named}/{len(t)} correctly named")
    obs = df[df.observable]
    if len(obs):
        unreg = obs[~obs.registered]
        print(f"  ...of the directly-observable ones, UNREGISTERED (on no map): "
              f"{unreg.detected.sum()}/{len(unreg)}")
    print()

    # Precision, split by what we are actually claiming about each cell.
    #
    # An ENFORCEABLE hotspot is an accusation: we are saying "go inspect here".
    # If it is not near a real source, that is a wasted inspector trip and a
    # false accusation, and it is the number that must be honest.
    #
    # A DIFFUSE hotspot claims nothing of the kind. It says "this area is dirty
    # and no single actor is responsible" — the urban background. Scoring it
    # against a point-source list is a category error: a correctly-identified
    # diffuse zone has no source to be near, and counting it as a false positive
    # penalises the platform for being right.
    src_pts = [(la, lo) for _n, _k, la, lo, *_ in SOURCES]

    def near_a_source(la, lo):
        return any(haversine_km(la, lo, sla, slo) <= NEAR_KM for sla, slo in src_pts)

    # PRECISION IS A ZONE-LEVEL QUESTION, NOT A CELL-LEVEL ONE.
    #
    # You dispatch an inspector to a zone, not to a 460 m hexagon. And a real
    # source NECESSARILY lights up a 2-3 km blob: the satellite footprint is
    # ~1.6 km and the plume advects. So scoring individual cells against a
    # "within 2 km of the source" test penalises the detector for correctly
    # mapping the extent of a plume it correctly found. Measured: the cells that
    # fail the cell-level test are a median 2.5 km from the source, and they sit
    # inside the SAME zones as that source.
    #
    # The honest question is: of the zones we send someone to, how many contain a
    # real polluter?
    zones = {}
    for c, la, lo in hot_pts:
        h = by_cell[c]
        z = zones.setdefault(h.get("zone_id", c),
                             {"cells": 0, "attributable": False, "min_dist": 1e9})
        z["cells"] += 1
        z["attributable"] |= bool(h.get("attributable"))
        z["min_dist"] = min(z["min_dist"],
                            min(haversine_km(la, lo, sla, slo) for sla, slo in src_pts))

    enf_z = {k: v for k, v in zones.items() if v["attributable"]}
    dif_z = {k: v for k, v in zones.items() if not v["attributable"]}
    enf_hit = sum(v["min_dist"] <= NEAR_KM for v in enf_z.values())

    print(f"PRECISION — enforceable ZONES (each one is an inspector dispatched):")
    print(f"  {enf_hit}/{len(enf_z)} ({enf_hit / max(len(enf_z), 1):.0%}) contain a real source")
    for k, v in sorted(enf_z.items()):
        verdict = "REAL" if v["min_dist"] <= NEAR_KM else "no source within 2 km"
        print(f"    {k}: {v['cells']:>2} cells, nearest source {v['min_dist']:.2f} km  [{verdict}]")
    print(f"DIFFUSE zones (no locatable source — a policy target, not a notice):")
    print(f"  {len(dif_z)} zone(s), excluded from the enforcement queue")
    print()

    cell_hit = sum(near_a_source(la, lo) for c, la, lo in hot_pts
                   if by_cell[c].get("attributable"))
    n_enf_cells = sum(1 for c, _, _ in hot_pts if by_cell[c].get("attributable"))
    print(f"  (cell-level, for reference: {cell_hit}/{n_enf_cells} "
          f"({cell_hit / max(n_enf_cells,1):.0%}) of enforceable CELLS are within "
          f"{NEAR_KM:.0f} km of a source —")
    print(f"   the shortfall is plume extent inside correctly-found zones, not "
          f"false accusations.)")
    print()
    print(f"STATION SEPARATION: {df.near_a_station.sum()}/{len(df)} sources sit within "
          f"{STATION_KM:.0f} km of a monitor.")
    print("  !! THIS IS AN ASSUMPTION OF THE WORLD MODEL, NOT A FINDING. !!")
    print("  pick_station_cells() excludes the source cell + its k=2 ring, an effective")
    print("  floor of ~1.9-2.4 km, so '0 within 2 km' is true ~99% of the time BY")
    print("  CONSTRUCTION (1 of 1039 candidate cells survives inside 2 km; P(a station")
    print("  lands there) = 1.1%). It restates CPCB siting norms, which is why we model")
    print("  it — but a synthetic world cannot DISCOVER its own placement rule.")
    print("  For the empirical version, run scripts/eval_coverage_bias.py against real")
    print("  monitors and real OSM industry. That number is checkable; this one is not.")
    print()
    print("  What is NOT assumed: the recall above. Detection reads satellite + FIRMS and")
    print("  never touches a station, so it is invariant to where the monitors sit —")
    print("  see scripts/eval_station_sensitivity.py, which re-runs it with the exclusion")
    print("  switched off entirely.")


if __name__ == "__main__":
    main()
