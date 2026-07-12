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

# What the instruments can physically see. Not a tuning knob — a statement about
# what NO2/SO2/aerosol-index columns and thermal fire detection actually measure.
OBSERVABLE = {"industrial", "waste_burning"}


def main():
    hotspots = json.loads((DATA_OUT / "hotspots.json").read_text())
    attrs = {a["cell"]: a for a in json.loads((DATA_OUT / "attributions.json").read_text())}
    stations = pd.read_parquet(DATA_RAW / "stations.parquet")
    station_cells = stations.cell.unique()
    station_pts = [cell_center(c) for c in station_cells]

    hot_pts = [(h["cell"], *cell_center(h["cell"])) for h in hotspots]

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

    obs = df[df.observable]
    blind = df[~df.observable]
    print(f"RECALL, physically observable sources : {obs.detected.mean():.0%} "
          f"({obs.detected.sum()}/{len(obs)})")
    print(f"  ...and correctly named             : {obs.correct.mean():.0%} "
          f"({obs.correct.sum()}/{len(obs)})")
    print(f"  ...of which UNREGISTERED (on no map): "
          f"{obs[~obs.registered].detected.sum()}/{len(obs[~obs.registered])}")
    print(f"RECALL, sources with no tracer        : {blind.detected.mean():.0%} "
          f"({blind.detected.sum()}/{len(blind)})  <- known blind spot, not a bug")
    print()

    # Precision: how many flagged cells sit near a real source?
    src_pts = [(la, lo) for _n, _k, la, lo, *_ in SOURCES]
    near_real = sum(any(haversine_km(la, lo, sla, slo) <= NEAR_KM for sla, slo in src_pts)
                    for _c, la, lo in hot_pts)
    n_hot = len(hot_pts)
    print(f"PRECISION: {near_real}/{n_hot} flagged cells ({near_real / max(n_hot,1):.0%}) "
          f"lie within {NEAR_KM:.0f} km of a real source")
    print()
    print(f"STATION BASELINE: {df.near_a_station.sum()}/{len(df)} sources sit within "
          f"{STATION_KM:.0f} km of a monitor.")
    print("  ^ this is the coverage-bias number: a station-only dashboard cannot")
    print("    see what no station is standing next to.")


if __name__ == "__main__":
    main()
