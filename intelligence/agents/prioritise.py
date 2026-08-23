"""Enforcement Priority Score & Dispatch Agent.

Calculates the deterministic Enforcement Priority Score (EPS) for each enforceable
zone, and routes inspection teams using a spatial-clustering + road-distance
algorithm that keeps each team's stops geographically compact.
"""
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any

import sys
from pathlib import Path
# Add project root to path so `shared` can be imported when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.config import DATA_OUT, BBOX
from shared.grid import cell_center, haversine_km
from shared.routing import route_waypoints
from intelligence.agents.detect import ATTRIBUTABLE_KM

COVER_RADIUS_KM = 2.0


def calculate_eps(zone_cells: List[Dict], attributions: Dict[str, Dict], panel: pd.DataFrame) -> Dict[str, Any]:
    """Calculate the deterministic EPS for a given zone.
    
    EPS = 100 * (0.35*severity + 0.25*attribution_conf + 0.20*actionability + 0.20*vulnerability)
    """
    # Max severity over the zone's cells
    severity = max(cell.get("severity", 0.0) for cell in zone_cells)
    
    # Kind and locatability
    kind = zone_cells[0].get("kind", "chronic")
    kind_weight = {"acute": 1.0, "emerging": 0.8, "chronic": 0.6}.get(kind, 0.6)
    
    # Reuse detect.py's OWN attributability distance rather than a second,
    # independent number: detect.py measured that 3km was a tautology on real
    # cities (every hotspot in 3 real cities sat within 3km of SOME candidate,
    # so the test never discriminated) and moved to 0.5km. A zone only reaches
    # this function at all if it's already `attributable` at that threshold, so
    # re-testing against a stale 3.0km here silently drifted back to the number
    # detect.py explicitly rejected — always reading 1.0 for distance-attributed
    # zones and hiding the real 0.5km/fire-only distinction.
    nearest_candidate_km = min(cell.get("nearest_candidate_km", 99.0) for cell in zone_cells)
    locatable = 1.0 if nearest_candidate_km <= ATTRIBUTABLE_KM else 0.6
    
    attributable = zone_cells[0].get("attributable", True)
    base = 1.0 if attributable else 0.0
    actionability = base * kind_weight * locatable
    
    # Attribution confidence
    cell_ids = [c["cell"] for c in zone_cells]
    confidences = [attributions.get(cid, {}).get("confidence", 0.0) for cid in cell_ids]
    attribution_conf = max(confidences) if confidences else 0.0
    
    # Vulnerability (max over the zone's cells from panel.parquet)
    # count of schools/hospitals within 1.5 km (lu_sensitive)
    if not panel.empty and "lu_sensitive" in panel.columns:
        zone_panel = panel[panel.cell.isin(cell_ids)]
        if not zone_panel.empty:
            max_sensitive = zone_panel["lu_sensitive"].max()
            # City-relative, not a fixed count of 5. Bengaluru, Delhi and Chennai
            # carry very different school/hospital densities per 1.5km — the same
            # "constant tuned once" problem as detect.py's old pm25 denominator.
            # p95 across every cell in THIS city's panel is "how crowded with
            # sensitive sites does it get here", so a zone scores high for
            # standing out in its own city, not against an arbitrary count.
            city_ref = float(panel["lu_sensitive"].quantile(0.95))
            city_ref = max(city_ref, 1.0)
            vulnerability = min(max_sensitive / city_ref, 1.0)
        else:
            vulnerability = 0.0
    else:
        vulnerability = 0.0
        
    # Not available: forecast
    FORECAST_WEIGHT = 0.0
    forecast_delta = 0.0
    
    severity_term = max(0.0, min(severity + FORECAST_WEIGHT * forecast_delta, 1.0))
    
    eps = 100.0 * (0.35 * severity_term + 
                   0.25 * attribution_conf + 
                   0.20 * actionability + 
                   0.20 * vulnerability)
    
    components = {
        "severity": round(severity_term, 3),
        "attribution_conf": round(attribution_conf, 3),
        "actionability": round(actionability, 3),
        "vulnerability": round(vulnerability, 3)
    }
    
    weighted_sum = (0.35 * components["severity"] + 
                    0.25 * components["attribution_conf"] + 
                    0.20 * components["actionability"] + 
                    0.20 * components["vulnerability"])
    assert math.isclose(weighted_sum * 100, eps, abs_tol=1.0), "EPS components do not sum to EPS"    
    return {"eps": round(eps, 1), "components": components}


def _cluster_stops(stops: List[Dict], n_teams: int) -> List[List[Dict]]:
    """Assign stops to teams via spatial k-means so each team's stops are
    geographically compact — no cross-city zigzag routes.

    When there are fewer stops than teams, each stop becomes its own cluster.
    """
    if not stops:
        return [[] for _ in range(n_teams)]

    k = min(n_teams, len(stops))
    coords = np.array([[s["lat"], s["lon"]] for s in stops])

    if k <= 1:
        return [list(stops)] + [[] for _ in range(n_teams - 1)]

    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(coords)

    clusters: List[List[Dict]] = [[] for _ in range(n_teams)]
    for stop, label in zip(stops, labels):
        clusters[int(label)].append(stop)
    return clusters


def run_dispatch(actions: List[Dict], all_hotspots: List[Dict], n_teams: int, stop_budget: int) -> List[Dict]:
    """Maximum-coverage set cover + spatial-clustering dispatch.

    1. Greedy max-coverage selects the best stops (unchanged).
    2. K-means clusters them into `n_teams` geographic groups.
    3. Nearest-neighbour orders stops within each cluster.
    4. OSRM computes real road distance, time and geometry per route.
    """
    # Citywide burden (sum of severity across all hotspot cells)
    citywide_burden = sum(h.get("severity", 0.0) for h in all_hotspots)
    if citywide_burden == 0:
        return []

    # Candidates: centroids of the enforceable zones
    candidates = []
    for a in actions:
        candidates.append({
            "action_id": a["action_id"],
            "zone_id": a["zone_id"],
            "ward_id": a["ward_id"],
            "eps": a["eps"],
            "lat": a["centroid"]["lat"],
            "lon": a["centroid"]["lon"],
            "covered_burden": 0.0,
            "covered_cells": set()
        })

    # Greedy max-coverage stop selection (unchanged)
    selected_stops = []
    covered_cells = set()

    for _ in range(stop_budget):
        best_candidate = None
        best_marginal_coverage = -1.0
        best_newly_covered = set()

        for cand in candidates:
            if cand in selected_stops:
                continue

            marginal_coverage = 0.0
            newly_covered = set()

            for h in all_hotspots:
                cid = h["cell"]
                if cid in covered_cells:
                    continue

                h_lat, h_lon = cell_center(cid)
                if haversine_km(cand["lat"], cand["lon"], h_lat, h_lon) <= COVER_RADIUS_KM:
                    burden = h.get("severity", 0.0) * (cand["eps"] / 100.0)
                    marginal_coverage += burden
                    newly_covered.add(cid)

            if marginal_coverage > best_marginal_coverage:
                best_marginal_coverage = marginal_coverage
                best_candidate = cand
                best_newly_covered = newly_covered

        if best_candidate and best_marginal_coverage > 0:
            selected_stops.append(best_candidate)
            covered_cells.update(best_newly_covered)
        else:
            break

    # Coverage metric
    total_covered_burden = sum(h.get("severity", 0.0) for h in all_hotspots if h["cell"] in covered_cells)
    coverage_pct = round(100.0 * (total_covered_burden / citywide_burden), 1)

    # ── Spatial clustering instead of round-robin ──────────────────────────
    clusters = _cluster_stops(selected_stops, n_teams)

    teams = []

    for ti, cluster in enumerate(clusters):
        team: Dict[str, Any] = {
            "team_id": f"T{ti + 1}",
            "stops": [],
            "route_km": 0.0,
            "route_duration_min": 0.0,
            "coverage_pct": coverage_pct,
            "route_geometry": [],
        }
        if not cluster:
            teams.append(team)
            continue

        # ── Nearest-neighbour ordering within the cluster ─────────────
        unvisited = list(cluster)
        route = []
        seq = 1

        nxt = unvisited.pop(0)
        current_lat, current_lon = nxt["lat"], nxt["lon"]
        route.append({
            "seq": seq,
            "action_id": nxt["action_id"],
            "zone_id": nxt["zone_id"],
            "ward_id": nxt["ward_id"],
            "eps": nxt["eps"],
            "lat": nxt["lat"],
            "lon": nxt["lon"],
        })
        seq += 1

        while unvisited:
            best_idx = -1
            min_dist = float("inf")
            for i, stop in enumerate(unvisited):
                dist = haversine_km(current_lat, current_lon, stop["lat"], stop["lon"])
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i

            nxt = unvisited.pop(best_idx)
            current_lat, current_lon = nxt["lat"], nxt["lon"]
            route.append({
                "seq": seq,
                "action_id": nxt["action_id"],
                "zone_id": nxt["zone_id"],
                "ward_id": nxt["ward_id"],
                "eps": nxt["eps"],
                "lat": nxt["lat"],
                "lon": nxt["lon"],
            })
            seq += 1

        team["stops"] = route

        # ── OSRM road routing (stop₁ → stop₂ → …) ────────────
        waypoints = [(s["lat"], s["lon"]) for s in route]
        routing = route_waypoints(waypoints)
        team["route_km"] = routing["distance_km"]
        team["route_duration_min"] = routing["duration_min"]
        team["route_geometry"] = routing["geometry"]

        teams.append(team)

    return teams


def run(n_teams: int = 4, stop_budget: int = 10) -> None:
    try:
        hotspots = json.loads((DATA_OUT / "hotspots.json").read_text())
        attributions_list = json.loads((DATA_OUT / "attributions.json").read_text())
    except FileNotFoundError:
        print("[prioritise] hotspots.json or attributions.json not found. Run previous steps.")
        return

    attributions = {a["cell"]: a for a in attributions_list}

    try:
        panel = pd.read_parquet(DATA_OUT / "panel.parquet")
    except Exception:
        panel = pd.DataFrame()

    # Group by zone_id
    by_zone = {}
    for h in hotspots:
        zone = h.get("zone_id") or h["cell"]
        by_zone.setdefault(zone, []).append(h)

    actions = []
    action_idx = 1

    for zone_id, cells in by_zone.items():
        attributable = cells[0].get("attributable", True)
        if not attributable:
            continue

        eps_data = calculate_eps(cells, attributions, panel)

        # Centroid
        lats, lons = [], []
        for c in cells:
            lat, lon = cell_center(c["cell"])
            lats.append(lat)
            lons.append(lon)
        centroid = {"lat": round(sum(lats)/len(lats), 4), "lon": round(sum(lons)/len(lons), 4)}

        anchor = max(cells, key=lambda x: x.get("severity", 0.0))
        pm25_med = round(anchor.get("pm25_med", 0.0), 1)

        # Primary source and confidence from the anchor cell's attribution
        anchor_attr = attributions.get(anchor["cell"], {})
        source = anchor_attr.get("primary_source", "unknown")
        confidence = anchor_attr.get("confidence", 0.0)

        actions.append({
            "action_id": f"A{action_idx:02d}",
            "zone_id": zone_id,
            "ward_id": anchor.get("ward_id", "unknown") if not (isinstance(anchor.get("ward_id"), float) and __import__("math").isnan(anchor.get("ward_id"))) else "unknown",
            "ward_name": anchor.get("ward_name", "unknown") if not (isinstance(anchor.get("ward_name"), float) and __import__("math").isnan(anchor.get("ward_name"))) else "unknown",
            "cells": [c["cell"] for c in cells],
            "centroid": centroid,
            "eps": eps_data["eps"],
            "components": eps_data["components"],
            "kind": anchor.get("kind", "chronic"),
            "source": source,
            "confidence": confidence,
            "n_cells": len(cells),
            "pm25_med": pm25_med,
            "status": "pending"
        })
        action_idx += 1

    actions.sort(key=lambda x: x["eps"], reverse=True)

    (DATA_OUT / "actions.json").write_text(
        json.dumps(actions, indent=2).replace("NaN", "null")
    )
    print(f"[prioritise] Wrote {len(actions)} actions to actions.json")

    teams = run_dispatch(actions, hotspots, n_teams=n_teams, stop_budget=stop_budget)
    (DATA_OUT / "dispatch.json").write_text(
        json.dumps(teams, indent=2).replace("NaN", "null")
    )
    print(f"[prioritise] Wrote dispatch plan for {n_teams} teams (max {stop_budget} stops) to dispatch.json")


if __name__ == "__main__":
    run()
