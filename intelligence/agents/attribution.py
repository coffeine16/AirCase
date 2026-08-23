"""Source Attribution Agent — the innovation core.

For each hotspot: assemble a structured evidence profile from data already in
the panel, score each source category deterministically, then have the LLM
write the grounded explanation (rule-based prose if no LLM). The LLM never
picks the answer alone — category scores are arithmetic; the LLM explains.

Evidence per hotspot:
  candidates        named sources within 5 km: distance, bearing, wind alignment
  pollutant_signature  satellite ratios vs city percentiles (SO2->industrial,
                       AAI->burning, NO2 @ rush hours->traffic)
  fire_activity     FIRMS detections near the cell, trailing 6 h
  landuse_context   source-kind counts within 1.5 km
  meteorology       wind, boundary layer (is air trapped? where does it come from?)

Output: data/outputs/attributions.json
"""
import json

import numpy as np
import pandas as pd

from shared.config import DATA_RAW, DATA_OUT
from shared.grid import cell_center, haversine_km, bearing_deg, wind_alignment, circular_mean_deg
from shared.wards import ward_frame
from intelligence.agents.llm_gateway import complete_json
from intelligence.models.forecast.spatial import DIST_DECAY_KM

CATEGORIES = ["industrial", "construction", "waste_burning", "traffic"]
MAX_SOURCE_KM = 5.0
WARDS = dict(zip(ward_frame().cell, ward_frame().ward_id))


# ------------------------------------------------------------ evidence
def build_evidence(cell: str, ts: pd.Timestamp, panel_row: pd.Series,
                   osm: pd.DataFrame, sat_pct: dict, fire_activity: dict,
                   wind_hist: list[float] | None = None,
                   citizen: dict | None = None) -> dict:
    lat, lon = cell_center(cell)
    # Bearings live on a circle: the arithmetic mean of 350 and 10 is 180, which
    # points the evidence chain at exactly the wrong suspect. Chronic hotspots
    # summarise a week of wind, so they must use the circular mean.
    wind_from = circular_mean_deg(wind_hist) if wind_hist else float(panel_row.wind_from_deg)
    hour = int(panel_row.hour)

    def align(slat, slon):
        # chronic hotspots: 'frequently downwind' = mean alignment over the
        # week's winds; anomalies: alignment with the wind right now
        if wind_hist:
            return float(np.mean([wind_alignment(slat, slon, lat, lon, w) for w in wind_hist]))
        return wind_alignment(slat, slon, lat, lon, wind_from)

    candidates = []
    for r in osm[osm.kind.isin(CATEGORIES)].itertuples():
        d = haversine_km(r.lat, r.lon, lat, lon)
        if d > MAX_SOURCE_KM:
            continue
        candidates.append({
            "name": r.name, "type": r.kind, "distance_km": round(d, 2),
            "wind_alignment": round(align(r.lat, r.lon), 2),
        })
    candidates.sort(key=lambda c: (-c["wind_alignment"] / (1 + c["distance_km"])))

    sig = {}
    for col, pcts in sat_pct.items():
        v = float(panel_row[col])
        sig[col] = {"value": round(v, 1),
                    "city_percentile": int(100 * np.searchsorted(pcts, v) / len(pcts))}

    return {
        "cell": cell, "ts": str(ts), "ward_id": WARDS.get(cell, "unassigned"),
        "pm25_estimate": round(float(panel_row.get("pm25_hat", np.nan)), 1),
        "candidates": candidates[:8],
        "pollutant_signature": sig,
        "fire_activity": fire_activity,
        # Reports from the n8n channels (web form / Telegram), synced from Supabase
        # by scripts/sync_supabase.py. Empty when the channel layer is not running —
        # which must leave every score EXACTLY as it was.
        "citizen_corroboration": citizen or {"n_reports": 0, "categories": {}},
        "landuse_context": {k: int(panel_row[f"lu_{k}"]) for k in CATEGORIES},
        "meteorology": {"wind_from_deg": round(wind_from), "wind_ms": round(float(panel_row.wind_ms), 1),
                        "blh_m": round(float(panel_row.blh_m)), "hour_local": hour,
                        "air_trapped": bool(panel_row.blh_m < 500)},
    }


def citizen_corroboration_for(reports: list[dict], ward_id: str,
                              ts: pd.Timestamp, window_h: int) -> dict:
    """Reports matching this zone's WARD within its evidence window.

    Matching is by ward — the administrative unit a citizen actually knows and the
    unit n8n validates against — never by raw distance to an unverified phone GPS.

    GUARD: the "unassigned" pseudo-ward (cells outside municipal limits) never
    matches. All outside-limits cells share that one label, so without the guard a
    single report anywhere outside city limits would "corroborate" every
    outside-limits zone across the whole bbox — cross-contamination wearing the
    face of evidence. No real ward, no ward-channel corroboration.
    """
    empty = {"n_reports": 0, "categories": {}, "window_hours": window_h}
    if not reports or not ward_id or ward_id == "unassigned":
        return empty
    cats: dict[str, int] = {}
    for r in reports:
        if r.get("ward_id") != ward_id or r.get("category") in (None, "other"):
            continue
        try:
            rts = pd.Timestamp(r.get("ts"))
            if rts.tzinfo is None:
                rts = rts.tz_localize("UTC")
        except (ValueError, TypeError):
            continue
        if ts - pd.Timedelta(hours=window_h) <= rts <= ts:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
    return {"n_reports": sum(cats.values()), "categories": cats,
            "window_hours": window_h}


# ------------------------------------------------- deterministic scores
def category_scores(ev: dict) -> dict:
    """Plain arithmetic per category. The LLM never touches this."""
    s = {c: 0.0 for c in CATEGORIES}
    for cand in ev["candidates"]:
        # upwind proximity: alignment high & distance small -> strong.
        #
        # MAX, not SUM. Summing asks "how many mapped sites of this type are near
        # me", which is a question about OSM's coverage, not about who is polluting.
        # Our synthetic world had 2 traffic corridors; real Delhi has 1,240 mapped
        # ones, so traffic accumulated an unbeatable score by sheer count and buried
        # a burning landfill with a strong fire signal underneath it. (Measured:
        # both Bhalswa and Okhla were attributed to `traffic`.)
        #
        # The right question is "how strong is the BEST candidate of this type",
        # which is what an inspector would ask.
        # DIST_DECAY_KM, not a second, unreconciled literal -- imported from
        # spatial.py so the two kernels can never silently diverge (they
        # did: this used to hardcode its own 2.0, and the test meant to
        # catch a mismatch only ever re-checked spatial.py's own constant
        # against itself, never attribution.py's actual behaviour).
        v = cand["wind_alignment"] * np.exp(-cand["distance_km"] / DIST_DECAY_KM)
        s[cand["type"]] = max(s[cand["type"]], v)
    # SO2 -> industrial and AAI -> burning are GONE as scoring signals.
    #
    # They are textbook fingerprints and they are unusable here: measured on real
    # S5P over two cities, both have SNR < 1.1 — SO2 is 49% NEGATIVE with a MAD 30x
    # its median. Citing "SO2 at p85 citywide" in an evidence chain is citing a coin
    # flip, and that chain is what an administrator puts in front of a violator.
    # (Measured: it was the deciding evidence for calling Okhla landfill industrial.)
    #
    # What replaces them: for burning, FIRMS fire — direct observation, no inference.
    # For industry, the named OSM candidate plus NO2, which is a real combustion
    # tracer. See intelligence/agents/detect.py::POLLUTANTS for the measurements.
    sig = ev["pollutant_signature"]
    hour = ev["meteorology"]["hour_local"]
    if sig.get("no2_col", {}).get("city_percentile", 0) >= 80 and (7 <= hour <= 10 or 17 <= hour <= 20):
        s["traffic"] += 0.8
    # FIRE IS DIRECT OBSERVATION, AND IT OUTRANKS INFERENCE.
    #
    # Every other term in this function is an *inference* — this site is near, the
    # wind points that way, the land use looks industrial. A FIRMS thermal anomaly is
    # not an inference: it is a satellite measuring heat. Something is on fire, at
    # that place, right then.
    #
    # It used to cap at 1.4 while a construction site at zero distance could reach
    # 1.45 (candidate 1.0 + land-use 0.45) — so proximity to a building site could
    # out-argue a satellite watching the ground burn. Measured: it mislabelled one of
    # our two burning sources, and on real Delhi it was part of why Okhla came back
    # as `traffic`.
    #
    # Scored on the FRACTION of the window burning, not a raw count: a count is not
    # comparable across a 6-hour acute window and a 30-day chronic one. Sustained
    # burning is what separates a landfill from somebody's bonfire — so a single
    # detection stays modest (1.0), and persistent burning becomes decisive (3.0).
    frac = ev["fire_activity"]["fire_hour_fraction"]
    if frac > 0:
        s["waste_burning"] += 1.0 + 2.0 * min(frac / 0.10, 1.0)
    # CITIZEN CORROBORATION: a human photographing a source is a real, independent
    # instrument — it is also the ONLY instrument that can see construction dust
    # (0/3 blind spot: coarse PM, no satellite tracer, does not burn). But a single
    # report is unverified, so the weight is modest and HARD-CAPPED: three reports
    # saturate it. A brigade of reports must never out-shout the satellite or fire
    # channels (0.25*3 = 0.75 max, below a single sustained-fire signal at 1.0-3.0).
    for cat, n in ev.get("citizen_corroboration", {}).get("categories", {}).items():
        if cat in s:
            s[cat] += 0.25 * min(int(n), 3)
    for k in CATEGORIES:
        s[k] += 0.15 * min(ev["landuse_context"][k], 3)
    if 8 <= hour <= 18:
        s["construction"] *= 1.2   # construction is a daytime activity
    else:
        s["construction"] *= 0.5
    return {k: round(v, 3) for k, v in s.items()}


def independent_signals(ev: dict, top: str) -> int:
    """How many INDEPENDENT evidence types point at `top`.

    Independence is the point: a named candidate upwind, a matching pollutant
    fingerprint, satellite-confirmed fire, and land-use context are four different
    instruments. Four instruments agreeing is a real result; one instrument
    shouting is not.
    """
    n = 0
    if any(c["type"] == top and c["wind_alignment"] > 0.4 for c in ev["candidates"]):
        n += 1
    # only NO2 is a usable fingerprint; SO2/AAI are noise (see category_scores)
    if top == "traffic" and ev["pollutant_signature"].get("no2_col", {}).get("city_percentile", 0) >= 80:
        n += 1
    if top == "waste_burning" and ev["fire_activity"]["fire_hour_fraction"] > 0:
        n += 1
    if ev["landuse_context"].get(top, 0) > 0:
        n += 1
    if ev.get("citizen_corroboration", {}).get("categories", {}).get(top, 0) > 0:
        n += 1
    # cap at 4: agreement stays in [0,1] and the VALIDATED confidence calibration
    # (hits 0.66 / misses 0.42, 100% precision above 0.70) is untouched whenever
    # citizen data is absent — a fifth agreeing instrument saturates rather than
    # inflating scores beyond what was calibrated.
    return min(n, 4)


def confidence_from(scores: dict, ev: dict, top: str) -> float:
    """Evidence agreement — never LLM self-report.

    Margin alone was not enough: measured against synthetic truth it scored 0.74
    on hits and 0.75 on misses, i.e. it carried no information about correctness,
    which quietly breaks the "every claim ships a confidence" principle. A wrong
    call with one weak candidate and nothing else can still win by a wide margin
    over three zeros. So confidence now combines three things that actually
    differ between hits and misses:

      margin   — how decisively the top category beat the runner-up
      strength — how much absolute evidence there is at all (a top score of 0.2
                 means we know essentially nothing, whatever the margin)
      agreement— how many INDEPENDENT instruments point the same way
    """
    vals = sorted(scores.values(), reverse=True)
    if vals[0] <= 0:
        return 0.05
    margin = (vals[0] - vals[1]) / vals[0]
    strength = float(np.clip(vals[0] / 2.0, 0, 1))
    agreement = independent_signals(ev, top) / 4.0
    conf = 0.10 + 0.30 * margin + 0.30 * strength + 0.30 * agreement

    # IGNORANCE GATE. Margin is a RATIO, so when every rival scores 0 it reads 1.0 —
    # a top score of 0.06 against three zeros looked as decisive as a top score of
    # 3.0, and returned 0.48. That is near-total ignorance wearing the face of
    # moderate certainty, and it is the exact failure a confidence number exists to
    # prevent. Scale the whole thing by how much evidence there actually IS.
    conf *= min(1.0, vals[0] / 0.5)
    return round(float(np.clip(conf, 0.05, 0.95)), 2)


# ---------------------------------------------------------- reasoners
PROMPT = """You are an air-pollution attribution analyst. Using ONLY the evidence
below, explain the most likely primary source of this PM2.5 hotspot. Do not
invent facts not present in the evidence. The category scores were computed
deterministically; your job is to explain, not to re-decide.

EVIDENCE:
{evidence}

CATEGORY SCORES (deterministic): {scores}

Return STRICT JSON only:
{{"primary_source": "<one of {cats}>",
  "reason": "<2-3 sentences citing specific evidence items>",
  "evidence_factors": ["<short factor>", "..."]}}"""


def rule_based_reason(ev: dict, scores: dict, top: str) -> dict:
    factors = []
    best_cand = next((c for c in ev["candidates"] if c["type"] == top), None)
    if best_cand:
        factors.append(f'{best_cand["name"]} is {best_cand["distance_km"]} km away, '
                       f'wind alignment {best_cand["wind_alignment"]}')
    sig = ev["pollutant_signature"]
    if top == "traffic" and sig.get("no2_col", {}).get("city_percentile", 0) >= 80:
        factors.append(f'NO2 column at p{sig["no2_col"]["city_percentile"]} citywide')
    fa = ev["fire_activity"]
    if top == "waste_burning" and fa["fire_hours"]:
        factors.append(f'satellite fire detections in {fa["fire_hours"]} hours '
                       f'({fa["fire_hour_fraction"]:.0%} of the window)')
    cit = ev.get("citizen_corroboration", {})
    if cit.get("categories", {}).get(top):
        factors.append(f'{cit["categories"][top]} citizen report(s) of '
                       f'{top.replace("_", " ")} in this ward within the window')
    if ev["meteorology"]["air_trapped"]:
        factors.append(f'shallow boundary layer ({ev["meteorology"]["blh_m"]} m) trapping emissions')
    if not factors:
        factors.append("no corroborating evidence beyond weak land-use context")
    reason = (f"Deterministic evidence scoring points to {top.replace('_', ' ')} "
              f"as the primary source. " + "; ".join(factors[:3]) + ".")
    return {"primary_source": top, "reason": reason, "evidence_factors": factors}


def attribute_one(ev: dict) -> dict:
    scores = category_scores(ev)
    top = max(scores, key=scores.get)
    conf = confidence_from(scores, ev, top)
    llm_out, provider = complete_json(PROMPT.format(
        evidence=json.dumps(ev, indent=1), scores=json.dumps(scores), cats=CATEGORIES))
    if llm_out and llm_out.get("primary_source") in CATEGORIES:
        # guardrail: LLM may only explain; if it disagrees with arithmetic, arithmetic wins
        if llm_out["primary_source"] != top:
            llm_out = rule_based_reason(ev, scores, top)
            provider += "+overridden"
        result = llm_out
    else:
        result, provider = rule_based_reason(ev, scores, top), "rules"
    return {"cell": ev["cell"], "ts": ev["ts"], "ward_id": ev["ward_id"],
            "pm25_estimate": ev["pm25_estimate"],
            "primary_source": result["primary_source"], "confidence": conf,
            "scores": scores, "reason": result["reason"],
            "evidence_factors": result.get("evidence_factors", []),
            "evidence": ev, "explained_by": provider}


# -------------------------------------------------------------- runner
# The evidence window must match the CLAIM being made. Attributing a standing
# violator off one hour of wind is how you name the wrong factory; attributing a
# fire off a 30-day median is how you miss it entirely.
# The evidence window MUST match the detection window that fired. `acute` used to
# reason over 6 hours while detection fires it off the 24h channel — so a kiln
# detected on a fire in the last 24h was explained using an evidence window that
# contained no fire at all, scored 0.00 on every category, and was labelled
# `traffic` on a residue of 0.06. Explain a hotspot with the evidence that caused it.
WINDOW_HOURS = {"chronic": 24 * 30, "emerging": 24 * 7, "acute": 24}


def run(top_n: int = 100) -> list[dict]:
    all_hot = json.loads((DATA_OUT / "hotspots.json").read_text())
    # Only ENFORCEABLE hotspots get a case built. A diffuse urban-background zone
    # is real pollution with nobody to serve a notice on: it stays on the map and
    # feeds ward advisories, but naming a "primary source" for it would be
    # inventing a culprit. Policy target, not inspection target.
    hotspots = [h for h in all_hot if h.get("attributable", True)][:top_n]
    n_diffuse = len(all_hot) - len([h for h in all_hot if h.get("attributable", True)])
    panel = pd.read_parquet(DATA_OUT / "panel.parquet")
    field = pd.read_parquet(DATA_OUT / "fusion_field.parquet")
    osm = pd.read_parquet(DATA_RAW / "osm.parquet")
    panel["ts"] = pd.to_datetime(panel.ts, utc=True)
    field["ts"] = pd.to_datetime(field.ts, utc=True)
    panel = panel.merge(field, on=["cell", "ts"], how="left")
    sat_pct = {c: np.sort(panel[c].dropna().values) for c in ("no2_col", "so2_col", "aai")}

    # ATTRIBUTE ONE ZONE, NOT ONE CELL.
    #
    # A zone IS a source. Attributing its cells independently produced one zone whose
    # 11 cells carried three different labels — an incoherent case file, and worse: a
    # burning source's FRINGE cells sit 2 km out where `fires_6h` is 0, so fire
    # evidence never reached them and a nearby construction site out-argued a
    # satellite watching the ground burn. (Measured: it cost us 1 of 2 correct names.)
    #
    # Same bug we already fixed in detection — persistence is a property of a SOURCE,
    # not a cell — which had simply survived over here. Evidence is now pooled across
    # the zone: fire ANYWHERE in the zone is fire evidence FOR the zone, and every
    # cell inherits the one verdict.
    by_zone: dict[str, list[dict]] = {}
    for h in hotspots:
        by_zone.setdefault(h.get("zone_id") or h["cell"], []).append(h)

    # citizen reports (synced from Supabase). Absent file -> empty list -> every
    # zone gets n_reports 0 and scoring is byte-identical to the no-channel world.
    reports_p = DATA_OUT / "citizen_reports.json"
    reports = []
    if reports_p.exists():
        try:
            reports = json.loads(reports_p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            reports = []

    out = []
    for zone_id, cells in by_zone.items():
        anchor = max(cells, key=lambda h: h["severity"])   # the zone's worst cell
        ts = pd.Timestamp(anchor["ts"])
        kind = anchor.get("kind", "chronic")
        cell_ids = [h["cell"] for h in cells]

        w = panel[(panel.cell.isin(cell_ids))
                  & (panel.ts > ts - pd.Timedelta(hours=WINDOW_HOURS[kind]))
                  & (panel.ts <= ts)]
        if w.empty:
            continue

        # MEDIAN, not mean: one spike hour must not be able to invent an
        # industrial signature out of a quiet month.
        row = w.median(numeric_only=True)
        row["hour"] = 12 if kind != "acute" else int(ts.hour)

        # Fire is pooled over the zone by its BURNIEST cell, not averaged: a landfill
        # burns at the landfill, not evenly across the 2 km blob it lights up.
        per_cell = w.groupby("cell").fires_6h.apply(lambda s: (s > 0).mean())
        frac = float(per_cell.max()) if len(per_cell) else 0.0
        fire_activity = {
            "fire_hours": int((w[w.cell == per_cell.idxmax()].fires_6h > 0).sum()) if len(per_cell) else 0,
            "fire_hour_fraction": round(frac, 4),
            "frp_p90": round(float(w.frp_6h.quantile(0.9)), 1),
        }

        citizen = citizen_corroboration_for(reports, WARDS.get(anchor["cell"], ""),
                                            ts, WINDOW_HOURS[kind])
        ev = build_evidence(anchor["cell"], ts, row, osm, sat_pct, fire_activity,
                            wind_hist=w.wind_from_deg.tolist(), citizen=citizen)
        ev["hotspot_kind"] = kind
        ev["evidence_window_hours"] = WINDOW_HOURS[kind]
        ev["zone_cells"] = len(cell_ids)
        verdict = attribute_one(ev)

        # every cell in the zone inherits the ONE verdict (the API is keyed by cell)
        for h in cells:
            rec = dict(verdict)
            rec["cell"] = h["cell"]
            rec["zone_id"] = zone_id
            out.append(rec)

    (DATA_OUT / "attributions.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if out:
        by_src = pd.Series([o["primary_source"] for o in out]).value_counts().to_dict()
        prov = pd.Series([o["explained_by"] for o in out]).value_counts().to_dict()
        print(f"[attribute] {len(out)} enforceable hotspots attributed {by_src} "
              f"(explained by {prov})")
    print(f"[attribute] {n_diffuse} diffuse urban-background hotspots left unattributed "
          f"(no locatable source — policy target, not an inspection)")
    return out


if __name__ == "__main__":
    run()
