"""Enforcement Memo Agent — Node 5. The demo climax, and the business case.

Turns a ranked action into a dispatch-ready enforcement memo: the evidence chain
verbatim, the ward it is addressed to, and THE LEGAL PROVISION IT VIOLATES.

    "We detected a burning landfill"          <- a detection
    "Here is the notice, citing the rule,
     addressed to the ward, ready to serve"  <- a product

THE NON-NEGOTIABLE RULE (principle 1, and it matters most here)
---------------------------------------------------------------
A DETERMINISTIC RULE ENGINE PICKS THE STATUTE. The LLM only writes the connective
prose. You cannot let a language model choose a legal citation — it will hallucinate
a section number that sounds right, and that document goes in front of a violator,
and possibly a judge. The eligibility conditions below are plain `eq / in / gte /
lte` comparisons against the situation, exactly as specified in the architecture
doc, and the same situation always yields the same citation.

If the LLM is unavailable, `rule_based_prose()` produces the identical schema. The
memo never fails silent.

⚠️ THE CITATIONS ARE INDICATIVE, NOT LEGAL ADVICE
-------------------------------------------------
These are the well-known provisions a pollution control board actually acts under,
and every memo says so on its face. Nobody on this team is a lawyer. The memo is a
DRAFT for an officer to review and sign, and it states that in the document itself —
which is also, as it happens, the honest way to ship it.

Reads `actions.json` (the EPS queue) when it exists. If the prioritisation agent has
not landed yet, it builds zone-level actions straight from hotspots + attributions so
this agent is never blocked on another one.

Output: data/outputs/memos.json  (+ POST /memo/{action_id} serves one)
"""
import json
from datetime import datetime, timezone

import pandas as pd

from shared.config import DATA_OUT, CITY
from intelligence.agents.llm_gateway import complete_json

# ---------------------------------------------------------------- AQI
# India National AQI breakpoints for PM2.5 (CPCB). PM2.5 concentration is NOT the
# AQI — they are different scales, and conflating them is the classic civic-tech
# error. GRAP stages key off the AQI, so the conversion has to be explicit.
NAQI_PM25 = [
    (0, 30, 0, 50, "Good"),
    (30, 60, 51, 100, "Satisfactory"),
    (60, 90, 101, 200, "Moderate"),
    (90, 120, 201, 300, "Poor"),
    (120, 250, 301, 400, "Very Poor"),
    (250, 1000, 401, 500, "Severe"),
]


def pm25_to_aqi(pm25: float) -> tuple[int, str]:
    """PM2.5 (ug/m3) -> (AQI, category) on the Indian NAQI scale."""
    for c_lo, c_hi, i_lo, i_hi, label in NAQI_PM25:
        if c_lo < pm25 <= c_hi:
            aqi = i_lo + (i_hi - i_lo) * (pm25 - c_lo) / (c_hi - c_lo)
            return int(round(aqi)), label
    return (0, "Good") if pm25 <= 0 else (500, "Severe")


# ------------------------------------------------------- the rule engine
# Each rule: `when` is a list of (field, operator, value). ALL must hold.
# Operators are exactly the four the architecture doc specifies.
LEGAL_RULES = [
    {
        "id": "SWM-2016-15zg",
        "when": [("source", "eq", "waste_burning")],
        "statute": "Solid Waste Management Rules, 2016",
        "provision": "Rule 15(zg)",
        "summary": "The local authority shall prohibit open burning of solid waste "
                   "on streets, open spaces, and at landfill sites.",
        "authority": "Municipal Commissioner / Local Body",
        "action": "Issue prohibitory notice; direct immediate extinguishment and "
                  "capping of the burning face; initiate penalty proceedings.",
        "penalty": "Environmental compensation as per CPCB/NGT schedule for open "
                   "burning of waste.",
    },
    {
        "id": "AIR-1981-31A",
        "when": [("source", "in", ["waste_burning", "industrial"]),
                 ("aqi", "gte", 301)],
        "statute": "Air (Prevention and Control of Pollution) Act, 1981",
        "provision": "Section 31A",
        "summary": "Power to issue directions, including closure, prohibition or "
                   "regulation of any industry, operation or process, and stoppage "
                   "of electricity or water supply.",
        "authority": "State Pollution Control Board",
        "action": "Issue direction under s.31A; consider stoppage of operations "
                  "pending compliance.",
        "penalty": "Non-compliance attracts prosecution under s.37.",
    },
    {
        "id": "EPA-1986-5",
        "when": [("source", "eq", "industrial")],
        "statute": "Environment (Protection) Act, 1986",
        "provision": "Section 5",
        "summary": "Power to give directions in writing to any person, officer or "
                   "authority, who shall be bound to comply.",
        "authority": "State Pollution Control Board / CPCB",
        "action": "Inspect consent conditions and stack emission compliance; issue "
                  "show-cause under s.5.",
        "penalty": "Contravention punishable under s.15.",
    },
    {
        "id": "CND-2016",
        "when": [("source", "eq", "construction")],
        "statute": "Construction and Demolition Waste Management Rules, 2016",
        "provision": "Rules 4 & 8",
        "summary": "Every waste generator shall prevent dust and segregate and store "
                   "C&D waste; the local authority shall enforce dust mitigation.",
        "authority": "Municipal Commissioner / Local Body",
        "action": "Inspect for anti-smog guns, barricading, covered transport and "
                  "water sprinkling; issue notice for dust-control non-compliance.",
        "penalty": "Environmental compensation per CPCB dust-mitigation schedule.",
    },
    {
        "id": "GRAP-STAGE",
        "when": [("aqi", "gte", 201)],
        "statute": "Graded Response Action Plan (CAQM)",
        "provision": "Stage escalation by AQI band",
        "summary": "Graded restrictions on construction, DG sets and dust-generating "
                   "activity, escalating with the AQI band.",
        "authority": "Commission for Air Quality Management",
        "action": "Apply the GRAP stage corresponding to the prevailing AQI band in "
                  "this ward.",
        "penalty": "As notified per stage.",
    },
]

_OPS = {
    "eq": lambda a, b: a == b,
    "in": lambda a, b: a in b,
    "gte": lambda a, b: a is not None and a >= b,
    "lte": lambda a, b: a is not None and a <= b,
}


def match_legal_basis(situation: dict) -> list[dict]:
    """Every rule whose conditions ALL hold. Deterministic. No LLM anywhere near it.

    Returns a list, not one rule: a burning landfill during a severe episode violates
    the SWM Rules *and* attracts a s.31A direction *and* sits inside a GRAP stage.
    An officer wants all three, not our opinion about which is best.
    """
    out = []
    for rule in LEGAL_RULES:
        if all(_OPS[op](situation.get(field), value) for field, op, value in rule["when"]):
            out.append({k: v for k, v in rule.items() if k != "when"})
    return out


# ----------------------------------------------------------- situation
def build_situation(action: dict, attribution: dict) -> dict:
    """The facts the rule engine and the prose both reason from. No inference here."""
    pm25 = float(action.get("pm25_med") or attribution.get("pm25_estimate") or 0.0)
    aqi, category = pm25_to_aqi(pm25)
    ev = attribution.get("evidence", {})
    return {
        "source": attribution.get("primary_source"),
        "confidence": attribution.get("confidence"),
        "pm25": round(pm25, 1),
        "aqi": aqi,
        "aqi_category": category,
        "kind": action.get("kind"),
        "zone_id": action.get("zone_id"),
        "ward_id": action.get("ward_id"),
        "ward_name": action.get("ward_name"),
        "n_cells": action.get("n_cells"),
        "evidence_factors": attribution.get("evidence_factors", []),
        "fire_hours": ev.get("fire_activity", {}).get("fire_hours", 0),
        "fire_fraction": ev.get("fire_activity", {}).get("fire_hour_fraction", 0.0),
        "blh_m": ev.get("meteorology", {}).get("blh_m"),
        "nearest_candidate_km": action.get("nearest_candidate_km"),
    }


# -------------------------------------------------------------- prose
PROMPT = """You are drafting the body of an enforcement memo for an Indian pollution
control board. Write ONLY the connective prose. You are NOT choosing the legal basis
— it has already been determined by a rule engine and is given to you. Do not cite
any statute, section or rule other than the ones supplied. Do not invent evidence.

SITUATION (all facts you may use):
{situation}

LEGAL BASIS (already decided — cite these and only these):
{legal}

Return STRICT JSON only:
{{"subject": "<one-line subject of the memo>",
  "background": "<2-3 sentences: what was detected, by which instrument, over what
                 period. Neutral, factual, no adjectives.>",
  "finding": "<2-3 sentences: what the evidence supports, and how confident, stated
              plainly. If confidence is below 0.6, say the finding is indicative and
              requires verification on site.>",
  "directive": "<2-3 sentences: what the addressee is being asked to do.>"}}"""


def rule_based_prose(sit: dict, legal: list[dict]) -> dict:
    """Identical schema, no LLM. The memo must never fail silent."""
    src = (sit["source"] or "unknown").replace("_", " ")
    conf = sit.get("confidence") or 0.0
    hedge = ("This finding is INDICATIVE and requires verification on site."
             if conf < 0.6 else
             "The evidence is consistent and independently corroborated.")
    factors = "; ".join(sit.get("evidence_factors", [])[:3]) or "see evidence chain"
    return {
        "subject": f"Air pollution source detected — {src} — {sit.get('ward_name') or sit.get('ward_id')} "
                   f"(zone {sit.get('zone_id')})",
        "background": (
            f"Satellite and ground observations over the monitoring window identify an "
            f"elevated pollution zone spanning {sit.get('n_cells')} cell(s) in "
            f"{sit.get('ward_name') or sit.get('ward_id')}. The zone is classified "
            f"'{sit.get('kind')}'. Estimated PM2.5 in the zone is {sit['pm25']} ug/m3, "
            f"corresponding to an AQI of {sit['aqi']} ({sit['aqi_category']})."),
        "finding": (
            f"The evidence indicates {src} as the primary source, at a computed "
            f"confidence of {conf:.2f}. Supporting evidence: {factors}. {hedge}"),
        "directive": (
            f"The addressee is directed to inspect the zone and act under the "
            f"provisions cited below: "
            + "; ".join(f"{l['statute']}, {l['provision']}" for l in legal) + "."),
    }


# --------------------------------------------------------------- memo
def generate_memo(action: dict, attribution: dict) -> dict:
    sit = build_situation(action, attribution)
    legal = match_legal_basis(sit)          # DETERMINISTIC. Always. No exceptions.

    prose, provider = complete_json(PROMPT.format(
        situation=json.dumps(sit, indent=1),
        legal=json.dumps([{k: l[k] for k in ("statute", "provision", "summary")} for l in legal], indent=1)))
    if not prose or not all(k in prose for k in ("subject", "background", "finding", "directive")):
        prose, provider = rule_based_prose(sit, legal), "rules"

    ref = (f"AQ/{CITY[:3].upper()}/{pd.Timestamp(action.get('ts', datetime.now(timezone.utc))).strftime('%Y%m%d')}"
           f"/{action.get('zone_id', 'Z00')}")

    return {
        "memo_id": ref,
        "action_id": action.get("action_id", action.get("zone_id")),
        "zone_id": action.get("zone_id"),
        "ward_id": action.get("ward_id"),
        "ward_name": action.get("ward_name"),
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "status": "draft",
        "situation": sit,
        "legal_basis": legal,                 # may be several; an officer wants them all
        **prose,
        "evidence_chain": attribution.get("evidence_factors", []),
        "evidence_full": attribution.get("evidence", {}),
        "drafted_by": provider,
        "disclaimer": ("Citations are indicative and generated by a deterministic rule "
                       "engine from the detected situation. This is a DRAFT for review "
                       "and signature by an authorised officer; it is not legal advice."),
    }


# -------------------------------------------------------------- runner
def _actions_from_hotspots() -> list[dict]:
    """Build zone-level actions when the EPS agent has not landed yet.

    So this agent is never blocked on another one. When actions.json exists it wins,
    and it brings the EPS ranking with it.
    """
    hs = json.loads((DATA_OUT / "hotspots.json").read_text())
    by_zone: dict[str, list[dict]] = {}
    for h in hs:
        if not h.get("attributable", True):
            continue                       # diffuse: a policy target, not a notice
        by_zone.setdefault(h["zone_id"], []).append(h)

    actions = []
    for i, (zid, cells) in enumerate(sorted(by_zone.items())):
        worst = max(cells, key=lambda c: c["severity"])
        actions.append({
            "action_id": f"A{i:02d}", "zone_id": zid,
            "ward_id": worst["ward_id"], "ward_name": worst["ward_name"],
            "kind": worst["kind"], "severity": worst["severity"],
            "pm25_med": worst["pm25_med"], "n_cells": len(cells),
            "nearest_candidate_km": worst.get("nearest_candidate_km"),
            "ts": worst["ts"], "cells": [c["cell"] for c in cells],
        })
    return actions


def run() -> list[dict]:
    path = DATA_OUT / "actions.json"
    if path.exists():
        actions = json.loads(path.read_text())
        print(f"[memo] {len(actions)} actions from the EPS queue")
    else:
        actions = _actions_from_hotspots()
        print(f"[memo] actions.json not found — built {len(actions)} zone actions from "
              f"hotspots (EPS agent not landed yet; ranking will improve when it does)")

    attrs = {a["cell"]: a for a in json.loads((DATA_OUT / "attributions.json").read_text())}

    memos = []
    for a in actions:
        cells = a.get("cells") or [a.get("cell")]
        att = next((attrs[c] for c in cells if c in attrs), None)
        if not att:
            continue
        memos.append(generate_memo(a, att))

    (DATA_OUT / "memos.json").write_text(json.dumps(memos, indent=2))
    by_prov = pd.Series([m["drafted_by"] for m in memos]).value_counts().to_dict()
    print(f"[memo] {len(memos)} memos written (prose by {by_prov})")
    for m in memos:
        cites = ", ".join(f"{l['statute'].split(',')[0]} {l['provision']}"
                          for l in m["legal_basis"]) or "no rule matched"
        print(f"[memo]   {m['memo_id']}  {m['situation']['source']:<14} "
              f"AQI {m['situation']['aqi']:>3}  -> {cites}")
    return memos


if __name__ == "__main__":
    run()
