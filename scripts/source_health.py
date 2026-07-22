"""Measure how gappy each data source actually is, and what survives when it lags.

THE QUESTION THIS ANSWERS: sources update asynchronously — CPCB stations drop out,
Sentinel-5P sees through cloud only ~2 days in 3, FIRMS fires are event-driven and
sparse. If the platform quietly depended on any one of them being fresh, it would
be brittle in exactly the way a city cannot afford.

It does not, and that is a design decision we can show receipts for:

  * DETECTION NEVER READS A STATION. It runs on satellite NO2 contrast + FIRMS
    fire persistence, both of which cover every cell equally. Measured, not
    asserted: scripts/eval_station_sensitivity.py re-runs detection with station
    siting dialled from realistic to none and recall does not move.
  * EVERY AGGREGATE IS A MEDIAN OVER A WINDOW (24h/7d/30d), never a mean over the
    latest reading. A source missing a third of its days still yields a valid
    30-day median; one spike hour cannot manufacture a source.
  * FIRE PERSISTENCE IS A FRACTION OF THE WINDOW, not a count, so it is defined
    for a source that reports 18 times in 60 days.
  * CRITICAL SOURCES REFUSE TO FAKE. pollers.NO_FALLBACK aborts a live run rather
    than substitute synthetic satellite/fire/OSM — each of those invents a place
    we would then accuse.

Writes data/outputs/source_health.json.

    PYTHONPATH=. AQ_CITY=delhi python scripts/source_health.py
"""
import json
import sys
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode the subscript in "NO₂".
# The JSON is written UTF-8 either way; this only stops the summary print from
# crashing after the work is already done.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd  # noqa: E402
from shared.config import CITY, DATA_OUT, DATA_RAW  # noqa: E402


def _read(name: str) -> pd.DataFrame | None:
    p = DATA_RAW / name
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:  # noqa: BLE001
        return None


def build() -> dict:
    sources = []

    # ── stations ─────────────────────────────────────────────────────────────
    st = _read("stations.parquet")
    if st is not None and len(st):
        n = st["station_id"].nunique()
        hours = st["ts"].dt.floor("h").nunique()
        possible = n * hours
        cov = round(100 * len(st) / possible, 1) if possible else None
        sources.append({
            "source": "OpenAQ / CPCB stations",
            "cadence": "hourly, best effort",
            "observed": f"{n} stations over {hours} hours",
            "completeness_pct": cov,
            "feeds": ["fusion exposure field", "citizen AQI display"],
            "if_stale": "Detection is UNAFFECTED — it never reads a station. The "
                        "exposure map and the citizen AQI number degrade.",
            "load_bearing_for_detection": False,
        })

    # ── satellite ────────────────────────────────────────────────────────────
    sat = _read("satellite.parquet")
    if sat is not None and len(sat):
        cov = round(100 * sat["no2_col"].notna().mean(), 1)
        sources.append({
            "source": "Sentinel-5P (TROPOMI) NO₂",
            "cadence": "one overpass per day, cloud-masked",
            "observed": f"{sat['cell'].nunique():,} cells × {sat['date'].nunique()} days",
            "completeness_pct": cov,
            "feeds": ["detection (NO₂ neighbourhood contrast)"],
            "if_stale": f"{100 - cov:.0f}% of cell-days are already cloud-masked and "
                        "detection still works: the 24h/7d/30d medians absorb gaps by "
                        "construction. Sustained loss degrades the NO₂ tier; the fire "
                        "tier is independent.",
            "load_bearing_for_detection": True,
        })

    # ── fires ────────────────────────────────────────────────────────────────
    fires = _read("fires.parquet")
    if fires is not None:
        sources.append({
            "source": "NASA FIRMS thermal anomalies",
            "cadence": "event-driven, several passes daily",
            "observed": f"{len(fires)} detections in the window",
            "completeness_pct": None,
            # A percentage would be a lie here: there is no expected rate to be
            # complete AGAINST. Days with no fire are the normal case, not a gap.
            "completeness_note": "not a rate — event-driven",
            "feeds": ["detection (fire persistence)", "attribution (fire evidence)"],
            "if_stale": "Sparsity is the NORMAL case, not a failure — this is why "
                        "persistence is a fraction of window-hours rather than a "
                        "count. Sustained loss degrades the thermal tier; the NO₂ "
                        "tier is independent.",
            "load_bearing_for_detection": True,
        })

    # ── weather ──────────────────────────────────────────────────────────────
    w = _read("weather.parquet")
    if w is not None and len(w):
        cov = round(100 * w["wind_ms"].notna().mean(), 1)
        sources.append({
            "source": "Open-Meteo (wind, boundary layer)",
            "cadence": "hourly",
            "observed": f"{len(w):,} hourly records",
            "completeness_pct": cov,
            "feeds": ["attribution (plume alignment)", "advisory (trapped-air flag)"],
            "if_stale": "Attribution loses wind alignment but keeps land-use and fire "
                        "evidence, and the confidence drops accordingly because it "
                        "counts agreeing instruments.",
            "load_bearing_for_detection": False,
        })

    # ── OSM ──────────────────────────────────────────────────────────────────
    osm = _read("osm.parquet")
    if osm is not None:
        sources.append({
            "source": "OpenStreetMap land use",
            "cadence": "static per run",
            "observed": f"{len(osm):,} features",
            "completeness_pct": None,
            "completeness_note": "not a rate — static snapshot",
            "feeds": ["attribution (named candidate sites)"],
            "if_stale": "Attribution loses named candidates but not the finding: the "
                        "two burning landfills we detect appear on NO map at all, and "
                        "were named from fire evidence alone.",
            "load_bearing_for_detection": False,
        })

    # ── citizens ─────────────────────────────────────────────────────────────
    rep = DATA_OUT / "citizen_reports.json"
    n_rep = len(json.loads(rep.read_text(encoding="utf-8"))) if rep.exists() else 0
    sources.append({
        "source": "Citizen reports (Telegram / web)",
        "cadence": "event-driven",
        "observed": f"{n_rep} reports synced",
        "completeness_pct": None,
        "completeness_note": "not a rate — event-driven",
        "feeds": ["attribution (corroboration)"],
        "if_stale": "Nothing breaks. Corroboration is capped at +0.75 and can never "
                    "out-vote an instrument — it is deliberately not load-bearing.",
        "load_bearing_for_detection": False,
    })

    return {
        "city": CITY,
        "claim": "No single source being stale can silence source detection.",
        "why": [
            "Detection reads satellite NO₂ contrast + FIRMS fire persistence only — "
            "never a station. Measured: eval_station_sensitivity.py moves station "
            "siting from realistic to none and recall does not change.",
            "Every aggregate is a MEDIAN over a 24h/7d/30d window, never a mean over "
            "the latest reading, so a source missing a third of its days still "
            "produces a valid estimate.",
            "The two detection instruments are independent: cloud blinds the "
            "satellite, cloud does not blind thermal fire detection.",
            "Critical sources refuse to fake. A live run ABORTS rather than "
            "substitute synthetic satellite, fire or OSM data (pollers.NO_FALLBACK), "
            "because each of those invents a place we would then accuse.",
        ],
        "honest_limit": "The instruments fail TOGETHER in monsoon: cloud masks NO₂ "
                        "and nothing burns when it is wet. Real FIRMS returns 2 fires "
                        "over Bengaluru in 60 days of July. Redundancy across sources "
                        "is not redundancy against a season.",
        "sources": sources,
    }


def main() -> None:
    health = build()
    out = DATA_OUT / "source_health.json"
    out.write_text(json.dumps(health, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[source-health] {len(health['sources'])} sources -> {out}")
    for s in health["sources"]:
        c = f"{s['completeness_pct']}%" if s["completeness_pct"] is not None else "—"
        print(f"   {s['source']:34} {c:>7}  detection-critical={s['load_bearing_for_detection']}")


if __name__ == "__main__":
    main()
