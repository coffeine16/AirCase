"""One-time (or periodic) historical backfill for model training.

Pulls a LONG window of real station/weather/fire data into
data/historical/<city>/, so training doesn't re-poll rate-limited free APIs
on every iteration the way the operational 60-day panel does.

Reuses the SAME collector functions the operational pipeline uses
(ingestion/collectors/pollers.py::fetch_stations/fetch_weather/fetch_fires) —
same pagination, same FIRMS day-range chunking, same station QC (median+MAD
sensor-fault filtering). This script only changes WHERE the result is
written and HOW FAR BACK it reaches. No new API logic.

Satellite (Sentinel-5P) is deliberately NOT pulled here: it needs GEE
Application Default Credentials, which are not configured in this
environment. Historical satellite backfill is a separate prerequisite.

RUN ONE CITY PER PROCESS. shared.config resolves CITY/BBOX/DATA_RAW at
IMPORT time, and ingestion/collectors/pollers.py imports BBOX from it at
ITS OWN import time too — so reassigning os.environ["AQ_CITY"] mid-process
and looping would silently backfill every city using the FIRST city's bbox
(same class of bug as CLAUDE.md's "never bind a tunable as a default
argument" gotcha, one level up at the module). Set AQ_CITY before the
interpreter starts, once per city:

    $env:AQ_CITY="delhi"; $env:AQ_WINDOW_END="2025-11-30"
    python scripts/backfill_historical.py --days 730
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import CITY, ROOT
from ingestion.collectors.pollers import fetch_stations, fetch_weather, fetch_fires


def run(days: int) -> dict:
    out = ROOT / "data" / "historical" / CITY
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"city": CITY, "days_requested": days, "sources": {}}

    for name, fn in (("stations", fetch_stations), ("weather", fetch_weather), ("fires", fetch_fires)):
        print(f"[backfill] {CITY}/{name}: requesting {days} days ...")
        try:
            df = fn(days=days)
        except Exception as e:                      # noqa: BLE001 — one source failing must not sink the others
            print(f"[backfill] {CITY}/{name} FAILED: {type(e).__name__}: {e}")
            manifest["sources"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            continue

        df.to_parquet(out / f"{name}.parquet", index=False)
        entry = {"ok": True, "rows": len(df)}
        if "ts" in df.columns and len(df):
            entry["start"] = str(df.ts.min())
            entry["end"] = str(df.ts.max())
        manifest["sources"][name] = entry
        print(f"[backfill] {CITY}/{name}: wrote {len(df):,} rows -> {out / f'{name}.parquet'}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[backfill] {CITY}: manifest -> {out / 'manifest.json'}")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=730)
    args = p.parse_args()
    run(args.days)
