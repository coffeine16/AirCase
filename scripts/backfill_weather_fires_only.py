"""One-off: fill weather+fires for cities that got a bulk-archive-only station
pull (data/historical/<city>/stations.parquet exists, weather/fires don't).

Does NOT touch stations.parquet — only adds weather.parquet/fires.parquet and
merges those two entries into the existing manifest.json. Same collector
functions as scripts/backfill_historical.py; RUN ONE CITY PER PROCESS (see
that script's docstring for why — AQ_CITY/BBOX are bound at import time).

    $env:AQ_CITY="pune"; $env:PYTHONPATH="."
    python scripts/backfill_weather_fires_only.py --days 730
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import CITY, ROOT
from ingestion.collectors.pollers import fetch_weather, fetch_fires

FETCHERS = {"weather": fetch_weather, "fires": fetch_fires}


def run(days: int, sources: list[str], merge: bool) -> dict:
    out = ROOT / "data" / "historical" / CITY
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    for name in sources:
        fn = FETCHERS[name]
        print(f"[fill] {CITY}/{name}: requesting {days} days ...")
        try:
            df = fn(days=days)
        except Exception as e:                      # noqa: BLE001
            print(f"[fill] {CITY}/{name} FAILED: {type(e).__name__}: {e}")
            manifest["sources"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            continue

        path = out / f"{name}.parquet"
        if merge and path.exists():
            existing = pd.read_parquet(path)
            df = (pd.concat([existing, df], ignore_index=True)
                    .drop_duplicates(subset="ts")
                    .sort_values("ts")
                    .reset_index(drop=True))

        df.to_parquet(path, index=False)
        entry = {"ok": True, "rows": len(df)}
        if "ts" in df.columns and len(df):
            entry["start"] = str(df.ts.min())
            entry["end"] = str(df.ts.max())
        manifest["sources"][name] = entry
        print(f"[fill] {CITY}/{name}: wrote {len(df):,} rows -> {path}")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[fill] {CITY}: manifest updated -> {manifest_path}")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--sources", default="weather,fires")
    p.add_argument("--merge", action="store_true",
                    help="concat+dedupe onto an existing parquet instead of overwriting")
    args = p.parse_args()
    run(args.days, args.sources.split(","), args.merge)
