"""Export the current city's serving JSON into the frontend's per-city static dir.

The frontend switches cities by fetching /data/<city>/<contract>.json — no backend
needed (demo insurance, the whole design principle). This dumps everything the UI
reads for the CURRENT city (whatever is in data/outputs/ now) into
app/frontend/public/data/<city>/, converting the fusion parquet to the same JSON
shape the /fusion endpoint returns.

    PYTHONPATH=. AQ_CITY=delhi python scripts/export_city_static.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd  # noqa: E402
from shared.config import CITY, DATA_OUT, DATA_RAW  # noqa: E402

FE_DATA = Path(__file__).parent.parent / "app" / "frontend" / "public" / "data"
FE_AUDIO = Path(__file__).parent.parent / "app" / "frontend" / "public" / "audio"

# contracts the frontend fetches, copied as-is
JSON_CONTRACTS = [
    "hotspots.json", "attributions.json", "actions.json", "dispatch.json",
    "ledger.json", "audit.json", "wards.json", "forecast.json", "memos.json",
    "advisories.json",
    # Ward-level 3-hourly series behind the citizen timeline. Without this the
    # timeline silently renders nothing — cityFetch returns its empty fallback.
    "forecast_ward.json",
    # Validation surfaces — the numbers the brief is scored on. Shipping them
    # makes the rigour visible in the product instead of only in a slide.
    "forecast_eval.json", "loso.json", "source_health.json",
]


def _fusion_json() -> dict:
    """Rebuild the /fusion?hour_offset=0 payload from the parquet."""
    p = DATA_OUT / "fusion_field.parquet"
    if not p.exists():
        return {"ts": None, "n_hours": 0, "cells": []}
    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    latest = df[df["ts"] == df["ts"].max()]
    # attach ward_id from wards.json so the citizen map can colour by ward
    wards = {}
    wp = DATA_OUT / "wards.json"
    if wp.exists():
        for c in json.loads(wp.read_text(encoding="utf-8"))["cells"]:
            wards[c["cell"]] = c["ward_id"]
    # NaN -> null, deliberately. fusion._predict returns NaN where it has no
    # credible estimate (see its comment), and json.dumps would otherwise emit a
    # bare NaN token, which is invalid JSON that every browser JSON.parse
    # rejects outright -- one unestimated cell would blank the entire map.
    # null is the value FusionCell.pm25 is already typed for.
    import math
    def _pm(v):
        f = float(v)
        return None if math.isnan(f) else round(f, 1)
    cells = [{"cell": r.cell, "ward_id": wards.get(r.cell, "unassigned"),
              "pm25": _pm(r.pm25_hat)} for r in latest.itertuples()]
    return {"ts": str(latest["ts"].iloc[0]), "n_hours": int(df["ts"].nunique()), "cells": cells}


def _stations_json() -> list:
    p = DATA_RAW / "stations.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p).dropna(subset=["pm25"])
    if df.empty:
        return []
    latest = df.sort_values("ts").groupby("station_id").last().reset_index()
    return [{"station_id": r.station_id, "cell": r.cell, "lat": float(r.lat),
             "lon": float(r.lon), "pm25": round(float(r.pm25), 1), "ts": str(r.ts)}
            for r in latest.itertuples()]


def _fires_json() -> list:
    p = DATA_RAW / "fires.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    keep = [c for c in ("lat", "lon", "frp", "confidence", "ts") if c in df.columns]
    return json.loads(df[keep].to_json(orient="records", date_format="iso"))


def _satellite_json() -> list:
    p = DATA_RAW / "satellite.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    per = df.groupby("cell")["no2_col"].median().dropna()
    return [{"cell": c, "no2": round(float(v), 2)} for c, v in per.items()]


def _export_audio() -> int:
    """Ship this city's voice clips + manifest so the ward page can play them on a
    static deploy (no backend). Ward ids repeat across cities, so audio MUST be
    namespaced per city or Delhi's W002 would play for Chennai's W002."""
    src = DATA_OUT / "audio"
    if not src.exists():
        return 0
    dest = FE_AUDIO / CITY
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*"):        # keep it in sync, not accumulating
        old.unlink()
    n = 0
    for p in list(src.glob("*.mp3")) + list(src.glob("manifest.json")):
        shutil.copy2(p, dest / p.name)
        n += 1
    return n


def main() -> None:
    dest = FE_DATA / CITY
    dest.mkdir(parents=True, exist_ok=True)

    n = 0
    for name in JSON_CONTRACTS:
        src = DATA_OUT / name
        if src.exists():
            shutil.copy2(src, dest / name)
            n += 1
    # derived-from-parquet contracts
    (dest / "fusion_field.json").write_text(json.dumps(_fusion_json()), encoding="utf-8")
    (dest / "stations.json").write_text(json.dumps(_stations_json()), encoding="utf-8")
    (dest / "fires.json").write_text(json.dumps(_fires_json()), encoding="utf-8")
    (dest / "satellite.json").write_text(json.dumps(_satellite_json()), encoding="utf-8")

    n_audio = _export_audio()
    print(f"[export] {CITY}: {n} contracts + fusion/stations/fires/satellite "
          f"-> public/data/{CITY}/"
          + (f"; {n_audio} audio files -> public/audio/{CITY}/" if n_audio else ""))


if __name__ == "__main__":
    main()
