"""Assembles a LONG-WINDOW training panel from the historical backfill
(data/historical/<city>/), for the forecaster's training loop — NOT the
operational 60-day panel (ingestion/preprocessing/panel.py::build_panel,
which is unrelated and untouched).

No satellite: data/historical/ never backfills Sentinel-5P (no GEE
credentials available in this environment — see
docs/superpowers/specs/2026-08-15-forecast-rework-design.md section 2).
Land use is static, so this reuses whatever OSM snapshot the operational
pipeline last wrote to data/raw/<city>/osm.parquet.
"""
from pathlib import Path

import pandas as pd

from shared.config import ROOT, DATA_RAW_BASE
from shared.grid import city_cells
from shared.wards import attach_wards
from ingestion.preprocessing.panel import _landuse_features, _fire_features


CHUNK_DAYS = 60  # bounds peak memory: _fire_features rebuilds its own cell x hour
                  # spine internally (duplicating what the outer spine already holds),
                  # and at the full 2-year x 1210-cell scale that transient double-hold
                  # is what actually exhausts RAM, not the final panel's own size.
                  # Chunking by time keeps every intermediate bounded to one chunk.


def build_historical_panel(city: str, hist_dir: Path | None = None,
                            raw_dir: Path | None = None) -> pd.DataFrame:
    hist = hist_dir or (ROOT / "data" / "historical" / city)
    raw = raw_dir or (DATA_RAW_BASE / city)

    stations = pd.read_parquet(hist / "stations.parquet")
    weather = pd.read_parquet(hist / "weather.parquet")
    fires = pd.read_parquet(hist / "fires.parquet")
    osm = pd.read_parquet(raw / "osm.parquet")

    cells = city_cells()
    stations["ts"] = pd.to_datetime(stations.ts, utc=True).dt.floor("h")
    weather["ts"] = pd.to_datetime(weather.ts, utc=True).dt.floor("h")
    hours = pd.DatetimeIndex(sorted(set(stations.ts) & set(weather.ts)))
    if len(hours) == 0:
        raise ValueError(
            f"no overlapping hours between historical stations "
            f"({stations.ts.min()} .. {stations.ts.max()}) and weather "
            f"({weather.ts.min()} .. {weather.ts.max()}) for {city} — "
            f"check {hist}/manifest.json")

    landuse = _landuse_features(cells, osm)  # static, built once outside the loop

    chunk_hours = CHUNK_DAYS * 24
    chunks = []
    for start in range(0, len(hours), chunk_hours):
        chunk = hours[start:start + chunk_hours]
        part = pd.MultiIndex.from_product([cells, chunk], names=["cell", "ts"]).to_frame(index=False)

        st = (stations[stations.ts.isin(chunk)].groupby(["cell", "ts"], as_index=False)
                      .pm25.median().rename(columns={"pm25": "pm25_station"}))
        part = part.merge(st, on=["cell", "ts"], how="left")
        part = part.merge(weather, on="ts", how="left")
        part = part.merge(_fire_features(cells, fires, chunk), on=["cell", "ts"], how="left")
        part = part.merge(landuse, on="cell", how="left")
        chunks.append(part)
        print(f"[historical_panel] {city}: chunk {chunk[0].date()}..{chunk[-1].date()} "
              f"({len(part):,} rows)")

    panel = pd.concat(chunks, ignore_index=True)
    del chunks
    panel = attach_wards(panel)

    panel["hour"] = panel.ts.dt.hour
    panel["dow"] = panel.ts.dt.dayofweek
    panel["city"] = city

    out = hist / "panel.parquet"
    panel.to_parquet(out, index=False)
    n_st = panel.pm25_station.notna().sum()
    print(f"[historical_panel] {city}: {len(panel):,} rows ({len(cells)} cells x "
          f"{len(hours)} hours); {n_st:,} station-labeled rows "
          f"({panel[panel.pm25_station.notna()].cell.nunique()} station cells)")
    return panel


if __name__ == "__main__":
    import sys
    build_historical_panel(sys.argv[1] if len(sys.argv) > 1 else "bengaluru")
