"""Phase 1 acceptance check: native spatial-LOSO over a representative
subset of real stations (a few per city, spanning all 8 cities) compared
against the real, already-committed v2 manifest's spatial_loso_rmse
(models/manifest.json). NOT a full 82-station run -- see this script's
own module docstring in the Phase 1 plan for why that's impractical as a
routine check. Run with:

    PYTHONPATH=. python scripts/check_native_spatial_loso_phase1.py

Loads real per-city fires and concatenates them into one `fires` frame,
same as train.py's train_and_promote (`all_fires = pd.concat(fires_by_
city.values())`) before it calls the pandas spatial_loso that produced
the committed 38.47. Omitting this (as a first draft of this script did)
would leave fire_pressure_regional at 0.0 for every row -- an
uncontrolled variable relative to the committed number, the exact trap
check_native_pipeline_phase0.py's own docstring already flags for
city-LOSO."""
import json
from pathlib import Path

import pandas as pd

from intelligence.models.forecast import HORIZONS
from intelligence.models.forecast.features import FEATURE_COLUMNS, station_cells_only, downcast_panel
from intelligence.models.forecast.native.streaming import run_spatial_loso_native

CITIES = ["ahmedabad", "bengaluru", "chennai", "delhi", "hyderabad", "kolkata", "mumbai", "pune"]
STATIONS_PER_CITY = 3  # keeps this a representative, not exhaustive, check


def main():
    panels, fires = [], []
    for c in CITIES:
        p = pd.read_parquet(f"data/historical/{c}/panel.parquet")
        p["city"] = c
        panels.append(station_cells_only(p))
        fires_p = Path(f"data/historical/{c}/fires.parquet")
        if fires_p.exists():
            fires.append(pd.read_parquet(fires_p))
    full_panel = downcast_panel(pd.concat(panels, ignore_index=True))
    all_fires = pd.concat(fires, ignore_index=True) if fires else None

    subset = []
    for city in CITIES:
        city_cells = full_panel[(full_panel.city == city) & (full_panel.pm25_station.notna())].cell.unique()
        subset.extend(sorted(city_cells)[:STATIONS_PER_CITY])
    print(f"[check] {len(subset)} stations across {len(CITIES)} cities "
          f"(of {full_panel[full_panel.pm25_station.notna()].cell.nunique()} real stations total)")

    subset_panel = full_panel[full_panel.cell.isin(subset) | ~full_panel.pm25_station.notna()]
    native_result = run_spatial_loso_native(subset_panel, HORIZONS, FEATURE_COLUMNS, fires=all_fires)

    committed = json.loads(Path("models/manifest.json").read_text())
    committed_overall = committed["eval"]["spatial_loso_rmse"]

    print(f"\ncommitted overall spatial_loso_rmse (all 82 real stations): {committed_overall}")
    print("NOTE: the two numbers above are NOT directly comparable -- different, smaller "
          "station population (a representative subset vs all 82 real stations). This "
          "checks the native path is plausible and non-degenerate, not that it beats the "
          "committed model.")
    print(f"native overall rmse (this {len(subset)}-station subset): {native_result['overall_rmse']}")
    print(f"native baseline (persistence) rmse: {native_result['baseline_rmse']}")
    print(f"\nper-station native results:")
    for station, r in sorted(native_result["per_station"].items()):
        print(f"  {station}: rmse={r['rmse']} baseline_rmse={r['baseline_rmse']} n={r['n']}")


if __name__ == "__main__":
    main()
