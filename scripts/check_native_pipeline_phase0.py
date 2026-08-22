"""Phase 0 acceptance check: full 8-city native city-LOSO + final-fit run,
compared against the real, already-committed v2 manifest
(models/manifest.json). Not a pytest -- this is the slow, full-scale
real-data run the spec's Phase 0 acceptance bar calls for. Run with:

    PYTHONPATH=. python scripts/check_native_pipeline_phase0.py
"""
import json
from pathlib import Path

import pandas as pd

from intelligence.models.forecast import HORIZONS
from intelligence.models.forecast.features import FEATURE_COLUMNS, station_cells_only
from intelligence.models.forecast.native.streaming import run_city_loso_native

CITIES = ["ahmedabad", "bengaluru", "chennai", "delhi", "hyderabad", "kolkata", "mumbai", "pune"]


def main():
    panels = {}
    for c in CITIES:
        p = pd.read_parquet(f"data/historical/{c}/panel.parquet")
        p["city"] = c
        panels[c] = station_cells_only(p)

    native_result = run_city_loso_native(panels, HORIZONS, FEATURE_COLUMNS)

    committed = json.loads(Path("models/manifest.json").read_text())
    committed_city_loso = committed["eval"]["city_loso"]

    print(f"{'city':<12} {'committed':>10} {'native':>10} {'delta':>8}")
    max_delta = 0.0
    for city in CITIES:
        committed_rmse = committed_city_loso[city]["rmse"]
        native_rmse = native_result["per_city"][city]["rmse"]
        delta = abs(committed_rmse - native_rmse)
        max_delta = max(max_delta, delta)
        print(f"{city:<12} {committed_rmse:>10.2f} {native_rmse:>10.2f} {delta:>8.3f}")
    print(f"\nmax delta across all 8 cities: {max_delta:.3f}")


if __name__ == "__main__":
    main()
