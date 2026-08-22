"""Phase 0 acceptance check: full 8-city native city-LOSO run, compared
against the real, already-committed v2 manifest (models/manifest.json).
Not a pytest -- this is the slow, full-scale real-data run the spec's
Phase 0 acceptance bar calls for. Run with:

    PYTHONPATH=. python scripts/check_native_pipeline_phase0.py

SCOPE: this script covers city-LOSO only. run_final_fit_native is not
imported or invoked here, and final-fit has NO numeric parity evidence
beyond Task 5's plausibility checks (quantile ordering p10<=p50<=p90,
plausible PM2.5 value range) in
tests/intelligence/models/forecast/native/test_final_fit_parity.py --
there is no committed pandas final-fit run to diff it against at this
scale. That is a known, accepted scope gap for Phase 0, not something
this script silently papers over."""
import json
from pathlib import Path

import pandas as pd

from intelligence.models.forecast import HORIZONS
from intelligence.models.forecast.features import FEATURE_COLUMNS, station_cells_only
from intelligence.models.forecast.native.streaming import run_city_loso_native

CITIES = ["ahmedabad", "bengaluru", "chennai", "delhi", "hyderabad", "kolkata", "mumbai", "pune"]


def main():
    panels, fires_by_city = {}, {}
    for c in CITIES:
        p = pd.read_parquet(f"data/historical/{c}/panel.parquet")
        p["city"] = c
        panels[c] = station_cells_only(p)
        # Real per-fire-event table -- matches __init__.py::run()'s own
        # loading (fire_pressure_regional is 0.0 for every row without it,
        # which would compare native against a committed manifest trained
        # WITH real fires: an uncontrolled variable, not architecture-only
        # noise).
        fires_p = Path(f"data/historical/{c}/fires.parquet")
        if fires_p.exists():
            fires_by_city[c] = pd.read_parquet(fires_p)

    native_result = run_city_loso_native(panels, HORIZONS, FEATURE_COLUMNS,
                                          fires_by_city=fires_by_city)

    committed = json.loads(Path("models/manifest.json").read_text())
    committed_city_loso = committed["eval"]["city_loso"]

    print(f"{'city':<12} {'committed':>10} {'native':>10} {'delta':>8} {'delta%':>8}")
    max_delta = max_rel_delta = 0.0
    for city in CITIES:
        committed_rmse = committed_city_loso[city]["rmse"]
        native_rmse = native_result["per_city"][city]["rmse"]
        delta = abs(committed_rmse - native_rmse)
        rel_delta = delta / committed_rmse * 100
        max_delta = max(max_delta, delta)
        max_rel_delta = max(max_rel_delta, rel_delta)
        print(f"{city:<12} {committed_rmse:>10.2f} {native_rmse:>10.2f} "
              f"{delta:>8.3f} {rel_delta:>7.2f}%")
    print(f"\nmax delta across all 8 cities: {max_delta:.3f}")
    print(f"max relative delta across all 8 cities: {max_rel_delta:.2f}%")


if __name__ == "__main__":
    main()
