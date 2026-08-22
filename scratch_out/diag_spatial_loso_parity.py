"""Diagnostic run behind test_spatial_loso_parity.py's tolerance constant.
Runs spatial_loso (pandas) and run_spatial_loso_native side by side over
ALL 14 real stations across chennai/hyderabad/ahmedabad -- not the routine
test's 6-station subset -- to get the real per-station delta distribution
that justifies the test's 2.0 tolerance. Re-run with:

    PYTHONPATH=. python scratch_out/diag_spatial_loso_parity.py > scratch_out/diag_spatial_loso_parity.log 2>&1

Real wall-clock: ~30 min (14 pandas folds + 14 native folds, each rebuilding
features across all 3 cities). See diag_spatial_loso_parity.log for the
actual run this test's tolerance was set from."""
import pandas as pd
from intelligence.models.forecast import HORIZONS
from intelligence.models.forecast.features import FEATURE_COLUMNS, station_cells_only, downcast_panel
from intelligence.models.forecast.validation import spatial_loso
from intelligence.models.forecast.native.streaming import run_spatial_loso_native

cities = ['chennai', 'hyderabad', 'ahmedabad']
panels = []
for c in cities:
    p = pd.read_parquet(f'data/historical/{c}/panel.parquet')
    p['city'] = c
    panels.append(station_cells_only(p))
panel = downcast_panel(pd.concat(panels, ignore_index=True))

pandas_r = spatial_loso(panel, HORIZONS, FEATURE_COLUMNS)
native_r = run_spatial_loso_native(panel, HORIZONS, FEATURE_COLUMNS)
print("PANDAS_DONE", pandas_r)
print("NATIVE_DONE", native_r)
max_delta = 0.0
for s in pandas_r['per_station']:
    if s in native_r['per_station']:
        pr = pandas_r['per_station'][s]['rmse']
        nr = native_r['per_station'][s]['rmse']
        d = abs(pr - nr)
        max_delta = max(max_delta, d)
        print("STATION", s, 'pandas=', pr, 'native=', nr, 'delta=', d)
print("MAX_DELTA", max_delta)
