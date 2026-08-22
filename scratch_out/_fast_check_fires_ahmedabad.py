"""Fast smoke check before the expensive 3-city/14-station with-fires
diagnostic: single smallest city (ahmedabad, 50460 rows, 4 stations, per
scratch_out/_check_station_counts.py), first 2 real stations only, real
fires passed to BOTH spatial_loso and run_spatial_loso_native. Confirms the
with-fires call path runs without error and produces a sane delta before
committing ~30-90 min to the full run."""
import pandas as pd
from intelligence.models.forecast import HORIZONS
from intelligence.models.forecast.features import FEATURE_COLUMNS, station_cells_only, downcast_panel
from intelligence.models.forecast.validation import spatial_loso
from intelligence.models.forecast.native.streaming import run_spatial_loso_native

c = 'ahmedabad'
p = pd.read_parquet(f'data/historical/{c}/panel.parquet')
p['city'] = c
panel = downcast_panel(station_cells_only(p))
fires = pd.read_parquet(f'data/historical/{c}/fires.parquet')

station_cells = sorted(panel[panel.pm25_station.notna()].cell.unique())
subset = station_cells[:2]
print(f"{c}: {len(station_cells)} real stations total, using subset {subset}")
panel_subset = panel[panel.cell.isin(subset) | ~panel.pm25_station.notna()]

pandas_r = spatial_loso(panel_subset, HORIZONS, FEATURE_COLUMNS, fires=fires)
native_r = run_spatial_loso_native(panel_subset, HORIZONS, FEATURE_COLUMNS, fires=fires)
print("PANDAS", pandas_r)
print("NATIVE", native_r)
for s in subset:
    if s in pandas_r['per_station'] and s in native_r['per_station']:
        pr = pandas_r['per_station'][s]['rmse']
        nr = native_r['per_station'][s]['rmse']
        print("STATION", s, 'pandas=', pr, 'native=', nr, 'delta=', abs(pr - nr))
