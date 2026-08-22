"""Re-run of diag_spatial_loso_parity.py, but WITH real fires data passed to
BOTH the pandas and native calls -- closing the gap the final whole-branch
review flagged: every parity test/diagnostic run so far called spatial_loso
and run_spatial_loso_native with fires=None, but train.py's real
engine="native" dispatch passes fires=all_fires (real per-fire-event data).

This is not neutral: build_features computes a "citywide representative
wind" as a circular mean over whatever cells are in the panel PASSED TO IT
(see features.py's wind_by_hour/wind_ms_by_hour, ~line 359-412), and hands
this to fire_pressure (spatial.py), which has a wind-alignment weighting
term. In the pandas pooled path this "citywide" wind is averaged across ALL
cities in the panel passed to build_features; in native's per-city
streaming loop (run_spatial_loso_native builds each city's features from
`city_panel = panel[panel.city == city]` alone), it's that ONE city's own
wind average. Real fires exercise fire_pressure_regional's wind-alignment
term; fires=None (every prior run) left it at 0.0 for every row, silently
routing around this divergence mechanism entirely.

Same 3-city/14-station scale as the original diag_spatial_loso_parity.py
(chennai/hyderabad/ahmedabad, all real stations in those cities) so this is
a real apples-to-apples re-run, not a new, larger check. Fires loaded the
same way scripts/check_native_pipeline_phase0.py and
check_native_spatial_loso_phase1.py already do (data/historical/{city}/
fires.parquet, concatenated).

Re-run with:
    PYTHONPATH=. python scratch_out/diag_spatial_loso_parity_with_fires.py > scratch_out/diag_spatial_loso_parity_with_fires.log 2>&1

Real wall-clock: expect the same order of magnitude as the original
diagnostic (~30 min there; real fires add per-fire wind-alignment lookups
inside fire_pressure, so this run may take longer).

STATUS: this DID complete (not just left runnable) -- a smaller, faster
check (single city, 2 real stations, real fires:
scratch_out/_fast_check_fires_ahmedabad.py, deltas 0.18/0.20) was judged
sufficient to close the finding before this full run finished, but it
finished anyway shortly after and confirms the same conclusion at the
scale that actually exercises the cross-city wind-population divergence
this diagnostic exists to check: max per-station delta 1.56, overall_rmse
delta 0.27 (25.75 vs 25.48) -- comfortably under the 2.0 tolerance and the
same order of magnitude as the no-fires diagnostic's 1.49 ceiling. See
scratch_out/diag_spatial_loso_parity_with_fires.log for the full output."""
import pandas as pd
from intelligence.models.forecast import HORIZONS
from intelligence.models.forecast.features import FEATURE_COLUMNS, station_cells_only, downcast_panel
from intelligence.models.forecast.validation import spatial_loso
from intelligence.models.forecast.native.streaming import run_spatial_loso_native

cities = ['chennai', 'hyderabad', 'ahmedabad']
panels, fires = [], []
for c in cities:
    p = pd.read_parquet(f'data/historical/{c}/panel.parquet')
    p['city'] = c
    panels.append(station_cells_only(p))
    fires_p = f'data/historical/{c}/fires.parquet'
    try:
        fires.append(pd.read_parquet(fires_p))
    except FileNotFoundError:
        print(f"WARNING: no fires.parquet for {c}")
panel = downcast_panel(pd.concat(panels, ignore_index=True))
all_fires = pd.concat(fires, ignore_index=True) if fires else None
print(f"loaded {0 if all_fires is None else len(all_fires)} real fire events across {len(cities)} cities")

pandas_r = spatial_loso(panel, HORIZONS, FEATURE_COLUMNS, fires=all_fires)
native_r = run_spatial_loso_native(panel, HORIZONS, FEATURE_COLUMNS, fires=all_fires)
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
print("OVERALL_DELTA", abs(pandas_r["overall_rmse"] - native_r["overall_rmse"]))
