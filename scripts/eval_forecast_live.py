"""Real-city forecast number without the full GEE pipeline.

The forecast RMSE-vs-persistence metric is scored at STATIONS only, and stations need
just two collectors — OpenAQ (pm25) + Open-Meteo (met) — neither of which touches
GEE. So we can measure the honest Delhi number in ~1 minute instead of a 15-minute
satellite pipeline. Set AQ_CITY / AQ_WINDOW_END as usual.
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")
import pandas as pd
from shared.config import CITY, window_end
from shared.wards import attach_wards
from ingestion.collectors.pollers import fetch_stations, fetch_weather
from intelligence.models.forecast import evaluate, HORIZONS, TEST_TAIL_DAYS

st = fetch_stations()
wx = fetch_weather()
st["ts"] = pd.to_datetime(st.ts, utc=True).dt.floor("h")
wx["ts"] = pd.to_datetime(wx.ts, utc=True).dt.floor("h")
# minimal panel: one row per (station cell, hour) with pm25_station + met
sta = st.groupby(["cell", "ts"], as_index=False).pm25.median().rename(columns={"pm25": "pm25_station"})
panel = sta.merge(wx, on="ts", how="left")
panel["city"] = CITY
for c in ["lu_road", "lu_industrial", "lu_traffic"]:
    panel[c] = 0
# build_features requires these three and does NOT synthesise them:
# ward_id is the climatology's middle fallback scope, and fires_6h/frp_6h are
# in FEATURE_COLUMNS. Without them this script raised KeyError on the first
# call. Same ward helper build_panel() uses — not a reinvented assignment.
panel = attach_wards(panel)
panel["fires_6h"] = 0
panel["frp_6h"] = 0.0
print(f"[{CITY}] {panel.cell.nunique()} stations, {len(panel):,} station-hours, "
      f"window ends {window_end().date()}")

ev = evaluate(panel)
print("=" * 74)
print(f"PM2.5 FORECAST — REAL {CITY.upper()} — skill vs persistence "
      f"(last {TEST_TAIL_DAYS} days held out)")
print("=" * 74)
print(f"{'horizon':<9}{'n_test':>9}{'skill vs persist':>19}")
for h in HORIZONS:
    r = ev.get(f"h{h}", {})
    if "skill_vs_persistence_pct" not in r:
        print(f"{h}h  {r.get('note', 'n/a')}  (n_test={r.get('n_test')})"); continue
    print(f"{h}h{'':<6}{r['n_test']:>9}{r['skill_vs_persistence_pct']:>18.0f}%")
print("=" * 74)
