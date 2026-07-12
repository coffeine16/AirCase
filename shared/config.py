"""Central configuration. One place to change city, resolution, paths."""
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_OUT = ROOT / "data" / "outputs"

# ---- City: Bengaluru (swap bbox + ward geojson to change city) ----
CITY = "bengaluru"
BBOX = {"lat_min": 12.85, "lat_max": 13.10, "lon_min": 77.45, "lon_max": 77.75}
WARD_GEOJSON = ROOT / "data" / "BBMP.geojson"   # official ward boundaries

H3_RES = 8            # ~460 m edge -> satisfies "1 km grid" requirement
PANEL_HOURS = 24 * 14 # synthetic/backfill window: 14 days hourly

# API endpoints (all free)
OPENAQ_URL = "https://api.openaq.org/v3"
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"  # + /{KEY}/VIIRS_SNPP_NRT/{bbox}/2
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

for d in (DATA_RAW, DATA_OUT):
    d.mkdir(parents=True, exist_ok=True)
