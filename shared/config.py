"""Central configuration. One place to change city, resolution, paths."""
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_OUT = ROOT / "data" / "outputs"


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Populate os.environ from .env. Real env vars always win (CI sets secrets).

    Hand-rolled rather than python-dotenv: six lines, one fewer dependency, and
    the documented workflow ('copy .env.example to .env') has to actually work.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.split("#")[0].strip().strip("'\"")
        if key and val:
            os.environ.setdefault(key, val)


_load_dotenv()

# ---- City: Bengaluru (swap bbox + ward geojson to change city) ----
CITY = "bengaluru"
BBOX = {"lat_min": 12.85, "lat_max": 13.10, "lon_min": 77.45, "lon_max": 77.75}
WARD_GEOJSON = ROOT / "data" / "BBMP.geojson"   # official ward boundaries

H3_RES = 8            # ~460 m edge -> satisfies "1 km grid" requirement
PANEL_HOURS = 24 * 60 # synthetic/backfill window: 60 days hourly

# Multi-window detection. A real-time spike is noise; a source is something that
# is STILL there when you zoom out. We aggregate the signal over several windows
# and require agreement across them, which is what separates a long-term source
# (persistent across 7d and 30d) from a fire (loud in 24h, gone by 7d).
DETECT_WINDOWS_H = {"w24h": 24, "w7d": 24 * 7, "w30d": 24 * 30}

# Neighbourhood contrast rings (H3 k-ring indices). A cell is compared against
# the ANNULUS around it, not against the city: the dense urban core is high
# everywhere, and "high because the whole district is dense" is not a violator.
CONTRAST_INNER_K = 1         # the zone itself (~0.8 km). Wider dilutes a real
                             # spike into its own comparison group.
CONTRAST_OUTER_K = (5, 10)   # its surroundings (~4-8 km). Must sit OUTSIDE the
                             # satellite's ~2.5 km blur, or the source contaminates
                             # the baseline it is being measured against.

# Wards: point-in-polygon against WARD_GEOJSON when it exists, else a
# deterministic Voronoi tessellation of the H3 fabric (see shared/wards.py).
# BBMP has 198 wards; the fallback approximates that granularity.
N_FALLBACK_WARDS = 60

# Satellite realism: TROPOMI ground pixels are ~5.5 x 3.5 km, an order of
# magnitude coarser than an H3 res-8 cell. The synthetic satellite is blurred to
# match, so no model can cheat by reading a per-cell truth signal out of the
# columns. A 5.5 km box average has an equivalent Gaussian sigma of 5.5/sqrt(12)
# ~= 1.6 km — blurring harder than that is not conservatism, it is wrong: the
# source's own plume leaks into the annulus it gets compared against and cancels
# its own contrast.
SAT_BLUR_SIGMA_KM = 1.6

# Fires. FIRMS sees burning DIRECTLY — it is the one instrument that needs no
# inference — but a 2 km catchment smears one landfill across every cell around
# it. 1.5 km keeps the attribution honest about which cell is actually burning.
FIRE_RADIUS_KM = 1.5
# Fraction of window-hours on fire that counts as one unit of evidence. 5% of a
# 30-day window is ~36 hours of burning: unambiguous, not a barbecue.
FIRE_FRAC_SCALE = 0.05

# API endpoints (all free)
OPENAQ_URL = "https://api.openaq.org/v3"
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"  # + /{KEY}/VIIRS_SNPP_NRT/{bbox}/2
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

for d in (DATA_RAW, DATA_OUT):
    d.mkdir(parents=True, exist_ok=True)
