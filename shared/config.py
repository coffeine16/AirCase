"""Central configuration. One place to change city, resolution, paths."""
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Base trees. The per-CITY paths (DATA_RAW / DATA_OUT) are defined below, once
# CITY is known — a city's raw feeds and its outputs are not interchangeable, and
# a single flat directory meant the last pipeline run silently owned them. That
# bit us: an export run with AQ_CITY=delhi wrote Bengaluru's artifacts into the
# Delhi bundle, because the env var said Delhi while the directory held Bengaluru.
DATA_RAW_BASE = ROOT / "data" / "raw"
DATA_OUT_BASE = ROOT / "data" / "outputs"


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

# ---- City ----
# Override with AQ_CITY=delhi. Everything downstream is city-agnostic; only the
# bbox and the ward layer change.
#
# The DEFAULT MUST STAY bengaluru. The synthetic world's 9 hidden sources are
# anchored to Bengaluru coordinates, and the zero-setup quickstart + CI both run
# `run_pipeline.py --synthetic --full` with no AQ_CITY set. Default this to delhi and
# every source lands ~1700 km outside the grid, the world emits nothing, and the
# evaluation scores 0/2, 0/4, 0/3 with a GREEN build (measured). Delhi is reached by
# setting AQ_CITY=delhi, as documented — it is not the default, and the demo does not
# need it to be. synthetic.py now raises on the mismatch rather than going quiet.
CITIES = {
    "bengaluru": {"lat_min": 12.85, "lat_max": 13.10, "lon_min": 77.45, "lon_max": 77.75},
    "delhi":     {"lat_min": 28.45, "lat_max": 28.75, "lon_min": 76.95, "lon_max": 77.35},
    "chennai":   {"lat_min": 12.95, "lat_max": 13.20, "lon_min": 80.12, "lon_max": 80.32},
    # Added for historical-backfill coverage expansion (data/historical/<city>/).
    # Each confirmed via a real OpenAQ v3 /locations pm25 query before being added
    # here — see scratchfile_notes for the per-city station counts.
    "mumbai":    {"lat_min": 18.89, "lat_max": 19.27, "lon_min": 72.77, "lon_max": 72.99},
    "kolkata":   {"lat_min": 22.45, "lat_max": 22.65, "lon_min": 88.28, "lon_max": 88.45},
    "hyderabad": {"lat_min": 17.30, "lat_max": 17.55, "lon_min": 78.35, "lon_max": 78.60},
    "pune":      {"lat_min": 18.45, "lat_max": 18.65, "lon_min": 73.75, "lon_max": 73.95},
    "ahmedabad": {"lat_min": 22.95, "lat_max": 23.15, "lon_min": 72.50, "lon_max": 72.70},
}
CITY = os.environ.get("AQ_CITY", "bengaluru").lower()
if CITY not in CITIES:
    raise ValueError(f"AQ_CITY={CITY!r} unknown; choose from {list(CITIES)}")
BBOX = CITIES[CITY]

# Per-city data trees. Bound by VALUE at import, so one PROCESS serves one city —
# which is exactly right for the batch pipeline and for a worker. A server that
# must answer for several cities resolves the directory per request instead
# (see app/backend/main.py::_city_out) rather than mutating these.
DATA_RAW = DATA_RAW_BASE / CITY
DATA_OUT = DATA_OUT_BASE / CITY

WARD_GEOJSON = ROOT / "data" / f"{CITY}_wards.geojson"   # real boundaries if present

# ---- The window the live collectors pull ----
# Default: now. Set AQ_WINDOW_END=2025-11-30 to run the whole pipeline over a
# HISTORICAL episode instead.
#
# This exists because we measured the alternative. In July (monsoon) BOTH of the
# detector's best instruments are blind: cloud masks the S5P NO2 retrieval (29%
# coverage over Bengaluru), and nothing burns when it is wet — real FIRMS returns
# 2 fires over Bengaluru and 9 over Delhi in 60 days, against 281 in our synthetic
# world. Fire persistence is ~0 in every cell, so the FIRMS channel — half the
# detector, and the half that locates unregistered burning sources — contributes
# nothing at all.
#
# Delhi's stubble-burning season is Oct-Nov: dry skies (high NO2 coverage) and a
# landscape genuinely on fire. Running the platform there is not cherry-picking,
# it is pointing the instrument at the season it was built for and that the
# instrument can actually see.
WINDOW_END = os.environ.get("AQ_WINDOW_END")   # "YYYY-MM-DD" or None -> now


def window_end():
    """End of the collection window, as a UTC midnight Timestamp."""
    import pandas as pd
    if WINDOW_END:
        return pd.Timestamp(WINDOW_END, tz="UTC").normalize()
    return pd.Timestamp.now("UTC").normalize()

H3_RES = 8            # ~460 m edge -> satisfies "1 km grid" requirement
# A QUARTER, not two months. The forecaster's longest feature is roll_med_2160,
# a 90-day rolling median, so a 60-day panel could only ever fill 67% of that
# window — the model was trained on panels where it was complete, so serving it a
# partial one is train/serve skew in the feature meant to carry seasonal memory.
# Detection is unaffected either way (its longest window is 30d) and fusion only
# gains rows. 90 is also the practical ceiling: Open-Meteo's forecast endpoint
# stops at 92 days, and while a window this old routes to the ERA5 archive
# instead, staying under that keeps a "recent window" run working too.
PANEL_HOURS = 24 * 90 # synthetic/backfill window: 90 days hourly (one quarter)

# The synthetic world ends HERE, not at utcnow(). Anchoring it makes every
# reported number reproducible; with a sliding clock the same code gave 72
# hotspot cells at 22:00 and 93 at 02:00, and every stat silently carried an
# "as measured last Tuesday" caveat. Live mode is unaffected.
SYNTHETIC_ANCHOR = "2026-07-01T12:00:00"

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

# Google Earth Engine (Sentinel-5P). Auth is Application Default Credentials:
#   gcloud auth application-default login
# No service-account key: the project has constraints/iam.disableServiceAccountKeyCreation
# switched on, which is a good default we are not going to fight.
GEE_PROJECT = os.environ.get("GEE_PROJECT", "aq-intelligence")

# API endpoints (all free)
OPENAQ_URL = "https://api.openaq.org/v3"
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"  # ERA5, for historical windows
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"  # + /{KEY}/VIIRS_SNPP_NRT/{bbox}/2
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

for d in (DATA_RAW, DATA_OUT):
    d.mkdir(parents=True, exist_ok=True)
