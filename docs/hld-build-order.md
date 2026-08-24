# AirCase — HLD, in Build Order

**What this is.** The system as it actually exists in the repo, laid out in the order
you would wire it from scratch. Every claim here was checked against the source, not
against `CLAUDE.md` or `architecture.md` — both of which are stale in places (they
still describe the frontend, forecast and dispatch as "not built"; all three exist).

**How to read it.** Each layer states what it *reads*, what it *writes*, the one or
two decisions that actually matter, and how to prove it works before moving on. The
contracts between layers are files on disk — that is the integration surface, and it
is why you can build and test each layer alone.

**The spine, in one line:**

```
collectors → panel → fusion → [forecast] → 9-agent chain → JSON contracts → API → frontend
   (raw)     (join)  (exposure)  (time)      (intelligence)     (disk)      (read-only)
```

**The single most important architectural rule:** heavy compute runs in batch and
writes files; serving only reads them. Every layer below either produces a file or
consumes one. Nothing trains inside a request.

---

## Layer 0 — Configuration and the spatial fabric

Build this first. Everything imports it, and two of its decisions are load-bearing
for the entire system.

### `shared/config.py`

The single source of truth for city, resolution, paths, and every tunable.

- **City registry** — 8 cities, each a bbox. `AQ_CITY` env var selects one;
  default is `bengaluru` and **must stay that way** (the synthetic world's hidden
  sources are anchored to Bengaluru coordinates — default it to Delhi and every
  source lands ~1,700 km outside the grid).
- **Per-city paths** — `DATA_RAW = data/raw/<city>`, `DATA_OUT = data/outputs/<city>`.
  These are bound **by value at import**, so *one process serves one city*. That is
  correct for batch; the API works around it with a per-request context var (Layer 7).
- **Window control** — `AQ_WINDOW_END=2025-11-30` runs the whole pipeline over a
  historical episode instead of "now". This exists because season determines whether
  the instruments can see anything at all: in monsoon, cloud masks NO₂ and nothing
  burns, so both detector channels go blind.
- `_load_dotenv()` — hand-rolled `.env` reader, no `python-dotenv` dependency. Real
  env vars win over file values.

Key constants: `H3_RES = 8` (~460 m), `PANEL_HOURS = 24*60` (60 days),
`DETECT_WINDOWS_H = {24h, 7d, 30d}`, `CONTRAST_INNER_K = 1`, `CONTRAST_OUTER_K = (5,10)`,
`FIRE_RADIUS_KM = 1.5`, `SAT_BLUR_SIGMA_KM = 1.6`, `SYNTHETIC_ANCHOR` (fixed, so the
synthetic world is reproducible rather than sliding with the wall clock).

### `shared/grid.py`

The H3 fabric — the universal spatial key.

- `city_cells()` — every res-8 cell whose centre falls in the bbox.
- `weather_grid_cells()` — coarse res-6 cells for weather query points. **Derived
  from the fine cells' actual H3 parents**, not a second `polygon_to_cells` call on
  the bbox: those two sets differ, and on real Chennai the gap orphaned 11% of cells
  with silently-NaN weather.
- Geometry: `haversine_km`, `bearing_deg`, `wind_alignment` (cosine of wind-to
  vs source-to-destination bearing, clamped to ≥0), `circular_mean_deg` (arithmetic
  mean of 350° and 10° gives 180° — exactly wrong).

### `shared/wards.py`

The administrative unit. Two modes, one schema:

- **Real** — point-in-polygon (hand-rolled ray casting, no geopandas) against
  `data/<city>_wards.geojson`. All 8 cities now have one.
- **Fallback** — deterministic Voronoi over `N_FALLBACK_WARDS` seeds, marked
  `synthetic=True`.

`_ward_name()` carries a per-city key list because every municipality names its
properties differently: BBMP uses `KGISWardName`, Datameet Delhi uses `Ward_Name`,
KMC Kolkata exposes only `WARD` (a bare number), GCC Chennai uses `Ward_No` + `Zone_Name`.
**Adding a 9th city means adding its key here**, or every ward silently becomes
"Ward 001" by list position — which would be wrong on every document you address.

### `shared/routing.py`
OSRM road routing for dispatch, falling back to haversine × road-factor when the
public demo endpoint is unreachable.

**Verify Layer 0:** `python shared/grid.py` prints the cell count;
`python shared/wards.py` prints ward sizes and states whether it used real
boundaries or the Voronoi fallback.

---

## Layer 1 — Ingestion

`ingestion/collectors/pollers.py` — five sources, one `run()`, five parquet files.

| Source | Function | API | Produces |
|---|---|---|---|
| Stations | `fetch_stations` | OpenAQ v3 | hourly PM2.5 per station cell — the **label** |
| Weather | `fetch_weather` | Open-Meteo | wind u/v, temp, **boundary layer height**, per weather-grid cell |
| Fires | `fetch_fires` | NASA FIRMS | thermal detections, lat/lon/frp |
| Land use | `fetch_osm` | Overpass | industrial/construction/kiln/road/school/hospital |
| Satellite | `fetch_s5p` → `sentinel.py` | GEE Sentinel-5P | NO₂ / SO₂ / aerosol index columns |

Writes `data/raw/<city>/{stations,weather,fires,osm,satellite}.parquet`.

**The `NO_FALLBACK` guard is the most important thing in this layer.** Satellite,
fires and OSM may **never** silently degrade to synthetic, because each of them
*invents a place the system would then accuse*. Stations may degrade — they feed the
exposure map, not the detector. Critically, the guard also rejects **zero-row
payloads**: a loaded Overpass mirror returns HTTP 200 with an empty element list, and
the pipeline once wrote an empty OSM layer and proceeded as if the city contained no
industry.

**Every collector must fetch the whole panel window.** FIRMS asked for 2 days, OpenAQ
14, Open-Meteo 14 — against a 60-day panel. Since `build_panel()` *intersects* station
and weather hours, the shortest one silently truncates everything, the 30-day detection
window comes back empty, and every chronic source vanishes with no error. FIRMS caps
`DAY_RANGE` at 5, so it is walked in chunks.

`ingestion/synthetic.py` — the offline world: 9 hidden sources, deliberately
adversarial (different dispersion physics than the scorer, sources on no map, decoy
sites that emit nothing, satellite blurred to its true footprint). `RNG` is
module-level and stateful; `generate_all()` calls `_reset_rng()` first so the world is
a pure function of `WORLD_SEED`.

**Verify:** `python ingestion/collectors/pollers.py` — check row counts per source and
that no critical source fell back.

---

## Layer 2 — The panel (the join everything reads)

`ingestion/preprocessing/panel.py::build_panel()`

Row = (H3 cell, hour). This is the one table every downstream component reads.

Assembly order:
1. **Spine** — `MultiIndex.from_product([cells, hours])`. `hours` is the
   **intersection** of station and weather timestamps, both floored to the hour
   (OpenAQ stamps period-ends, Open-Meteo stamps hour-starts — exact equality on raw
   timestamps comes back empty against live APIs). Raises if the intersection is empty
   rather than building a lying panel.
2. **Station label** — `pm25_station`, aggregated by **median** where two stations
   share a cell. This is the one place a mean would corrupt a *training label*.
3. **Weather** — joined per weather-grid cell via exact H3 parent lookup.
4. **Satellite** — daily columns forward-filled onto hours. **`ffill` only, never
   `bfill`** — a trailing backfill would use a later day's overpass to fill the
   window's first hours, a look-ahead leak.
5. **Fires** — `_fire_features` is fully vectorised (a (cell × fire) proximity mask,
   then a 6-hour trailing cumulative-sum convolution). The panel is ~1.7M rows; do
   **not** reintroduce a per-cell/per-hour Python loop.
6. **Land use** — static per-cell counts within 1.5 km. Note `lu_road` (generic
   density, the diffuse-background proxy) is deliberately separate from `lu_traffic`
   (named corridors, a suspect).
7. **Wards** — attached *and persisted* to `wards.json`, because the API must read
   what the pipeline wrote rather than re-derive wards from an env var it cannot see.
8. **Time features** — `hour`, `dow`, `city`.

Writes `data/outputs/<city>/panel.parquet` + `wards.json`.

**Gotcha:** `pd.DatetimeIndex.values` strips the timezone and silently breaks the
merge. Use `.repeat()` on the index itself.

`ingestion/preprocessing/historical_panel.py` builds the equivalent long-window panel
from `data/historical/<city>/` for forecast training — a separate path, because
forecasting needs years and detection needs 60 days.

---

## Layer 3 — Fusion (the exposure field)

`intelligence/models/fusion.py`

LightGBM over satellite + meteorology + land use, trained **only on station cells**,
predicting all cells.

- **Predicts the deviation from a city baseline, not the absolute level** — a
  construction that cannot lose to the baseline, since a zero residual *is* the
  baseline.
- **Cyclical encoding** — `wind_from_deg`, `hour`, `dow` are fed as sin/cos pairs via
  `_with_cyclical()`, never raw, so wrap-adjacency (23:00 next to 00:00) is true by
  construction. `_with_cyclical` is called inside both `_train` and `_predict` so no
  caller can skip it.
- **NO₂ only** from the satellite. SO₂ and AAI are excluded as measured noise.
- **`loso_validation()`** — leave-one-station-out, retraining a model per held-out
  station. The baseline is rebuilt from the *other* stations only, or LOSO leaks.

Writes `fusion_field.parquet`, `loso.json`, `fusion_model.txt`.

> **Understand this before building on it:** the fusion field is an **exposure map,
> not a source detector**. Stations are sited away from sources, so the model never
> observes one and a tree ensemble cannot extrapolate to one. On real Delhi it is
> *worse* than a naive city-mean. Detection deliberately does not use it. Do not
> "fix" this by feeding the field back into detection.

---

## Layer 4 — Forecast

`intelligence/models/forecast/` — 8 modules, and the only layer with a **split
train/serve lifecycle**.

| Module | Role |
|---|---|
| `features.py` (794 L) | lags, climatology joins, spatial composite, fire pressure, wind vectors |
| `spatial.py` | station composite, positional block, wind-weighted decay |
| `climatology.py` | multi-scale cell → ward → city fallback |
| `model.py` | pooled **quantile** LightGBM (p10/p50/p90) |
| `train.py` | walk-forward folds, event oversampling, final fit |
| `validation.py` | spatial-LOSO, city-LOSO, walk-forward |
| `eval.py` | skill vs persistence + diurnal, coverage, quiet-vs-event |
| `checkpoint.py` | per-fold checkpointing (an HF Job's disk is deleted on exit) |

**Two entry points, and the distinction matters:**
- `run()` — trains, validates, gate-checks, promotes. **~4.3 hours** on the 8-city
  panel. Runs in GitHub Actions → HF Jobs (`.github/workflows/train-forecast.yml`),
  which syncs promoted weights back into `models/`.
- `serve()` — predicts using already-promoted weights. **This is what the pipeline
  calls.** `run_pipeline.py` calls `serve(cities=[CITY])`, deliberately not `run()`.

Writes `forecast.json` (per cell × 24 horizons, 3-hourly to 72h, with `pm25_p10/p90`),
`forecast_ward.json`, `forecast_eval.json`.

Promotion is gated by `models/manifest.json` — a code-enforced comparison against the
prior model's metrics.

> ⚠️ **Known defect:** the p10 lower bound collapses to 0 in **83% of Delhi rows and
> 91% of Chennai** (Bengaluru and Mumbai are clean). `manifest.json` records
> `quantile_coverage: 0.565` against an 0.80 target. The interval is not currently
> trustworthy in those two cities.

---

## Layer 5 — The agent chain

`intelligence/orchestrator.py` — a LangGraph `StateGraph`, with a sequential fallback
that has identical error isolation and an identical result schema if langgraph is not
importable.

**Why a graph when it is a straight line:** per-node error isolation (a broken
forecast must not take down the memo), and *one* definition of chain order shared by
both the pipeline runner and the API.

Agents communicate **through files**, not through graph state. The state carries run
bookkeeping only.

| # | Agent | Reads | Writes |
|---|---|---|---|
| 1 | `detect.py` | panel, fusion_field, osm | `hotspots.json` |
| 2 | `attribution.py` | hotspots, panel, fusion_field, osm, citizen_reports | `attributions.json` |
| 3 | *forecast* | — | **existence check only** |
| 4 | `prioritise.py` | hotspots, attributions, panel | `actions.json`, `dispatch.json` |
| 5 | `memo.py` | hotspots, actions, attributions | `memos.json` |
| 6 | `advisory.py` | wards, panel, forecast, fusion_field | `advisories.json` |
| 7 | `voice.py` | advisories | `audio/*.mp3` + manifest |
| 8 | `ledger.py` | actions, memos, inspection_status | `ledger.json` |
| 9 | `audit.py` | fusion_field, stations, wards | `audit.json` |

**Node 3 is an existence check, not a call into training.** Forecast training used to
sit in this live chain and retrain on every `POST /run/agent`, which pushed a Cloud Run
request past budget and got the container killed mid-chain.

### Node 1 — Detection (`detect.py`)

Runs on **NO₂ contrast + FIRMS fire persistence only** (`POLLUTANTS = ["no2_col"]`).
SO₂ and AAI stay in the panel for exposure and evidence but are excluded from scoring
as measured noise.

- Per-cell **median** over each window, scored against the **annulus** at k=5–10
  (~4–8 km), in robust MAD units. Contrast rather than a citywide rank, because the
  dense urban core is high everywhere and "this district is dense" is not a violator.
  The outer ring must sit outside the satellite's blur or the source contaminates its
  own baseline.
- Multi-window agreement (24h/7d/30d) → `chronic` / `emerging` / `acute`.
- **`_reconcile_zones()`** — cells within `ZONE_LINK_KM = 2.0` are clustered into a
  zone and every cell inherits the zone's most persistent verdict. Persistence is a
  property of a **source**, not a cell: a chronic source's fringe cells only go hot
  when the wind points at them, so classified independently they look `emerging` —
  sending an inspector to find a facility that does not exist.
- **`_mark_attributable()`** — is there anyone to serve a notice on? A named OSM
  candidate within `ATTRIBUTABLE_KM = 0.5`, a FIRMS fire, or a point-tracer contrast.
  A zone high only in NO₂ over dense roads is **diffuse urban background** — real
  pollution, no actor, stays on the map, excluded from the enforcement queue.

> Note `ATTRIBUTABLE_KM` is **0.5 km**, not the 3 km in `architecture.md`. 3 km was
> measured to be a tautology — every hotspot in three real cities sat within 3 km of
> *something*. `prioritise.py` imports this constant rather than redeclaring it.

### Node 2 — Attribution (`attribution.py`)

The pattern every new agent should copy:

```
build_evidence()  →  deterministic category scores  →  complete_json() for PROSE ONLY
                                                    →  rule_based_reason() fallback
```

Evidence: named candidates within 5 km (distance, bearing, wind alignment), pollutant
signature, fire activity, land-use context, meteorology, citizen corroboration.

**Score candidates with MAX, not SUM.** Summing over every nearby OSM site asks "how
many mapped sites of this type are near me" — a question about OSM's *coverage*.
Synthetic had 2 traffic corridors; real Delhi has 1,240, so traffic won by count and
both burning landfills were attributed to traffic.

Confidence is computed from **evidence agreement** — margin, absolute strength, and
count of independent agreeing instruments — never LLM self-report.

### Node 4 — Prioritisation (`prioritise.py`)

```
EPS = 100 × (0.35·severity + 0.25·attribution_conf + 0.20·actionability + 0.20·vulnerability)
```

Deterministic and LLM-free by design. Dispatch is greedy maximum-coverage set cover
(≥63%-of-optimal guarantee) → per-team nearest-neighbour routes via OSRM.

Both `severity` (in `detect.py`) and `vulnerability` are **city-relative** — p95 of
that city's own distribution — rather than fixed constants, so a term tuned once on
Bengaluru does not saturate on Delhi.

### Nodes 5–9

- **Memo** — a deterministic rule engine picks the statute (`eq/in/gte/lte` conditions);
  the LLM writes only connective prose. You cannot let a model choose a legal citation.
- **Advisory** — CPCB's own published NAQI text per band, per language. Every language
  ships a rule-based template, so coverage does not evaporate without an API key.
- **Voice** — Cloud TTS, cached by text hash, never fails the pipeline. Carries
  `text_verification` per language so a machine-written line cannot launder itself into
  sounding official by becoming audio.
- **Ledger** — keeps *response time* (genuinely measured) rigorously apart from
  *intervention effectiveness* (counterfactual, weaker).
- **Audit** — blind spots (dirty cells with no monitor → next-sensor placement) and
  sensor flags (station flat while the field around it is high).

### Cross-cutting — `llm_gateway.py`

Gemini → Groq → None, with hardened JSON parsing and a **circuit breaker**
(`CIRCUIT_TRIP = 2`) so one dead credential doesn't cost hundreds of round-trips.
`reset_circuits()` is called at the start of every chain run, so a long-lived server
isn't disabled until restart.

---

## Layer 6 — Channels

Deliberately dumb: n8n moves bytes, it never scores or reasons.

- `db/schema.sql` — `citizen_reports` (with `device_id`), `inspection_status`.
- `scripts/sync_supabase.py` — pulls both tables down to
  `citizen_reports.json` / `inspection_status.json` before the agents run. Skips
  gracefully without keys; the pipeline must never fail because channels aren't
  configured.

---

## Layer 7 — Serving

`app/backend/main.py` — ~24 read-only endpoints over precomputed contracts.

The interesting part is **multi-city**. `shared.config` binds `DATA_OUT` by value at
import, so one process would serve one city. Rather than edit every endpoint, an HTTP
middleware resolves `?city=` once per request into a `ContextVar`, and the path
helpers read it. Endpoints stay unchanged and *cannot* accidentally read the wrong
city, because they no longer name one. Unknown city → 400; valid city with no output
→ 404 naming the fix — **never another city's data**.

`POST /run/agent` executes the same orchestrator graph. Safe to expose precisely
because every heavy stage was kept out of it.

---

## Layer 8 — Frontend

Next.js 16 + React 19 + deck.gl/MapLibre. Two roles: admin console and citizen view.

**The data flow is the clever bit** — `lib/api.ts::cityFetch()`:
1. If a backend is configured *and* the contract has a live endpoint → try
   `${API_BASE}${live}?city=…`
2. Else (or on failure) → `/data/<city>/<file>.json` from the static bundle
3. Else → the **empty** fallback

Step 3 matters: a missing file yields *empty*, never another city's data. An empty
layer is a visible absence; the wrong city's layer is an invisible lie.

`scripts/export_city_static.py` is the bridge — dumps `data/outputs/<city>/` into
`app/frontend/public/data/<city>/`, converting the fusion parquet to the same JSON
shape `/fusion` returns. This is what makes the demo survive a dead backend.

---

## Running it

```bash
pip install -r requirements.txt
$env:PYTHONPATH = "."                                 # PowerShell

# Offline, reproducible, zero setup
python scripts/run_pipeline.py --synthetic --full

# Live — needs FIRMS_KEY + OPENAQ_API_KEY in .env and GEE auth
$env:AQ_CITY = "delhi"; $env:AQ_WINDOW_END = "2025-11-30"
python scripts/run_pipeline.py --full

# Serve + export
uvicorn app.backend.main:app --reload --port 8000
python scripts/export_city_static.py
```

**`--full` matters.** Without it you get ingest → panel → fusion, and
`hotspots.json` / `attributions.json` are left on disk from a *previous world*. The
API serves them, the map renders them, and nothing says they no longer correspond to
the panel underneath. If you regenerate the panel, regenerate what reads it.

**Evaluation** (the honest numbers): `eval_detection.py` (the headline),
`eval_attribution.py` (accuracy + confidence calibration), `eval_hotspot_recovery.py`,
`eval_station_sensitivity.py`, `compare_cities.py`.

---

## Suggested wiring order

Each step is independently verifiable. Do not move on until the check passes.

| # | Build | Check |
|---|---|---|
| 1 | `config.py` + `grid.py` | `python shared/grid.py` prints cell count |
| 2 | `wards.py` | prints real-boundary vs Voronoi, ward sizes |
| 3 | `synthetic.py` | offline world generates, reproducibly |
| 4 | `pollers.py` (synthetic first) | 5 parquet files in `data/raw/<city>/` |
| 5 | `panel.py` | row count = cells × hours; station-labelled rows > 0 |
| 6 | `fusion.py` | `loso.json` R² is sane |
| 7 | `detect.py` | `hotspots.json` — cells cluster into few zones |
| 8 | `attribution.py` | `attributions.json` has evidence + confidence |
| 9 | `prioritise.py` | `actions.json` EPS ordering is defensible |
| 10 | `memo.py`, `advisory.py`, `voice.py` | each writes its contract |
| 11 | `ledger.py`, `audit.py` | close the chain |
| 12 | `orchestrator.py` | `run_chain()` → 9/9 done |
| 13 | `main.py` | endpoints return the contracts |
| 14 | `export_city_static.py` + frontend | map renders with backend **off** |
| 15 | Live collectors | swap synthetic → real, one source at a time |
| 16 | Forecast training | the long pole — CI/HF Jobs, not local |

Leave forecast training last. It is the only stage measured in hours, it needs its own
CI path, and `serve()` works off committed weights without it.

---

## Things that will mislead you

Collected from the code's own comments — each of these was a real bug.

1. **The fusion field is not a detector.** Trained on station cells, stations sited
   away from sources. Worse than naive city-mean on real Delhi.
2. **SO₂ and AAI are noise.** SNR 0.66–1.03; real S5P SO₂ over a city is 49%
   *negative*. Taking `max()` across one signal and two noise fields means the max is
   usually noise — it flagged 28% of Delhi as hotspots.
3. **The 100% trap.** The project once reported 100% attribution accuracy because the
   synthetic world handed the scorer its answer key. If a number comes back at 100%,
   assume leakage before success.
4. **Never bind a tunable as a default argument.** `def f(k=SOME_GLOBAL)` binds once at
   definition, so `module.SOME_GLOBAL = 0` never reaches it. This made the
   station-sensitivity sweep test k=2 three times while reporting "recall is invariant".
5. **An empty critical source is as dangerous as a failed one, and it does not raise.**
6. **Model an instrument's noise before you model its signal.** The synthetic world's
   *sources* were adversarial while its *sensors* were perfect — the same class of bug
   as the 100% trap, one level deeper.

---

## Current defects (verified this session, not in the docs)

- **Frontend does not build.** `WardAccountability.tsx` and `citizen/explore/page.tsx`
  import `confidenceLabel` and `ACTION_STATUS_*`, which do not exist in
  `lib/constants.ts` / `lib/colors.ts`; `useReports.ts` calls `api.getReports(deviceId)`
  against a 0-arg signature. 7 TypeScript errors, all in **uncommitted local work** —
  HEAD itself is clean. Hitting `/citizen/[wardId]` or `/citizen/explore` 500s **and
  poisons every other route** until the dev server restarts.
- **28% of enforcement actions** (13 of 46 across all cities) point at ward "Outside
  city limits" — Pune's entire queue is 2 of 2.
- **Forecast p10 collapses to 0** in Delhi (83%) and Chennai (91%).
- **`forecast_eval.json` exists for only 3 of 8 cities** — handled deliberately on the
  Validation page ("not scored out of sample"), not a bug.
- **Python env is incomplete here** — `lightgbm`, `h3`, `fastapi`, `pytest` are not
  installed, so the backend, pipeline and tests cannot run on this machine.
