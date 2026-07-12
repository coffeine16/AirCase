# AQ Intelligence Platform

AI-powered urban air quality intelligence: signal -> attribution -> action.
From AQI dashboards to enforcement dispatch — names *who* is polluting,
*where*, with *what evidence*, and *what to do about it today*.

Full design doc: @./docs/architecture.md — read it before making architectural
changes. This file is operational context; that file is the source of truth
for *why* things are shaped the way they are.

## Monorepo layout (one deployable unit per top-level folder)
- `app/` — presentation: FastAPI serving layer (`app/backend/main.py`) +
  frontend (React+Vite+Leaflet, not yet built)
- `ingestion/` — data platform: `collectors/` (6 sources, live API + synthetic
  fallback per source) + `preprocessing/panel.py` (cell x hour feature table)
  + `synthetic.py` (adversarial hidden-source world for offline dev + truth-scored eval)
- `intelligence/` — `models/fusion.py` (LightGBM exposure field) +
  `models/signals.py` (robust multi-window statistics) + `agents/`
  (detect.py, attribution.py, llm_gateway.py)
- `shared/` — `config.py` (city bbox, H3 res, windows, paths) + `grid.py` (H3
  fabric, bearings, wind alignment) + `wards.py` (ward layer)
- `scripts/` — pipeline runner + evaluation scripts
- `docs/` — architecture.md (the 9-layer design), evaluation notes

## Non-negotiable design principles (do not violate silently)
1. **Deterministic arithmetic ranks; LLMs only explain.** Category scores,
   priority scores, and any ranking must be plain reproducible math. Never let
   an LLM call decide a score or ranking — it may only generate prose that
   explains a score already computed. If an LLM's answer conflicts with the
   arithmetic result, the arithmetic wins and the LLM output is discarded
   (see `intelligence/agents/attribution.py::attribute_one` for the pattern).
2. **Every LLM-dependent path needs a deterministic rule-based fallback.**
   No feature may go silent or crash because an API key is missing or a
   provider rate-limits. See `intelligence/agents/llm_gateway.py` (Gemini ->
   Groq -> None) and `attribution.py::rule_based_reason` for the pattern.
3. **Heavy compute in batch, serving stays read-only.** `app/backend/main.py`
   only reads precomputed files from `data/outputs/`. Never add live model
   inference to a request handler.
4. **Every evidence-based claim ships its evidence chain + confidence, and the
   confidence must actually discriminate.** Computed from evidence agreement,
   never LLM self-report — and *validated against truth*. A confidence that
   reads the same on hits and misses is decoration; ours previously did
   (0.74 vs 0.75) and had to be rebuilt. It is now margin + absolute evidence
   strength + count of independent agreeing instruments.
5. **Channels are dumb, agents are smart.** Messaging infra only moves bytes;
   no scoring or reasoning belongs there.
6. **Robust statistics only — never the mean.** Median for every aggregate, MAD
   for every spread. The outliers in this domain *are* the phenomenon: a mean
   lets one spike hour manufacture a chronic source out of one bonfire.
7. **One window is not a signal.** Every source claim is aggregated over
   24h/7d/30d and must survive the zoom-out. See `models/signals.py`.
8. **The evaluation must be able to fail.** Never let the synthetic generator
   hand the scorer its answer key. See "the 100% trap" below.

## Commands
```bash
pip install -r requirements.txt
$env:PYTHONPATH = "."                                    # PowerShell
python scripts/run_pipeline.py --synthetic --full        # ingest -> panel -> fusion -> detect -> attribute
python scripts/run_pipeline.py --synthetic               # fusion stage only (leaves hotspots.json STALE)
python intelligence/agents/detect.py                     # satellite+fire zone detection
python intelligence/agents/attribution.py                # source attribution
python scripts/eval_detection.py                         # THE headline stat (recall/precision vs truth)
python scripts/eval_attribution.py                       # attribution accuracy + confidence calibration
python scripts/eval_hotspot_recovery.py                  # fusion as an EXPOSURE map (not a detector)
uvicorn app.backend.main:app --reload --port 8000        # serving API
```
Drop `--synthetic` once real API keys are in `.env` (copy from `.env.example`).
`shared/config.py` loads `.env` itself — no python-dotenv dependency.

## The two things that will mislead you if you don't know them

**1. The fusion field is an EXPOSURE map, not a source detector.**
It trains only on station cells, and stations are sited *away* from sources — so
it never observes a source, cannot learn a source response, and (tree ensemble)
cannot extrapolate to one. Measured: training stations see a mean source
contribution of 0.25 ug/m3 while the city reaches 210. Its field is
background-dominated and it is *worse than naive* on the dirtiest decile.
Detection runs on **satellite contrast + FIRMS fire persistence** instead
(`agents/detect.py`), which are the only layers with uniform coverage.
Do not "fix" this by feeding the fusion field back into detection.

**2. The 100% trap.** This project once reported *100% attribution accuracy*.
It was an artefact: the synthetic world emitted its hidden sources straight into
the OSM layer with exact coordinates and exact category labels, and used the
same `exp(-d/2)*wind_alignment` kernel the attribution scorer uses. The scorer
was handed the answer key. `ingestion/synthetic.py` is now deliberately
adversarial — different dispersion physics, sources that appear on no map, decoy
sites that emit nothing, a satellite blurred to its true ~5.5 km footprint, and
column-vs-surface decoupling. If a number ever comes back at 100%, assume
leakage before you assume success.

## Current state (keep this section updated as phases land)
- ✅ Ingestion (6 sources, live+fallback) + H3 fabric + ward layer
  (`shared/wards.py`: real GeoJSON if present, deterministic Voronoi otherwise).
- ✅ Fusion **exposure** field. LOSO R2 ~0.90, RMSE ~6.4 vs naive station-mean
  9.99 (~36% better citywide). Not a detector — see above.
- ✅ Detection (`agents/detect.py`): satellite neighbourhood contrast + fire
  persistence, robust median/MAD, multi-window 24h/7d/30d ->
  chronic / emerging / acute.
- ✅ Attribution: deterministic category scores, LLM explains only, rule
  fallback. Confidence now discriminates (hits 0.66 / misses 0.42).
- ✅ Zones + enforceable/diffuse split (`detect.py`). Hotspot CELLS are clustered
  into source ZONES (an inspector is dispatched to a zone, not a 460 m hexagon),
  and each zone is tagged `attributable`: does any instrument point at a *place*
  (OSM candidate <3 km, FIRMS fire, or SO2/AAI point-tracer contrast)? A cell that
  is only high in NO2 over dense roads is **diffuse urban background** — real
  pollution, no one to serve a notice on, a policy target. It stays on the map and
  is excluded from the enforcement queue.
- 📊 Headline numbers (synthetic, `scripts/eval_detection.py`):
  - **4/4** physically observable sources detected and correctly named,
    including **2/2 that appear on no map at all**
  - enforceable-**zone** precision **4/4**; attribution accuracy **92%** (100% on
    unregistered); precision **100%** at confidence >= 0.70
  - **0 of 9** sources sit within 2 km of a monitor — the coverage-bias number
  - ⚠️ 4/4 is **n=4**. Per "the 100% trap" below, do not sell it as a robust rate.
    The conservative companion is cell-level precision **77%**, whose shortfall is
    *plume extent inside correctly-found zones* (median 2.5 km from the source that
    produced it), not false accusations. Quote both.
- ⚠️ **Known blind spots, stated not hidden:** construction dust (coarse PM, no
  satellite tracer, doesn't burn) recall **0/3**; traffic corridors (NO2
  confounded with the diffuse urban road network) recall **0/2**. Closing these
  needs the OSM permit layer, PM10/PM2.5 coarse fraction, and citizen reports.
- ⬜ NOT built: forecast agent, prioritisation/dispatch (EPS + routing), memo
  agent (legal matcher), advisory agent, frontend, n8n channels, GEE Sentinel-5P
  collector (satellite is synthetic-only today — live mode silently mixes real
  stations with a synthetic satellite, which is scientifically invalid).

## Conventions
- Python 3.11, type hints on function signatures, no framework beyond what's
  already imported per file (pandas/numpy/lightgbm/h3/fastapi).
- New agents under `intelligence/agents/` follow the attribution.py shape:
  a `build_evidence()` function, a deterministic scoring function, an LLM
  call via `llm_gateway.complete_json()`, and a rule-based fallback with the
  identical output schema.
- Outputs are written to `data/outputs/*.json` (frontend/API contracts) or
  `*.parquet` (large tabular data consumed by other Python code, not the API).
- `data/raw/` and `data/outputs/` are gitignored — regenerated by running
  scripts, never hand-edited or committed.
- Commit messages: conventional format (`feat(scope):`, `fix:`, `refactor:`,
  `docs:`, `ci:`) with a body explaining *why*, not just *what*.

## Known gotchas
- **Persistence is a property of a SOURCE, not of a cell.** A chronic source's
  fringe cells only go hot when the wind points at them, so over 30 days they look
  intermittent and classify as `emerging` — which would send an inspector hunting
  for a newly commissioned facility that does not exist. (Measured: the lone
  `emerging` and `acute` flags in one run were both fringe cells 2.0 km from the
  chronic landfill.) `detect.py::_reconcile_zones` clusters cells within 2 km and
  makes every cell inherit its zone's most persistent verdict. Do not classify
  cells independently.
- Windows PowerShell: use `$env:PYTHONPATH = "."` not `PYTHONPATH=. cmd`, and
  `Remove-Item -Recurse -Force` not `rm -rf`.
- `fusion.py` LOSO retrains 12 LightGBM models (one per held-out station).
- The panel is 60 days x 1210 cells = 1.74M rows. `_fire_features` is vectorised
  for exactly this reason; do not reintroduce a per-cell/per-hour Python loop.
- The synthetic world has 9 hidden sources around a Bengaluru bbox. If you change
  `config.BBOX`, regenerate the sources to match or switch fully to live mode.
- `pd.DatetimeIndex.values` strips the timezone and will silently break merges
  against the panel. Use `.repeat()` on the index itself.
