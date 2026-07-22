"""Read-only serving API — the app-layer backend.

Heavy compute happens in the batch pipeline; this only reads precomputed
contracts from data/outputs/. Run:

    uvicorn app.backend.main:app --reload --port 8000

Endpoints:
    GET /health
    GET /hotspots                latest ranked hotspots (anomaly + chronic)
    GET /attribution/{cell}      full evidence chain for one hotspot
    GET /attributions            all attributions (summary fields)
    GET /fusion?hour_offset=0    fusion field snapshot (0 = latest hour)
    GET /wards                   cell -> ward map (real boundaries or fallback)
    GET /loso                    fusion validation metrics
"""
import json
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from shared.config import DATA_OUT, DATA_RAW

app = FastAPI(
    title="AirCase API",
    version="0.2",
    description=(
        "Read-only serving layer over the batch pipeline's precomputed contracts.\n\n"
        "**Multi-city.** One instance serves every city whose pipeline has run. "
        "Pass `?city=` on any endpoint to choose; omit it for the instance "
        "default. `GET /health` lists which cities this instance can actually "
        "answer for. An unknown city returns 400; a city with no pipeline output "
        "returns 404 naming the fix — never another city's data.\n\n"
        "**Heavy compute is not here.** Ingestion, the panel build and fusion "
        "training run offline in batch. `POST /run/agent` executes the 9-agent "
        "chain over artifacts those stages already produced, which is why it "
        "fits inside a request."
    ),
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Voice advisories (MP3) are static files written by the batch pipeline.
# Mount the directory so the frontend can play them directly.
from fastapi.staticfiles import StaticFiles
(DATA_OUT / "audio").mkdir(parents=True, exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(DATA_OUT / "audio")), name="audio")


# ── Per-request city scope ────────────────────────────────────────────────────
# shared.config binds DATA_OUT/DATA_RAW by VALUE at import, so one process serves
# one city — right for the batch pipeline, wrong for a server that must answer for
# three. Rather than edit 23 endpoints, a middleware resolves ?city= once per
# request into a ContextVar and the path helpers read it. Endpoints stay unchanged
# and CANNOT accidentally read the wrong city, because they no longer name one.
from contextvars import ContextVar  # noqa: E402
from typing import Literal  # noqa: E402
from shared.config import CITIES, CITY, DATA_OUT_BASE, DATA_RAW_BASE  # noqa: E402

_req_city: ContextVar[str] = ContextVar("req_city", default=CITY)


@app.middleware("http")
async def _city_scope(request, call_next):
    want = (request.query_params.get("city") or CITY).lower()
    if want not in CITIES:
        return JSONResponse({"detail": f"unknown city {want!r}; choose from {sorted(CITIES)}"},
                            status_code=400)
    token = _req_city.set(want)
    try:
        return await call_next(request)
    finally:
        _req_city.reset(token)


def city_param(
    city: str | None = Query(
        default=None,
        # An ENUM in the schema so /docs renders a dropdown, but the annotation
        # stays `str`. Declaring it as a Literal made FastAPI validate before the
        # middleware ran, and `?city=DELHI` — which the middleware has always
        # accepted, since it lowercases — started returning 422. Documenting a
        # parameter must not narrow what it accepts.
        json_schema_extra={"enum": sorted(CITIES)},
        description=(
            "Which city to serve. Every contract is stored per city and this "
            "selects among them. Omit to use the instance default "
            f"({CITY}). Available: {', '.join(sorted(CITIES))}."
        ),
        examples=[CITY],
    ),
) -> str:
    """DECLARED so the parameter is visible, not just functional.

    The middleware above already resolves ?city= for every request, and that is
    what the path helpers read. But FastAPI builds /docs and /openapi.json from
    function SIGNATURES, and no endpoint named `city` — so the schema advertised
    `parameters: NONE` on all 23 routes and the API looked single-city to anyone
    reading its documentation. Two reviewers read it that way.

    Declaring it as a dependency on every route puts it in the schema (with an
    enum, so /docs renders a dropdown) without moving the resolution logic out of
    the middleware. An API that works but does not document its main capability
    is indistinguishable, to a reader, from one that lacks it.
    """
    return _req_city.get()


def _city() -> str:
    return _req_city.get()


def _city_out() -> Path:
    """Output tree for THIS request's city."""
    return DATA_OUT_BASE / _city()


def _city_raw() -> Path:
    return DATA_RAW_BASE / _city()


def _json(name: str):
    """Read a precomputed contract for this request's city. UTF-8 EXPLICITLY.

    read_text() uses the platform default, which on Windows is cp1252 — it raises
    UnicodeDecodeError the moment an advisory contains Kannada or Hindi. The citizen
    endpoints are the whole point of the language-coverage work, so this is not
    optional.
    """
    p = _city_out() / name
    if not p.exists():
        raise HTTPException(
            404, f"{name} not generated for {_city()} — run the pipeline for that city")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/health", dependencies=[Depends(city_param)])
def health():
    """Liveness + WHICH CITY this instance actually serves.

    Data is per-city on disk, so report which cities this instance can actually
    answer for — the UI should offer a live run only for those, rather than
    discovering the gap as a 404.
    """
    available = sorted(
        c for c in CITIES
        if (DATA_OUT_BASE / c / "hotspots.json").exists()
    )
    return {
        "ok": True,
        "city": _city(),               # this request's scope
        "default_city": CITY,          # what an unqualified request gets
        "cities_available": available,  # cities with pipeline output on disk
        "outputs_present": sorted(p.name for p in _city_out().glob("*")),
    }


@app.get("/hotspots", dependencies=[Depends(city_param)])
def hotspots():
    return _json("hotspots.json")


@app.get("/attributions", dependencies=[Depends(city_param)])
def attributions():
    return [{k: a[k] for k in ("cell", "ward_id", "ts", "primary_source", "confidence", "reason")}
            for a in _json("attributions.json")]


@app.get("/wards", dependencies=[Depends(city_param)])
def wards():
    """cell -> ward map, as WRITTEN BY THE PIPELINE.

    Never recomputed here. The API cannot know which city the data on disk belongs
    to — config.CITY reads an env var the server process may not have — and guessing
    wrong silently maps Delhi cells through a Bengaluru ward table, yielding NaN for
    every ward and a 500 on any endpoint that joins against it.
    """
    return _json("wards.json")


@app.get("/attribution/{cell}", dependencies=[Depends(city_param)])
def attribution(cell: str):
    for a in _json("attributions.json"):
        if a["cell"] == cell:
            return a
    raise HTTPException(404, f"no attribution for cell {cell}")


@app.get("/fusion", dependencies=[Depends(city_param)])
def fusion(hour_offset: int = 0):
    p = _city_out() / "fusion_field.parquet"
    if not p.exists():
        raise HTTPException(404, "fusion field not generated yet")
    f = pd.read_parquet(p)
    f["ts"] = pd.to_datetime(f.ts, utc=True)
    hours = sorted(f.ts.unique())
    if not 0 <= hour_offset < len(hours):
        raise HTTPException(400, f"hour_offset must be in [0, {len(hours) - 1}] "
                                 f"(0 = latest hour); got {hour_offset}")
    at = hours[-1 - hour_offset]
    wmap = {c["cell"]: c["ward_id"] for c in _json("wards.json")["cells"]}
    snap = f[f.ts == at].dropna(subset=["pm25_hat"])   # NaN is not valid JSON
    return {"ts": str(at), "n_hours": len(hours),
            "cells": [{"cell": r.cell, "ward_id": wmap.get(r.cell, "unassigned"),
                       "pm25": round(float(r.pm25_hat), 1)}
                      for r in snap.itertuples()]}


@app.get("/loso", dependencies=[Depends(city_param)])
def loso():
    return _json("loso.json")


@app.get("/stations", dependencies=[Depends(city_param)])
def stations():
    """CAAQMS station locations + their latest reading. Feeds the map's station markers."""
    p = _city_raw() / "stations.parquet"
    if not p.exists():
        raise HTTPException(404, "stations not ingested yet — run the pipeline first")
    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df.ts, utc=True)
    # Real CPCB stations drop readings — NaN is NOT valid JSON and FastAPI raises a
    # 500 on it. Take each station's latest VALID reading, not its latest row.
    df = df.dropna(subset=["pm25"])
    if df.empty:
        return []
    latest = df.sort_values("ts").groupby("station_id").last().reset_index()
    return [{"station_id": r.station_id, "cell": r.cell,
             "lat": float(r.lat), "lon": float(r.lon),
             "pm25": round(float(r.pm25), 1), "ts": str(r.ts)}
            for r in latest.itertuples()]


@app.get("/satellite", dependencies=[Depends(city_param)])
def satellite():
    """Per-cell Sentinel-5P NO2 tropospheric column (median over the window).

    NO2 ONLY. SO2 and the aerosol index are in the raw file but are measured NOISE
    over a city (SNR < 1, 49% of SO2 negative) — see docs/architecture.md. Serving
    them would invite the map to render retrieval error as signal. Detection reads
    exactly this NO2 field; showing it explains WHY a hotspot fired.
    """
    p = _city_raw() / "satellite.parquet"
    if not p.exists():
        raise HTTPException(404, "satellite not ingested yet — run the pipeline first")
    df = pd.read_parquet(p)
    per_cell = df.groupby("cell")["no2_col"].median().dropna()
    if per_cell.empty:
        return []
    return [{"cell": cell, "no2": round(float(v), 2)}
            for cell, v in per_cell.items()]


@app.get("/forecast", dependencies=[Depends(city_param)])
def forecast(h: int | None = None):
    """PM2.5 forecast per cell. ?h=24|48|72 filters to one horizon; omit for all.

    Precomputed in batch (principle 3). Each row: {cell, horizon_h, pm25_hat, urgency}.
    """
    field = _json("forecast.json")
    if h is not None:
        field = [f for f in field if f["horizon_h"] == h]
    return field


@app.get("/forecast/eval", dependencies=[Depends(city_param)])
def forecast_eval():
    """RMSE at each horizon vs persistence + diurnal baselines. The rubric's number."""
    return _json("forecast_eval.json")


@app.get("/voice/{ward_id}", dependencies=[Depends(city_param)])
def voice(ward_id: str, lang: str = "en"):
    """Audio path for a ward's advisory in `lang`, plus how the TEXT was verified.

    Serves what the pipeline precomputed (principle 3). The `text_verification`
    field is carried through deliberately: audio FEELS more authoritative than
    text, so a `cross_checked` (not native-reviewed) advisory must not launder
    that fact by becoming a voice note. The frontend should surface it.
    """
    manifest_p = _city_out() / "audio" / "manifest.json"
    if not manifest_p.exists():
        raise HTTPException(404, "no voice audio generated yet — run the pipeline")
    for e in json.loads(manifest_p.read_text(encoding="utf-8")):
        if e["ward_id"] == ward_id and e["lang"] == lang:
            return {"ward_id": ward_id, "lang": lang,
                    "url": "/" + e["path"], "voice": e["voice"],
                    "text_verification": e.get("text_verification")}
    raise HTTPException(404, f"no {lang} audio for ward {ward_id}")


@app.get("/advisories", dependencies=[Depends(city_param)])
def advisories():
    """Ward health advisories, ranked by risk. Multi-language."""
    return _json("advisories.json")


@app.get("/ward/{ward_id}/summary", dependencies=[Depends(city_param)])
def ward_summary(ward_id: str):
    """The citizen view: my ward's air, my forecast, my advisory in my language."""
    for a in _json("advisories.json"):
        if a["ward_id"] == ward_id:
            return a
    raise HTTPException(404, f"no advisory for ward {ward_id}")


@app.post("/run/agent", dependencies=[Depends(city_param)])
def run_agent(body: dict):
    """Execute the agent chain (or one agent) via the LangGraph orchestrator.

    The ONE sanctioned exception to "serving never computes" (the architecture
    doc's Layer 8 trigger endpoint, in the request/response form the frontend's
    useAgentRun.ts actually calls). Safe because the graph covers AGENTS ONLY —
    deterministic re-scoring of artifacts already on disk (~seconds). Ingestion,
    the panel and fusion/LOSO are unreachable from here by construction.

    Optional `dispatch_config` in body: {n_teams: int, stop_budget: int} —
    forwarded to the prioritisation agent.
    """
    from intelligence.orchestrator import AGENT_NAMES
    agent = (body or {}).get("agent", "all")
    if agent != "all" and agent not in AGENT_NAMES:
        raise HTTPException(400, f"unknown agent {agent!r}; choose from {AGENT_NAMES} or 'all'")

    # Read once, here, so BOTH execution paths below get it. It was previously
    # parsed after the in-process `return`, which made it unreachable on the fast
    # path and absent from the subprocess path — so a dispatch config sent from
    # the console was silently ignored on every city.
    dispatch_config = (body or {}).get("dispatch_config") or None

    city = _city()
    if not (DATA_OUT_BASE / city / "panel.parquet").exists():
        raise HTTPException(404, f"no pipeline artifacts for {city} — the agents read a "
                                 f"precomputed panel, which this deployment does not carry")

    # Fast path: this process was started for this city, so DATA_OUT already
    # points at the right tree.
    if city == CITY:
        from intelligence.orchestrator import run_chain
        return run_chain(agent, dispatch_config=dispatch_config)

    # Another city. shared.config binds DATA_OUT at IMPORT, so we cannot simply
    # set an env var and call run_chain — the agents already captured the wrong
    # paths. Mutating module state under a live server is worse: it races other
    # in-flight requests. Run the chain in a child process with AQ_CITY set, which
    # is correct by isolation. Costs one interpreter start (~4s) on top of ~15s.
    import os
    import subprocess
    import sys
    import tempfile
    from shared.config import ROOT

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "result.json"
        # dispatch_config travels as JSON in argv rather than as another env var:
        # it is structured, and json.loads on the far side keeps the contract
        # identical to the in-process call above.
        child = (
            "import json,sys;"
            "from intelligence.orchestrator import run_chain;"
            "cfg=json.loads(sys.argv[3]) or None;"
            "r=run_chain(sys.argv[1], dispatch_config=cfg);"
            "open(sys.argv[2],'w',encoding='utf-8').write(json.dumps(r,default=str))"
        )
        env = {**os.environ, "AQ_CITY": city, "PYTHONPATH": str(ROOT)}
        proc = subprocess.run(
            [sys.executable, "-c", child, agent, str(out), json.dumps(dispatch_config)],
            cwd=str(ROOT), env=env, capture_output=True, timeout=240)
        if not out.exists():
            tail = (proc.stderr or b"").decode(errors="replace")[-400:]
            raise HTTPException(500, f"agent run for {city} failed: {tail}")
        return json.loads(out.read_text(encoding="utf-8"))


@app.get("/ledger", dependencies=[Depends(city_param)])
def ledger():
    """Intervention ledger: response-time reduction (real) + frozen counterfactuals
    awaiting real outcomes. See intelligence/agents/ledger.py for the honesty split."""
    return _json("ledger.json")


@app.get("/memos", dependencies=[Depends(city_param)])
def memos():
    """All drafted enforcement memos (the EPS queue turned into documents)."""
    return _json("memos.json")


@app.get("/memo/{action_id}", dependencies=[Depends(city_param)])
def memo(action_id: str):
    """One memo, with its evidence chain and its legal basis.

    GET, not POST: the memo is PRECOMPUTED by the batch pipeline, like everything
    else this API serves (principle 3 — serving never computes). The frontend's
    "Generate memo" button fetches a document that already exists; it does not
    trigger an LLM call inside a request handler.
    """
    for x in _json("memos.json"):
        if action_id in (x.get("action_id"), x.get("zone_id"), x.get("memo_id")):
            return x
    raise HTTPException(404, f"no memo for {action_id}")


@app.get("/fires", dependencies=[Depends(city_param)])
def fires():
    """FIRMS thermal detections in the window. The instrument that finds burning
    sources directly — and the one that located Bhalswa."""
    p = _city_raw() / "fires.parquet"
    if not p.exists():
        raise HTTPException(404, "fires not ingested yet — run the pipeline first")
    df = pd.read_parquet(p)
    if df.empty:
        return []
    df["ts"] = pd.to_datetime(df.ts, utc=True)
    return [{"lat": float(r.lat), "lon": float(r.lon),
             "frp": round(float(r.frp), 2), "confidence": str(r.confidence),
             "ts": str(r.ts)}
            for r in df.sort_values("ts").itertuples()]

@app.get("/actions", dependencies=[Depends(city_param)])
def actions():
    """Ranked enforceable hotspots."""
    return _json("actions.json")
@app.get("/dispatch", dependencies=[Depends(city_param)])
def dispatch():
    """Team routes for maximum-coverage dispatch."""
    return _json("dispatch.json")
@app.get("/compare")
def compare():
    """Multi-city comparison rows. Each row is one FULL LIVE pipeline run over a
    real city (real Sentinel-5P / FIRMS / OpenAQ / OSM), distilled by
    scripts/city_summary.py. Returns [] until at least one city has been run."""
    p = DATA_OUT_BASE / "city_comparison.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))
@app.get("/audit", dependencies=[Depends(city_param)])
def audit():
    """Monitoring network audit (F4): blind spots (dirty unmonitored cells ->
    next-sensor placement) + sensor flags. Empty envelope until the pipeline runs."""
    p = _city_out() / "audit.json"
    if not p.exists():
        return {"blind_spots": [], "sensor_flags": [], "placement_recommendations": []}
    return json.loads(p.read_text(encoding="utf-8"))