"""Quantile forecaster package (spec:
docs/superpowers/specs/2026-08-15-forecast-rework-design.md). Public API:
run(), evaluate(), HORIZONS, TEST_TAIL_DAYS — kept import-compatible with
the old single-file forecast.py so scripts/run_pipeline.py and
scripts/eval_forecast_live.py need no changes beyond what Task 11 already
makes."""
import json

import pandas as pd

from shared.config import ROOT
from intelligence.models.forecast.features import build_features, FEATURE_COLUMNS
from intelligence.models.forecast.model import predict_quantiles, mask_unknown_city, UNKNOWN_CITY
from intelligence.models.forecast.train import train_and_promote
from intelligence.models.forecast.eval import skill_vs_baseline

HORIZONS = list(range(3, 73, 3))
TEST_TAIL_DAYS = 14


def evaluate(panel: pd.DataFrame, oracle_met: bool = False) -> dict:
    """Kept for scripts/eval_forecast_live.py's existing import. `oracle_met`
    is no longer supported by the new model (the old model's leakage-sizing
    experiment doesn't apply to a quantile/multi-city model) — always
    ignored, and the caller gets an explicit note rather than a silent gap."""
    frame = mask_unknown_city(build_features(panel, HORIZONS))
    split = frame.ts.max() - pd.Timedelta(days=TEST_TAIL_DAYS)
    train_frame, test_frame = frame[frame.ts <= split], frame[frame.ts > split].dropna(subset=["y"])
    results = {}
    for h in HORIZONS:
        te_h = test_frame[test_frame.horizon == h]
        if te_h.empty or train_frame.empty:
            results[f"h{h}"] = {"n_test": 0, "note": "insufficient data"}
            continue
        from intelligence.models.forecast.model import train_quantile_models
        models = train_quantile_models(train_frame, FEATURE_COLUMNS, num_boost_round=200)
        pred = predict_quantiles(models, te_h, FEATURE_COLUMNS)
        results[f"h{h}"] = {
            "n_test": len(te_h),
            "skill_vs_persistence_pct": skill_vs_baseline(te_h["y"].values, pred["pm25_p50"].values, te_h["lag_0"].values),
        }
    if oracle_met:
        results["_note"] = "oracle_met is not supported by the quantile/multi-city model; ignored"
    return results


def _predict_field(panels: dict, served_manifest: dict, served_models: dict,
                    feature_cols: list[str]) -> dict[str, list[dict]]:
    """Per-city forecast.json field, predicted with the SERVED model.

    Uses `served_manifest["cities"]` (the cities the served model was
    ACTUALLY trained on — train_and_promote already writes this) for the
    categorical reconstruction, NOT `panels.keys()` (the current run's
    data). These differ whenever a refused promotion falls back to a prior
    model trained on a different city set — exactly the scenario this
    session's own city-registry expansion (3 -> 8 cities) makes concrete.
    Using the wrong source silently corrupts LightGBM's integer-coded
    categorical predictions rather than raising: get this wrong and a
    city's forecast quietly reflects a DIFFERENT city's learned behaviour.
    """
    city_categories = sorted(set(served_manifest["cities"]) | {UNKNOWN_CITY})
    fields = {}
    for city, panel in panels.items():
        frame = build_features(panel, HORIZONS)
        frame["city"] = pd.Categorical(frame["city"], categories=city_categories)
        latest = frame.ts.max()
        latest_rows = frame[frame.ts == latest].dropna(subset=["lag_0"])
        if latest_rows.empty:
            continue
        pred = predict_quantiles(served_models, latest_rows, feature_cols)
        fields[city] = [{"cell": c, "horizon_h": int(h), "pm25_hat": round(float(p50), 1),
                         "pm25_p10": round(float(p10), 1), "pm25_p90": round(float(p90), 1)}
                        for c, h, p50, p10, p90 in zip(latest_rows.cell, latest_rows.horizon,
                                                        pred.pm25_p50, pred.pm25_p10, pred.pm25_p90)]
    return fields


def run(cities: list[str] | None = None) -> dict:
    """New entrypoint. Trains on each city's historical panel if
    data/historical/<city>/panel.parquet exists (richer, longer window),
    else falls back to that city's operational data/outputs/<city>/panel.parquet
    (matches the old model's data source, for a city with no backfill yet)."""
    from shared.config import CITIES, DATA_OUT_BASE
    cities = cities or list(CITIES)
    panels = {}
    for city in cities:
        hist_panel = ROOT / "data" / "historical" / city / "panel.parquet"
        op_panel = DATA_OUT_BASE / city / "panel.parquet"
        path = hist_panel if hist_panel.exists() else op_panel
        if not path.exists():
            print(f"[forecast] {city}: no panel at {path}, skipping")
            continue
        panels[city] = pd.read_parquet(path)

    if not panels:
        raise RuntimeError("no city panels available — run the operational pipeline "
                            "or the historical backfill first")

    out_dir = ROOT / "data" / "outputs" / "_forecast_models"
    prior = None
    manifest_p = out_dir / "manifest.json"
    if manifest_p.exists():
        try:
            prior = json.loads(manifest_p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            # A prior manifest that exists but can't be read is as dangerous
            # as a failed run if silently ignored -- log it loudly rather
            # than let a corrupted file quietly reset promotion history.
            print(f"[forecast] prior manifest at {manifest_p} is unreadable "
                  f"({type(e).__name__}: {e}) — treating as no prior model")
            prior = None

    manifest = train_and_promote(panels, HORIZONS, FEATURE_COLUMNS, out_dir, prior_manifest=prior)

    # forecast.json must reflect whichever model is ACTUALLY served now —
    # NOT a third model trained separately here. train_and_promote already
    # trained and gate-checked one model; if it was refused (the new model
    # regressed), the PREVIOUSLY promoted model (from `prior`) is what is
    # really still served. Training yet another fresh model inline in this
    # loop (an earlier draft did exactly this) would silently bypass the
    # entire promotion gate: forecast.json would then always reflect
    # whatever was most recently trained, regardless of whether the gate
    # approved it — turning the gate into decoration.
    served_manifest = manifest if manifest["promoted"] else prior
    if served_manifest is None:
        print("[forecast] no promoted model available (this run was refused and no prior "
              "model exists) — skipping forecast.json")
        return manifest

    import lightgbm as lgb
    model_dir = out_dir / served_manifest["version"]
    try:
        served_models = {
            0.1: lgb.Booster(model_file=str(model_dir / "model_p10.txt")),
            0.5: lgb.Booster(model_file=str(model_dir / "model_p50.txt")),
            0.9: lgb.Booster(model_file=str(model_dir / "model_p90.txt")),
        }
    except Exception as e:   # noqa: BLE001 — a missing/corrupted model dir must not crash the whole run
        print(f"[forecast] served model at {model_dir} failed to load "
              f"({type(e).__name__}: {e}) — skipping forecast.json this run")
        return manifest

    # DATA_OUT_BASE / city, NOT the singular DATA_OUT: DATA_OUT is bound at
    # process-import time to whichever single AQ_CITY started this process
    # (see Global Constraints). This loop writes several cities' outputs in
    # one process — using the bare DATA_OUT here would silently write every
    # city's forecast.json into whichever one city happened to be active,
    # exactly the bug class this plan warns about elsewhere.
    fields = _predict_field(panels, served_manifest, served_models, FEATURE_COLUMNS)
    for city, field in fields.items():
        city_out = DATA_OUT_BASE / city
        city_out.mkdir(parents=True, exist_ok=True)
        (city_out / "forecast.json").write_text(json.dumps(field, separators=(",", ":")))
        (city_out / "forecast_eval.json").write_text(json.dumps(manifest["eval"], indent=2))
        print(f"[forecast] {city}: wrote forecast.json ({len(field)} cell-horizons)")

    return manifest
