"""Quantile forecaster package (spec:
docs/superpowers/specs/2026-08-15-forecast-rework-design.md). Public API:
serve(), run(), evaluate(), HORIZONS, TEST_TAIL_DAYS — kept import-compatible
with the old single-file forecast.py so scripts/run_pipeline.py and
scripts/eval_forecast_live.py need no changes beyond what Task 11 already
makes.

TRAINING AND SERVING ARE TWO DIFFERENT ENTRYPOINTS, ON PURPOSE

    serve()  — load the promoted weights, predict, write the JSON. Seconds.
    run()    — train + gate-check + promote, THEN serve. Hours.

`run()` used to be the only entrypoint, and scripts/run_pipeline.py --full
called it. That made the routine "regenerate the city's outputs" command
retrain every model from scratch: train_and_promote has no skip-if-fresh
guard (by design — a training run should always train), and its own timing
comment records 4.3h for the real 8-city run. Neither the operational
pipeline nor CI can afford that, and neither should have to: a pipeline run
must not fail because a TRAINING run failed.

Training already has a home: .github/workflows/train-forecast.yml, which
runs it on HF Jobs and syncs the promoted weights back into models/. That is
the only thing that should be paying for training. Everything downstream
just serves what was promoted, which is the whole point of persisting
weights in the first place."""
import json

import numpy as np
import pandas as pd

from shared.config import ROOT
from intelligence.models.forecast.features import (
    build_features, attach_climatology, downcast_panel, FEATURE_COLUMNS,
)
from intelligence.models.forecast.climatology import build_climatology
from intelligence.models.forecast.model import (
    predict_quantiles, train_quantile_models, mask_unknown_city, UNKNOWN_CITY,
)
from intelligence.models.forecast.train import train_and_promote
from intelligence.models.forecast.eval import skill_vs_baseline, apply_interval_scale

HORIZONS = list(range(3, 73, 3))
TEST_TAIL_DAYS = 14


def evaluate(panel: pd.DataFrame, oracle_met: bool = False,
             fires: pd.DataFrame | None = None) -> dict:
    """Kept for scripts/eval_forecast_live.py's existing import. `oracle_met`
    is no longer supported by the new model (the old model's leakage-sizing
    experiment doesn't apply to a quantile/multi-city model) — always
    ignored, and the caller gets an explicit note rather than a silent gap."""
    frame = mask_unknown_city(build_features(panel, HORIZONS, fires=fires,
                                              restrict_to_station_cells=True))
    split = frame.ts.max() - pd.Timedelta(days=TEST_TAIL_DAYS)
    # Climatology from the TRAIN side only, re-attached to the whole frame:
    # building it inside build_features used the full panel, so every test
    # row's clim_* partly came from that row's own future.
    frame = attach_climatology(frame, build_climatology(panel[panel.ts <= split]))
    # dropna on y is not cosmetic: build_features emits a row for every
    # (cell, hour, horizon), and non-station cells have no label at all yet
    # survive the lag filter because the composite fills their lags.
    # LightGBM neither raises nor warns on NaN labels — it silently trains a
    # model no better than the mean.
    train_frame = frame[frame.ts <= split].dropna(subset=["y"])
    test_frame = frame[frame.ts > split].dropna(subset=["y"])
    results = {}
    if train_frame.empty:
        return {f"h{h}": {"n_test": 0, "note": "insufficient data"} for h in HORIZONS}
    # ONE model for every horizon: `horizon` is a feature, and train_frame
    # does not depend on h — only the test slice does. Training inside the
    # loop built 24x more boosters than the eval needs.
    models = train_quantile_models(train_frame, FEATURE_COLUMNS, num_boost_round=200)
    for h in HORIZONS:
        te_h = test_frame[test_frame.horizon == h]
        if te_h.empty:
            results[f"h{h}"] = {"n_test": 0, "note": "insufficient data"}
            continue
        pred = predict_quantiles(models, te_h, FEATURE_COLUMNS)
        results[f"h{h}"] = {
            "n_test": len(te_h),
            "skill_vs_persistence_pct": skill_vs_baseline(te_h["y"].values, pred["pm25_p50"].values, te_h["lag_0"].values),
        }
    if oracle_met:
        results["_note"] = "oracle_met is not supported by the quantile/multi-city model; ignored"
    return results


# Only frame.ts == frame.ts.max() ever survives out of _predict_field's
# build_features call below, so exploding the WHOLE historical panel (up to
# 2 years) across HORIZONS just to throw away all but the last timestamp is
# pure waste -- on a real multi-city panel this is the single biggest
# contributor to the training Job's OOM. roll_med_168 (7-day rolling
# median) is the longest lookback any feature needs, so trimming the panel
# to this many trailing hours before build_features leaves the kept row's
# features byte-identical while cutting the exploded frame's row count by
# roughly (panel span in hours / this constant).
PREDICT_LOOKBACK_HOURS = 24 * 10   # 168h roll_med window + 24h max lag + 3-day buffer


def _predict_field(panels: dict, served_manifest: dict, served_models: dict,
                    feature_cols: list[str],
                    fires_by_city: dict | None = None,
                    clim_panels: dict | None = None) -> dict[str, list[dict]]:
    """Per-city forecast.json field, predicted with the SERVED model.

    DEFENSIVE, not a live bugfix. `city_categories` is pinned to
    `served_manifest["cities"]` — the cities the served model was ACTUALLY
    trained on — rather than `panels.keys()` (the current run's data), which
    differ whenever a refused promotion falls back to a prior model trained
    on a different city set. Measured: LightGBM realigns a pandas
    Categorical column by category NAME at predict time, so with the
    Categorical input this function actually passes, both sources produce
    identical predictions today. Pinning to the manifest keeps that correct
    if the realignment convention ever changes, or if a future call site
    passes a raw string `city` column (which gets no name-based realignment).
    """
    fires_by_city = fires_by_city or {}
    clim_panels = clim_panels or {}
    city_categories = sorted(set(served_manifest["cities"]) | {UNKNOWN_CITY})
    # Raw p10/p90 measured 0.684 coverage against a 0.80 target on the first
    # real 8-city run -- bands a third too narrow. Widen by whatever the
    # served model's own held-out folds said was needed.
    scale = (served_manifest.get("eval") or {}).get("interval_scale")
    fields = {}
    for city, panel in panels.items():
        # Climatology needs the FULL history (a month/dow-hour bucket only
        # has a few dozen samples a year) -- built here, from the untrimmed
        # panel, and passed in explicitly so build_features below doesn't
        # rebuild it (and doesn't silently rebuild it from the trimmed slice).
        # Years of history where we have it (see _climatology_panel), else this
        # city's own 60-day window. The lags/features below always come from
        # `panel` — only the climatology tables use the longer history.
        clim_src = clim_panels.get(city)
        if clim_src is None:
            clim_src = panel
            print(f"[forecast] {city}: climatology from the 60-day operational "
                  f"window (no historical panel) — clim_month buckets hold ~1 sample")
        clim_tables = build_climatology(clim_src)
        cutoff = panel.ts.max() - pd.Timedelta(hours=PREDICT_LOOKBACK_HOURS)
        recent_panel = panel[panel.ts > cutoff]
        frame = build_features(recent_panel, HORIZONS, fires=fires_by_city.get(city),
                                clim_tables=clim_tables)
        frame["city"] = pd.Categorical(frame["city"], categories=city_categories)
        latest = frame.ts.max()
        latest_rows = frame[frame.ts == latest].dropna(subset=["lag_0"])
        if latest_rows.empty:
            continue
        pred = predict_quantiles(served_models, latest_rows, feature_cols)
        p10, p90 = apply_interval_scale(pred.pm25_p10, pred.pm25_p50, pred.pm25_p90, scale)
        # PM2.5 has a hard physical floor at zero. Widening the raw quantiles by
        # `interval_scale` pushes p10 below it on 93% of Delhi's ward rows
        # (measured), and a forecast that reports a negative concentration is
        # exactly the class of physically-impossible output this project deleted
        # the SO2 channel over. Clamp at 0 — a lower bound of "clean air" is the
        # honest floor, not a negative number the instrument cannot mean.
        #
        # NOTE this only fixes the SIGN. The width is still miscalibrated: a
        # single global scale is applied at every lead time, so h=3's band is
        # ~298 ug/m3 against h=72's ~317 — a 6% spread across a 24x change in
        # horizon, when a 3-hour forecast should be far tighter than a 3-day
        # one. Genuinely fixing that needs a per-horizon scale from the held-out
        # folds; see the note in the PR.
        # np.clip, not Series.clip: apply_interval_scale returns numpy arrays,
        # and Series.clip's `lower=` kwarg does not exist there.
        p10 = np.clip(p10, 0.0, None)
        p90 = np.clip(p90, 0.0, None)
        # `urgency` — is this cell forecast to get materially WORSE than it is
        # right now (>= 20 ug/m3)? Carried by the old single-file forecast.py,
        # dropped when this package replaced it, and read by
        # intelligence/agents/advisory.py::"worsening" — so its absence took the
        # advisory agent down with a KeyError mid-chain. lag_0 is the cell's most
        # recent observed value, which is exactly the "now" the old code compared
        # against (it used `now.get(cell)`).
        now_obs = latest_rows["lag_0"].to_numpy(dtype=float)
        p50 = np.asarray(pred.pm25_p50, dtype=float)
        urgency = np.isfinite(now_obs) & ((p50 - now_obs) >= 20.0)
        fields[city] = [{"cell": c, "horizon_h": int(h), "pm25_hat": round(max(float(m), 0.0), 1),
                         "pm25_p10": round(float(lo), 1), "pm25_p90": round(float(hi), 1),
                         "urgency": bool(u)}
                        for c, h, m, lo, hi, u in zip(latest_rows.cell, latest_rows.horizon,
                                                       pred.pm25_p50, p10, p90, urgency)]
    return fields


def _ward_series(field: list[dict], city_out) -> list[dict]:
    """Collapse the cell forecast to a per-ward, per-horizon MEDIAN.

    Ported from the old single-file forecast.py, which wrote this file and
    which this package replaced. Dropping it silently broke the citizen ward
    timeline: app/frontend/.../WardTimeline.tsx reads forecast_ward.json with
    a [] fallback, then returns null on fewer than 2 points — so the panel
    vanished from the page with no error anywhere to notice it by.

    Median, not mean, like every other aggregate in this project: one hot cell
    beside a landfill must not drag the whole ward's timeline up.

    The quantiles are medianed the same way and travel with it. A ward-level
    band is the honest shape of this number — the citizen timeline draws
    "how sure are we", and the p50 alone cannot say that.

    `city_out` is passed in rather than read from the module-level DATA_OUT:
    DATA_OUT binds at import time to whichever single AQ_CITY started the
    process, and this runs inside a loop over several cities (the same trap
    _predict_field documents).
    """
    wards_p = city_out / "wards.json"
    if not wards_p.exists() or not field:
        return []
    try:
        cells = json.loads(wards_p.read_text(encoding="utf-8"))["cells"]
    except (json.JSONDecodeError, OSError, KeyError):
        # A missing/!malformed ward layer must not kill the run — the cell
        # forecast is already written and is the primary product.
        return []
    cell_to_ward = {c["cell"]: c["ward_id"] for c in cells}

    df = pd.DataFrame(field)
    df["ward_id"] = df.cell.map(cell_to_ward)
    df = df[df.ward_id.notna() & (df.ward_id != "unassigned")]
    if df.empty:
        return []
    g = (df.groupby(["ward_id", "horizon_h"])
           .agg(pm25_hat=("pm25_hat", "median"),
                pm25_p10=("pm25_p10", "median"),
                pm25_p90=("pm25_p90", "median"),
                n_cells=("cell", "size"))
           .reset_index())
    return [{"ward_id": r.ward_id, "horizon_h": int(r.horizon_h),
             "pm25_hat": round(float(r.pm25_hat), 1),
             "pm25_p10": round(float(r.pm25_p10), 1),
             "pm25_p90": round(float(r.pm25_p90), 1),
             "n_cells": int(r.n_cells)}
            for r in g.itertuples()]


def _write_city_outputs(city_out, field: list[dict], eval_payload: dict,
                        write_forecast_eval: bool = False) -> None:
    """forecast.json + forecast_ward.json (+ the model's own metrics) for one city.

    Compact, not pretty-printed: 24 horizons x ~1700 cells is 40k rows, and
    `indent=2` alone cost 1.7 MB of leading spaces in a file the browser
    downloads. Nothing reads it by eye.

    TWO DIFFERENT EVAL FILES, because they answer different questions and have
    incompatible shapes:

      forecast_eval.json        per-HORIZON skill against persistence + diurnal
                                ({"h24": {"rmse_model", "skill_vs_persistence_pct"}}).
                                This is the rubric's number and what the admin
                                Validation page renders.
      forecast_model_eval.json  the served model's own training metrics
                                (walk-forward, spatial-LOSO, per-city LOSO,
                                interval_scale). Nothing reads this by horizon.

    The manifest's eval is the SECOND kind. Writing it into forecast_eval.json
    silently blanks the Validation page, whose lookup is
    `fe[city][h].skill_vs_persistence_pct` -- every horizon key is simply absent
    from the new shape. So SERVING never touches forecast_eval.json: predicting
    with an already-trained model produces no new evaluation, and overwriting
    last evaluation's answer with a differently-shaped one is a loss, not an
    update. Only a training run (`write_forecast_eval=True`) may replace it.
    """
    city_out.mkdir(parents=True, exist_ok=True)
    (city_out / "forecast.json").write_text(
        json.dumps(field, separators=(",", ":")), encoding="utf-8")
    (city_out / "forecast_model_eval.json").write_text(
        json.dumps(eval_payload, indent=2), encoding="utf-8")
    if write_forecast_eval:
        (city_out / "forecast_eval.json").write_text(
            json.dumps(eval_payload, indent=2), encoding="utf-8")
    print(f"[forecast] {city_out.name}: wrote forecast.json "
          f"({len(field)} cell-horizons)")

    ward_rows = _ward_series(field, city_out)
    if ward_rows:
        (city_out / "forecast_ward.json").write_text(
            json.dumps(ward_rows, separators=(",", ":")), encoding="utf-8")
        print(f"[forecast] {city_out.name}: wrote forecast_ward.json "
              f"({len(ward_rows)} ward-horizons)")
    else:
        # Loud, because the citizen timeline reads this file and renders
        # nothing at all when it is missing — the exact silent failure this
        # function exists to stop recurring.
        print(f"[forecast] {city_out.name}: NO forecast_ward.json — wards.json "
              f"missing or no cell mapped to a ward; the citizen ward timeline "
              f"will not render for this city")


def _load_promoted(out_dir, manifest: dict):
    """The three promoted quantile boosters, or None with a reason printed.

    Shared by run() and serve() so a stale/corrupt/feature-mismatched model
    is diagnosed identically no matter which entrypoint hit it.
    """
    import lightgbm as lgb
    model_dir = out_dir / manifest["version"]
    try:
        models = {
            0.1: lgb.Booster(model_file=str(model_dir / "model_p10.txt")),
            0.5: lgb.Booster(model_file=str(model_dir / "model_p50.txt")),
            0.9: lgb.Booster(model_file=str(model_dir / "model_p90.txt")),
        }
    except Exception as e:   # noqa: BLE001 — a missing/corrupted model dir must not crash the run
        print(f"[forecast] served model at {model_dir} failed to load "
              f"({type(e).__name__}: {e}) — skipping forecast.json this run")
        return None

    # Loading a booster succeeds regardless of how many features it was
    # trained on; the mismatch only surfaces as a raise deep inside predict().
    stale = {q: m.num_feature() for q, m in models.items()
             if m.num_feature() != len(FEATURE_COLUMNS)}
    if stale:
        print(f"[forecast] served model at {model_dir} was trained on {stale} features "
              f"but FEATURE_COLUMNS now has {len(FEATURE_COLUMNS)} — the feature set "
              f"changed since it was trained; skipping forecast.json until a model "
              f"trained on the current features is promoted")
        return None
    return models


def _load_panels(cities: list[str] | None, prefer_historical: bool = True):
    """Per-city (panel, fires).

    `prefer_historical` picks WHICH panel, and the two callers want opposite
    things:

      TRAINING (run()) wants data/historical/<city>/ — the 2-year backfill.
      A model learns seasonality it can only see over years.

      SERVING (serve()) wants the OPERATIONAL panel the rest of this pipeline
      just built. Two reasons, and the first is a correctness bug, not a
      preference. Under --synthetic the operational panel IS the synthetic
      world, while data/historical/ is real scraped data: preferring historical
      there silently forecasts a real city from inside a synthetic run, the same
      class of error as live mode mixing real stations with a synthetic
      satellite. Second, _predict_field only ever keeps the LAST timestamp and
      trims to PREDICT_LOOKBACK_HOURS (10 days) before building features, so a
      2-year panel is thrown away almost entirely — it just costs the memory.
      Loading 8 cities of it is what exhausted RAM mid-merge in features.py.
    """
    from shared.config import CITIES, DATA_OUT_BASE, DATA_RAW_BASE
    cities = cities or list(CITIES)
    panels, fires_by_city = {}, {}
    for city in cities:
        hist = ROOT / "data" / "historical" / city
        hist_panel = hist / "panel.parquet"
        operational = DATA_OUT_BASE / city / "panel.parquet"
        use_hist = hist_panel.exists() and (prefer_historical or not operational.exists())
        path = hist_panel if use_hist else operational
        if not path.exists():
            print(f"[forecast] {city}: no panel at {path}, skipping")
            continue
        panel = pd.read_parquet(path)
        # `city` is a constant provenance label (panel.py: `panel["city"] = CITY`),
        # not measured data — but build_features groups on it, so a panel built
        # before that line existed dies with KeyError: 'city' deep inside the
        # feature build. Every operational panel written before this column was
        # added is otherwise perfectly good real data, and we know exactly which
        # city it belongs to: we just read it out of that city's directory.
        # Backfilling it here means an existing panel does NOT have to be
        # re-ingested (live APIs + GEE auth) just to be forecast from.
        if "city" not in panel.columns:
            panel["city"] = city
            print(f"[forecast] {city}: panel predates the `city` column — labelled in place")
        panels[city] = downcast_panel(panel)
        fires_p = (hist if use_hist else DATA_RAW_BASE / city) / "fires.parquet"
        if fires_p.exists():
            fires_by_city[city] = pd.read_parquet(fires_p)
        else:
            print(f"[forecast] {city}: no fires at {fires_p} — fire_pressure_regional will be 0")
    return panels, fires_by_city


def _climatology_panel(city: str):
    """The LONG panel to build this city's climatology from, ward ids realigned.

    Climatology is the one thing serving genuinely wants years of, not the 60-day
    operational window. Its buckets are (scope, day-of-week x hour) and (scope,
    day-of-year): over 60 days a dow_hour bucket holds ~8 samples and a
    day-of-year bucket holds exactly ONE, so `clim_month` degenerates into "that
    single day's reading" wearing a climatology label. Over two years the same
    buckets hold ~104 and 2+. The model was TRAINED on climatology built from the
    2-year panels, so feeding it 60-day climatology at serve time is train/serve
    skew in the features it leans on hardest for the 98%+ of cells that have no
    station of their own.

    WARD IDS ARE REMAPPED, and skipping that would silently corrupt the result.
    lookup_climatology falls back cell -> ward -> city, and a historical panel
    carries whatever ward layer existed when it was BUILT. Measured: Delhi's
    historical ward ids match the current map 100% (it always had a real ward
    file), but Mumbai's match 0% -- 60 Voronoi wards then, 25 real BMC wards now.
    Joining those by id pairs "W005" the Voronoi polygon with "W005" the ward of
    Mumbai's E division, which are not the same place. So ward_id is recomputed
    here from the CURRENT ward layer rather than trusted from the file.

    Returns None when there is no historical panel, and the caller falls back to
    the operational one.
    """
    hist = ROOT / "data" / "historical" / city / "panel.parquet"
    if not hist.exists():
        return None
    from shared.wards import ward_map
    cols = ["cell", "ts", "pm25_station", "ward_id", "city"]
    try:
        panel = pd.read_parquet(hist, columns=cols)
    except Exception as e:   # noqa: BLE001 — a bad historical file must not sink serving
        print(f"[forecast] {city}: historical panel unreadable ({type(e).__name__}) "
              f"— climatology falls back to the operational window")
        return None
    panel["ts"] = pd.to_datetime(panel.ts, utc=True)
    wm = ward_map()
    fresh = panel.cell.map(wm).fillna("unassigned")
    # The files are correct as of the ward-realignment pass, so this should agree.
    # It is kept as a GUARD, not a patch: if a historical panel is ever rebuilt
    # against a different ward layer than the one serving uses, that must be
    # visible, because it silently corrupts the ward-scope climatology that 98%+
    # of cells fall back to. Loud, then corrected — never silently corrected.
    disagree = float((panel.ward_id != fresh).mean())
    if disagree > 0.001:
        print(f"[forecast] {city}: WARNING — {100*disagree:.1f}% of the historical "
              f"panel's ward_ids disagree with the current ward layer "
              f"({panel.ward_id.nunique()} in file vs {fresh.nunique()} now). "
              f"Using the current layer. Rebuild the historical panel: its stale "
              f"ward_ids are ALSO what training would group climatology by.")
    panel["ward_id"] = fresh
    panel["city"] = city
    return panel


def serve(cities: list[str] | None = None) -> dict:
    """Predict with the ALREADY-PROMOTED weights. No training. Seconds, not hours.

    This is what the operational pipeline calls. It is the serving half of
    run(): same feature build, same model, same output files — it simply does
    not train first, because the weights it needs were trained offline and
    committed (models/, synced by .github/workflows/sync-forecast-model.yml).

    Returns the served manifest, or {} when there is nothing promoted to serve
    (a fresh checkout that has never trained), which is a clear state, not an
    error — the caller keeps going and the rest of the pipeline still runs.
    """
    from shared.config import DATA_OUT_BASE

    out_dir = ROOT / "data" / "outputs" / "_forecast_models"
    manifest_p = out_dir / "manifest.json"
    if not manifest_p.exists():
        # Fall back to the repo-committed models/ directory, which is where
        # the training workflow syncs promoted weights to.
        out_dir = ROOT / "models"
        manifest_p = out_dir / "manifest.json"
    if not manifest_p.exists():
        print("[forecast] no promoted model found (looked in data/outputs/"
              "_forecast_models/ and models/) — run the training workflow, or "
              "intelligence.models.forecast.run() locally. Skipping forecast.")
        return {}

    try:
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[forecast] manifest at {manifest_p} is unreadable "
              f"({type(e).__name__}: {e}) — skipping forecast")
        return {}

    models = _load_promoted(out_dir, manifest)
    if models is None:
        return manifest

    panels, fires_by_city = _load_panels(cities, prefer_historical=False)
    if not panels:
        print("[forecast] no city panels available — run the operational pipeline "
              "or the historical backfill first. Skipping forecast.")
        return manifest

    clim_panels = {}
    for city in panels:
        cp = _climatology_panel(city)
        if cp is not None:
            span = (cp.ts.max() - cp.ts.min()).days
            print(f"[forecast] {city}: climatology from {span} days of history "
                  f"({len(cp):,} rows)")
            clim_panels[city] = cp
    fields = _predict_field(panels, manifest, models, FEATURE_COLUMNS,
                            fires_by_city=fires_by_city, clim_panels=clim_panels)
    eval_payload = manifest.get("eval") or {}
    for city, field in fields.items():
        _write_city_outputs(DATA_OUT_BASE / city, field, eval_payload)
    return manifest


def run(cities: list[str] | None = None, checkpoint_dir: str | None = None) -> dict:
    """New entrypoint. Trains on each city's historical panel if
    data/historical/<city>/panel.parquet exists (richer, longer window),
    else falls back to that city's operational data/outputs/<city>/panel.parquet
    (matches the old model's data source, for a city with no backfill yet).

    `checkpoint_dir`: passed straight through to train_and_promote -- see
    its own docstring and checkpoint.py. None (default) disables it."""
    from shared.config import CITIES, DATA_OUT_BASE, DATA_RAW_BASE
    cities = cities or list(CITIES)
    panels, fires_by_city = {}, {}
    for city in cities:
        hist = ROOT / "data" / "historical" / city
        hist_panel = hist / "panel.parquet"
        use_hist = hist_panel.exists()
        path = hist_panel if use_hist else (DATA_OUT_BASE / city / "panel.parquet")
        if not path.exists():
            print(f"[forecast] {city}: no panel at {path}, skipping")
            continue
        panel = pd.read_parquet(path)
        # `city` is a constant provenance label (panel.py: `panel["city"] = CITY`),
        # not measured data — but build_features groups on it, so a panel built
        # before that line existed dies with KeyError: 'city' deep inside the
        # feature build. Every operational panel written before this column was
        # added is otherwise perfectly good real data, and we know exactly which
        # city it belongs to: we just read it out of that city's directory.
        # Backfilling it here means an existing panel does NOT have to be
        # re-ingested (live APIs + GEE auth) just to be forecast from.
        if "city" not in panel.columns:
            panel["city"] = city
            print(f"[forecast] {city}: panel predates the `city` column — labelled in place")
        panels[city] = downcast_panel(panel)
        # The RAW per-fire-event table (lat/lon/frp), not the panel's
        # pre-aggregated fires_6h/frp_6h columns: spatial.fire_pressure needs
        # each detection's own position to weight it by distance and wind.
        # Without this every fire_pressure_regional value is 0.0 — the
        # feature was wired into FEATURE_COLUMNS but fed nothing.
        fires_p = (hist if use_hist else DATA_RAW_BASE / city) / "fires.parquet"
        if fires_p.exists():
            fires_by_city[city] = pd.read_parquet(fires_p)
        else:
            print(f"[forecast] {city}: no fires at {fires_p} — fire_pressure_regional will be 0")

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

    manifest = train_and_promote(panels, HORIZONS, FEATURE_COLUMNS, out_dir,
                                  prior_manifest=prior, fires_by_city=fires_by_city,
                                  checkpoint_dir=checkpoint_dir)

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

    # A booster whose feature count no longer matches FEATURE_COLUMNS is a
    # routine event here, not a corrupted file: it happens whenever this run
    # was REFUSED and the prior model predates a change to FEATURE_COLUMNS.
    # It must not take the whole run down after training has already succeeded.
    served_models = _load_promoted(out_dir, served_manifest)
    if served_models is None:
        return manifest

    # DATA_OUT_BASE / city, NOT the singular DATA_OUT: DATA_OUT is bound at
    # process-import time to whichever single AQ_CITY started this process
    # (see Global Constraints). This loop writes several cities' outputs in
    # one process — using the bare DATA_OUT here would silently write every
    # city's forecast.json into whichever one city happened to be active,
    # exactly the bug class this plan warns about elsewhere.
    fields = _predict_field(panels, served_manifest, served_models, FEATURE_COLUMNS,
                             fires_by_city=fires_by_city)
    for city, field in fields.items():
        _write_city_outputs(DATA_OUT_BASE / city, field, manifest["eval"],
                            write_forecast_eval=True)

    return manifest
