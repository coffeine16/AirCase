# intelligence/models/forecast/train.py
"""Top-level training orchestration + promotion gate (spec sections 5.4, 7).
A freshly trained model must match or beat the currently-served model on
walk-forward skill, spatial-LOSO, and city-LOSO before it's allowed to
replace it — code-enforced, not eyeballed. quantile_coverage is computed
and recorded in every manifest but deliberately NOT gated: unlike RMSE
(lower always better) or skill (higher always better), coverage has no
single "better" direction — both over- and under-coverage relative to the
~0.80 target are miscalibration. Gating it needs a distance-from-target
comparison this version doesn't implement; stated here as a real scope
limit, not a silent gap."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from intelligence.models.forecast.features import build_features
from intelligence.models.forecast.model import (
    train_quantile_models, predict_quantiles, train_ceiling_baseline, mask_unknown_city,
)
from intelligence.models.forecast.validation import (
    walk_forward_folds, event_weights, spatial_loso, run_city_loso,
)
from intelligence.models.forecast.eval import skill_vs_baseline, quantile_coverage, quiet_vs_event_breakdown


def _regressed(new_val: float | None, prior_val: float | None,
               higher_is_better: bool, tolerance_pct: float) -> bool:
    """True if `new_val` is a regression vs `prior_val` beyond tolerance.
    Missing or non-finite `new_val`/`prior_val` means there is nothing to
    compare -- returns False (does not block). This is deliberate: a
    diagnostic like walk-forward skill can legitimately come back None on
    too little history (expected on a small panel, not a failure), and
    that must not be conflated with the primary metric (spatial-LOSO)
    genuinely breaking -- see the separate, stricter spatial_loso_ok check
    in train_and_promote below, which is where a non-finite result DOES
    block promotion unconditionally."""
    if new_val is None or prior_val is None:
        return False
    if not np.isfinite(new_val) or not np.isfinite(prior_val):
        return False
    if higher_is_better:
        allowed = prior_val * (1 - tolerance_pct / 100)
        return new_val < allowed
    allowed = prior_val * (1 + tolerance_pct / 100)
    return new_val > allowed


def _version_id() -> str:
    # Wall-clock, NOT derived from the panels' own data timestamps: two
    # training runs on the same day's data (a retry after a rejected
    # promotion, a manual re-run) would otherwise produce an IDENTICAL
    # version string and silently overwrite each other's artifacts at the
    # same path. This is regular application code, not a Workflow script —
    # wall-clock time is exactly the right source here, not a hazard.
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H%M%S")


def train_and_promote(panels_by_city: dict[str, pd.DataFrame], horizons: list[int],
                       feature_cols: list[str], out_dir: Path,
                       prior_manifest: dict | None = None,
                       regression_tolerance_pct: float = 5.0) -> dict:
    out_dir = Path(out_dir)
    full_panel = pd.concat(panels_by_city.values(), ignore_index=True)
    frame = mask_unknown_city(build_features(full_panel, horizons))

    folds = walk_forward_folds(frame)
    fold_skills = []
    for train_end, test_start, test_end in folds:
        tr = frame[frame.ts <= train_end]
        te = frame[(frame.ts > test_start) & (frame.ts <= test_end)].dropna(subset=["y"])
        if tr.empty or te.empty:
            continue
        models = train_quantile_models(tr, feature_cols, num_boost_round=300)
        pred = predict_quantiles(models, te, feature_cols)
        persistence = te["lag_0"].values
        fold_skills.append(skill_vs_baseline(te["y"].values, pred["pm25_p50"].values, persistence))

    loso_result = spatial_loso(full_panel, horizons, feature_cols)
    city_result = run_city_loso(panels_by_city, horizons, feature_cols)

    # LightGBM's Dataset already built inside train_quantile_models doesn't
    # take sample weights via that simple call, so the SERVED model (which
    # needs the real-event oversampling weight) is trained directly here
    # instead — walk-forward/LOSO above stay unweighted on purpose, since
    # they measure generalisation, not the production fit. Only ONE
    # training pass for the served model, not two: an earlier draft called
    # train_quantile_models AND this weighted loop, discarding the first
    # result unused — doubling the cost of the single most expensive stage
    # in this function for nothing.
    final_train = frame.dropna(subset=["y"])
    import lightgbm as lgb
    from intelligence.models.forecast.model import PARAMS, QUANTILES
    final_models = {}
    for q in QUANTILES:
        ds = lgb.Dataset(final_train[feature_cols], label=final_train["y"],
                          weight=event_weights(final_train), categorical_feature=["city"])
        final_models[q] = lgb.train({**PARAMS, "alpha": q}, ds, num_boost_round=500)

    baseline_reg = train_ceiling_baseline(final_train, [c for c in feature_cols if c != "city"])
    ceiling_pred = baseline_reg.predict(final_train[[c for c in feature_cols if c != "city"]].select_dtypes(include=[np.number]).fillna(0.0))
    pred_final = predict_quantiles(final_models, final_train, feature_cols)
    ceiling_skill = skill_vs_baseline(final_train["y"].values, pred_final["pm25_p50"].values, ceiling_pred)

    # quiet-vs-event breakdown (spec section 6) — "is_event" uses the SAME
    # definition as event_weights' boost condition (real trailing fire
    # activity), so the two stay consistent with each other.
    is_event_final = (final_train["fires_6h"].fillna(0) > 0).values
    quiet_event = quiet_vs_event_breakdown(final_train["y"].values, pred_final["pm25_p50"].values, is_event_final)

    eval_report = {
        "walk_forward_skill_median": round(float(np.median(fold_skills)), 1) if fold_skills else None,
        "walk_forward_skill_folds": len(fold_skills),
        "spatial_loso_rmse": loso_result["overall_rmse"],
        "spatial_loso_n_stations": loso_result["n_stations"],
        "city_loso": city_result["per_city"],
        "quantile_coverage": quantile_coverage(final_train["y"].values, pred_final["pm25_p10"].values,
                                                pred_final["pm25_p90"].values),
        "ceiling_skill_vs_linear": ceiling_skill,
        "quiet_vs_event": quiet_event,
    }

    version = _version_id()
    prior_eval = (prior_manifest or {}).get("eval", {})

    # spatial-LOSO is the primary validation metric (this plan's own
    # "headline number"). Unlike walk-forward/city-LOSO, which can
    # legitimately come back None/empty on too little history (expected on
    # a small panel, not a failure -- see _regressed's docstring), a
    # non-finite spatial-LOSO RMSE means real stations existed but scoring
    # genuinely broke, and that must never be silently promoted, WITH or
    # WITHOUT a prior to compare against. This closes finding #3 from Task
    # 10's review: `promoted` used to default True unconditionally and only
    # checked finiteness inside the `prior_manifest is not None` branch, so
    # a NaN-RMSE first run promoted silently.
    spatial_loso_ok = np.isfinite(eval_report["spatial_loso_rmse"])

    # The other two gated metrics (walk-forward skill: higher better;
    # city-LOSO: lower better, compared as the MEDIAN across whatever
    # cities are present in each run -- robust to the city set changing
    # between runs, and consistent with this project's median-not-mean
    # convention everywhere else).
    prior_city_rmses = [v["rmse"] for v in prior_eval.get("city_loso", {}).values()]
    new_city_rmses = [v["rmse"] for v in city_result["per_city"].values()]
    prior_city_median = float(np.median(prior_city_rmses)) if prior_city_rmses else None
    new_city_median = float(np.median(new_city_rmses)) if new_city_rmses else None

    promoted = spatial_loso_ok and not any([
        _regressed(eval_report["spatial_loso_rmse"], prior_eval.get("spatial_loso_rmse"),
                   higher_is_better=False, tolerance_pct=regression_tolerance_pct),
        _regressed(eval_report["walk_forward_skill_median"], prior_eval.get("walk_forward_skill_median"),
                   higher_is_better=True, tolerance_pct=regression_tolerance_pct),
        _regressed(new_city_median, prior_city_median,
                   higher_is_better=False, tolerance_pct=regression_tolerance_pct),
    ])

    manifest = {"version": version, "trained_at": pd.Timestamp.utcnow().isoformat(),
                "cities": sorted(panels_by_city), "eval": eval_report, "promoted": promoted}

    if promoted:
        version_dir = out_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        for q, m in final_models.items():
            m.save_model(str(version_dir / f"model_p{int(q * 100)}.txt"))
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[train] promoted {version}: walk-forward skill "
              f"{eval_report['walk_forward_skill_median']}%, spatial-LOSO RMSE "
              f"{eval_report['spatial_loso_rmse']}")
    else:
        reason = ("spatial-LOSO RMSE is non-finite (no scoreable stations)"
                  if not spatial_loso_ok else
                  f"regressed beyond the {regression_tolerance_pct}% tolerance "
                  f"on spatial-LOSO, walk-forward skill, and/or city-LOSO vs the current model")
        print(f"[train] {version} trained but NOT promoted — {reason}")
    return manifest
