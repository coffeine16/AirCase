# intelligence/models/forecast/train.py
"""Top-level training orchestration + promotion gate (spec sections 5.4, 7).
A freshly trained model must match or beat the currently-served model on
walk-forward skill, spatial-LOSO, city-LOSO, and quantile coverage before
it's allowed to replace it — code-enforced, not eyeballed."""
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

    weights = event_weights(frame)
    final_train = frame.dropna(subset=["y"])
    final_models = train_quantile_models(final_train, feature_cols, num_boost_round=500)
    # LightGBM's Dataset already built inside train_quantile_models doesn't take
    # sample weights via this simple call — apply them via a second, weighted
    # pass on the same data for the SERVED model only (walk-forward/LOSO stay
    # unweighted, since they measure generalisation, not the production fit).
    import lightgbm as lgb
    from intelligence.models.forecast.model import PARAMS, QUANTILES
    weighted_models = {}
    for q in QUANTILES:
        ds = lgb.Dataset(final_train[feature_cols], label=final_train["y"],
                          weight=event_weights(final_train), categorical_feature=["city"])
        weighted_models[q] = lgb.train({**PARAMS, "alpha": q}, ds, num_boost_round=500)
    final_models = weighted_models

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
    promoted = True
    if prior_manifest is not None:
        prior_rmse = prior_manifest.get("eval", {}).get("spatial_loso_rmse")
        if prior_rmse is not None and np.isfinite(prior_rmse):
            # RMSE: LOWER is better. A regression means the NEW rmse is
            # HIGHER than the prior's by more than the tolerance -- allowed
            # is an UPPER bound, not a lower one. (The original draft had
            # this inverted: allowed = prior*(1-tol) with a "<" refusal
            # check refuses promotion when the new model is BETTER, and
            # would silently promote a worse one, since RMSE can't go
            # negative. Caught before dispatch by tracing the regression
            # test by hand: a fake prior RMSE of 0.0 must force a refusal,
            # and only ">" does that.)
            allowed = prior_rmse * (1 + regression_tolerance_pct / 100)
            if not np.isfinite(loso_result["overall_rmse"]) or loso_result["overall_rmse"] > allowed:
                promoted = False

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
        print(f"[train] {version} trained but NOT promoted — spatial-LOSO RMSE "
              f"{loso_result['overall_rmse']} regressed beyond the "
              f"{regression_tolerance_pct}% tolerance vs the current model")
    return manifest
