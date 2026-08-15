"""Cross-validation strategies (spec sections 5.1, 5.2, 5.3). No single
fixed holdout — walk-forward folds, spatial-LOSO (Task 9), city-LOSO, and
real-event oversampling that touches ONLY sample weights, never a
synthetic ignition schedule (spec 5.3's hard rule)."""
import numpy as np
import pandas as pd

from intelligence.models.forecast.features import build_features
from intelligence.models.forecast.model import (
    train_quantile_models, predict_quantiles, mask_unknown_city, UNKNOWN_CITY,
)
from intelligence.models.forecast.eval import skill_vs_baseline


def walk_forward_folds(frame: pd.DataFrame, ts_col: str = "ts",
                        min_train_days: int = 180, test_days: int = 21,
                        step_days: int = 21) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Yields (train_end, test_start, test_end) triples. Train is always
    "everything up to train_end" (expanding window); test is the following
    `test_days`. Returns [] rather than raising when there isn't enough
    history for even one fold — callers must check."""
    start, end = frame[ts_col].min(), frame[ts_col].max()
    folds = []
    train_end = start + pd.Timedelta(days=min_train_days)
    while train_end + pd.Timedelta(days=test_days) <= end:
        folds.append((train_end, train_end, train_end + pd.Timedelta(days=test_days)))
        train_end += pd.Timedelta(days=step_days)
    return folds


def event_weights(frame: pd.DataFrame, fire_col: str = "fires_6h", boost: float = 4.0) -> np.ndarray:
    """Per-row training weight: `boost`x for rows with real trailing fire
    activity, 1.0 otherwise (spec 5.3). Real FIRMS detections only."""
    is_event = (frame[fire_col].fillna(0) > 0)
    return np.where(is_event, boost, 1.0)


def city_loso_splits(cities: list[str]) -> list[tuple[str, list[str]]]:
    """One (held_out_city, train_cities) pair per city (spec 5.2)."""
    return [(c, [x for x in cities if x != c]) for c in cities]


def spatial_loso(panel: pd.DataFrame, horizons: list[int], feature_cols: list[str]) -> dict:
    """Direct analog of fusion.py::loso_validation, applied to the
    forecaster (spec 5.2). For each real station: retrain excluding it
    entirely from training, forecast its cell using composite-only features
    with itself excluded from its own composite (spec 3.1's self-exclusion
    rule via features.build_features's loso_exclude), score against its
    real held-out readings."""
    station_cells = sorted(panel[panel.pm25_station.notna()].cell.unique())
    per_station, all_true, all_pred = {}, [], []
    for held_out in station_cells:
        # build_features() internally re-joins its working frame against `p`
        # (the sorted copy of what's passed in) by index label after a merge
        # that resets that label to a fresh 0..n-1 RangeIndex (fire_pressure
        # join). Every other caller always passes an already-0-based panel,
        # so the label and the row position silently coincided. A LOSO
        # training panel is panel[panel.cell != held_out] -- a slice that
        # keeps the ORIGINAL (non-0-based) row labels -- and without this
        # reset that mismatch raises a KeyError deep inside build_features.
        # Not a features.py bug to fix here: this satisfies its existing,
        # previously-unexercised precondition at the call site instead.
        train_panel = panel[panel.cell != held_out].reset_index(drop=True)
        train_frame = mask_unknown_city(build_features(train_panel, horizons))
        test_panel = panel[panel.cell == held_out].reset_index(drop=True)
        test_frame = build_features(test_panel, horizons, loso_exclude=held_out)
        if train_frame.empty or test_frame.empty:
            continue
        models = train_quantile_models(train_frame, feature_cols, num_boost_round=200)
        pred = predict_quantiles(models, test_frame, feature_cols)
        truth = test_frame["y"].values
        rmse = float(np.sqrt(np.nanmean((truth - pred["pm25_p50"].values) ** 2)))
        per_station[held_out] = {"rmse": round(rmse, 2), "n": len(test_frame)}
        all_true.extend(truth)
        all_pred.extend(pred["pm25_p50"].values)
    overall = float(np.sqrt(np.nanmean((np.array(all_true) - np.array(all_pred)) ** 2))) if all_true else float("nan")
    return {"overall_rmse": round(overall, 2), "per_station": per_station, "n_stations": len(per_station)}


def run_city_loso(panels_by_city: dict[str, pd.DataFrame], horizons: list[int],
                   feature_cols: list[str]) -> dict:
    """Train on N-1 cities, test on the held-out city's real stations, as
    if the model had never seen that city (spec 5.2)."""
    per_city = {}
    for held_out, train_cities in city_loso_splits(list(panels_by_city)):
        train_panel = pd.concat([panels_by_city[c] for c in train_cities], ignore_index=True)
        train_frame = mask_unknown_city(build_features(train_panel, horizons))
        test_panel = panels_by_city[held_out]
        test_frame = build_features(test_panel, horizons)
        test_frame["city"] = pd.Categorical([UNKNOWN_CITY] * len(test_frame),
                                             categories=train_frame.city.cat.categories)
        if train_frame.empty or test_frame.dropna(subset=["y"]).empty:
            continue
        models = train_quantile_models(train_frame, feature_cols, num_boost_round=200)
        scored = test_frame.dropna(subset=["y"])
        pred = predict_quantiles(models, scored, feature_cols)
        rmse = float(np.sqrt(np.nanmean((scored["y"].values - pred["pm25_p50"].values) ** 2)))
        per_city[held_out] = {"rmse": round(rmse, 2), "n": len(scored)}
    return {"per_city": per_city}


if __name__ == "__main__":
    frame = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=24 * 300, freq="h", tz="UTC")})
    print("folds:", walk_forward_folds(frame))
    print("city-loso:", city_loso_splits(["bengaluru", "delhi", "chennai"]))
