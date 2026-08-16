"""Cross-validation strategies (spec sections 5.1, 5.2, 5.3). No single
fixed holdout — walk-forward folds, spatial-LOSO (Task 9), city-LOSO, and
real-event oversampling that touches ONLY sample weights, never a
synthetic ignition schedule (spec 5.3's hard rule)."""
import numpy as np
import pandas as pd

from intelligence.models.forecast.features import build_features, downcast_panel, station_cells_only
from intelligence.models.forecast.model import (
    train_quantile_models, predict_quantiles, mask_unknown_city, UNKNOWN_CITY,
)


def _align_city(frame: pd.DataFrame, categories, relabel_unknown: bool = False) -> pd.DataFrame:
    """Give `frame` the same `city` category set the training frame carries.
    Applied to BOTH LOSO functions' test frames so the two paths stay
    consistent — a reader should not have to work out why one aligns and the
    other does not."""
    out = frame.copy()
    values = [UNKNOWN_CITY] * len(out) if relabel_unknown else out["city"].astype(str)
    out["city"] = pd.Categorical(values, categories=categories)
    return out


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


def spatial_loso(panel: pd.DataFrame, horizons: list[int], feature_cols: list[str],
                  fires: pd.DataFrame | None = None) -> dict:
    """Direct analog of fusion.py::loso_validation, applied to the
    forecaster (spec 5.2). For each real station: retrain excluding it
    entirely from training, forecast its cell using composite-only features
    with itself excluded from its own composite (spec 3.1's self-exclusion
    rule via features.build_features's loso_exclude), score against its
    real held-out readings."""
    station_cells = sorted(panel[panel.pm25_station.notna()].cell.unique())
    per_station, all_true, all_pred = {}, [], []
    for i, held_out in enumerate(station_cells, 1):
        # Each fold rebuilds features for the WHOLE panel (composite fidelity
        # requires full-city context), so this loop is the dominant cost of
        # the whole training run on a real multi-city panel and can run
        # silently for a long time with no other output. One line per fold
        # is the difference between "still running" and "looks hung".
        print(f"[spatial_loso] fold {i}/{len(station_cells)}: holding out {held_out}")
        # ONE build on the FULL panel, filtered by cell afterwards. Slicing
        # the panel down to the held-out cell BEFORE building features (an
        # earlier version did) leaves composite_grid/positional_block with no
        # other station to compose from -- every spatial feature comes back
        # NaN -- and rebuilds the climatology from that single station, whose
        # cell/ward/city scopes then all collapse onto the held-out station's
        # own value, which is also the prediction target. The model is handed
        # the answer and nothing else, and the resulting "RMSE" measures
        # nothing. Filtering after the build keeps the composite real and the
        # self-exclusion (loso_exclude) honest.
        frame = build_features(panel, horizons, loso_exclude=held_out, fires=fires,
                                restrict_to_station_cells=True)
        train_frame = mask_unknown_city(
            frame[frame.cell != held_out].dropna(subset=["y"]))
        test_frame = frame[frame.cell == held_out]
        if train_frame.empty or test_frame.empty:
            continue
        test_frame = _align_city(test_frame, train_frame.city.cat.categories)
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
                   feature_cols: list[str],
                   fires_by_city: dict[str, pd.DataFrame] | None = None) -> dict:
    """Train on N-1 cities, test on the held-out city's real stations, as
    if the model had never seen that city (spec 5.2).

    LIMITATION, stated so nobody reads this as a stronger guarantee than it
    is: the held-out city's test frame is built from that city's OWN panel,
    so its spatial features (composite lags, positional block, nearest-station
    distance) are computed from that same city's remaining stations. This is
    zero information about the held-out CITY's learned behaviour — which is
    what this split measures — but it is NOT zero information about the
    held-out city's stations. Cell-level independence is spatial_loso's job,
    not this one's."""
    fires_by_city = fires_by_city or {}
    per_city = {}
    for held_out, train_cities in city_loso_splits(list(panels_by_city)):
        if not train_cities:
            continue   # a single-city registry has no N-1 split to make
        print(f"[run_city_loso] holding out {held_out}, training on {train_cities}")
        # re-downcast after concat: pd.concat reverts a categorical column
        # back to plain string dtype whenever the pieces' category sets
        # differ, which every per-city `city`/`cell`/`ward_id` column does.
        # station_cells_only first -- both frames below run with
        # restrict_to_station_cells=True, which discards every non-station
        # cell anyway (see that flag's docstring), so concatenating N-1
        # cities' FULL grids just to throw most of it away is pure waste.
        train_panel = downcast_panel(pd.concat(
            [station_cells_only(panels_by_city[c]) for c in train_cities], ignore_index=True))
        train_fires = [fires_by_city[c] for c in train_cities if c in fires_by_city]
        train_frame = mask_unknown_city(
            build_features(train_panel, horizons,
                           fires=pd.concat(train_fires, ignore_index=True) if train_fires else None,
                           restrict_to_station_cells=True)
            .dropna(subset=["y"]))
        test_panel = station_cells_only(panels_by_city[held_out])
        test_frame = build_features(test_panel, horizons, fires=fires_by_city.get(held_out),
                                     restrict_to_station_cells=True)
        if train_frame.empty or test_frame.dropna(subset=["y"]).empty:
            continue
        test_frame = _align_city(test_frame, train_frame.city.cat.categories, relabel_unknown=True)
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
