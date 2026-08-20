"""Cross-validation strategies (spec sections 5.1, 5.2, 5.3). No single
fixed holdout — walk-forward folds, spatial-LOSO (Task 9), city-LOSO, and
real-event oversampling that touches ONLY sample weights, never a
synthetic ignition schedule (spec 5.3's hard rule)."""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from intelligence.models.forecast.features import (
    build_features, attach_climatology, downcast_panel, station_cells_only,
)
from intelligence.models.forecast.climatology import build_climatology
from intelligence.models.forecast.checkpoint import load_fold, save_fold
from intelligence.models.forecast.model import (
    train_quantile_models, predict_quantiles, train_ceiling_baseline, mask_unknown_city, UNKNOWN_CITY,
)
from intelligence.models.forecast.eval import skill_vs_baseline


def _align_city(frame: pd.DataFrame, categories, relabel_unknown: bool = False) -> pd.DataFrame:
    """Give `frame` the same `city` category set the training frame carries.
    Applied to spatial_loso/city_loso/walk_forward's test frames so all
    three paths stay consistent — a reader should not have to work out why
    one aligns and the others do not.

    `relabel_unknown=True` (city_loso): the WHOLE test frame is one held-out
    city that LOSO's own design guarantees train has never seen -- blanket-
    relabel every row to UNKNOWN_CITY.

    `relabel_unknown=False` (spatial_loso, walk_forward): most test rows'
    cities WERE already seen in train; only the ones that genuinely weren't
    get relabelled to UNKNOWN_CITY -- e.g. a city whose real data starts
    AFTER a given walk-forward fold's train_end (a real possibility once
    cities have staggered start dates in the pooled panel, not a
    synthetic-only edge case: caught by a real 8-city sanity run, not a
    unit test), or a spatial_loso city with exactly one real station, whose
    held-out cell leaves train with zero rows from that city.
    FIXED: naively casting straight to a Categorical restricted to
    `categories` (the old behaviour) silently turned any unmatched value
    into NaN instead -- and NaN can't be sorted/compared against real city
    strings downstream, which crashed eval.py's event_by_outcome with
    "TypeError: '<' not supported between instances of 'float' and 'str'"
    on exactly that staggered-start-date scenario."""
    out = frame.copy()
    if relabel_unknown:
        out["city"] = pd.Categorical([UNKNOWN_CITY] * len(out), categories=categories)
        return out
    values = out["city"].astype(str)
    known = set(categories)
    values = values.where(values.isin(known), UNKNOWN_CITY)
    cats = categories if UNKNOWN_CITY in known else list(categories) + [UNKNOWN_CITY]
    out["city"] = pd.Categorical(values, categories=cats)
    return out


def walk_forward_folds(frame: pd.DataFrame, ts_col: str = "ts",
                        min_train_days: int = 180, test_days: int = 42,
                        step_days: int = 42) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Yields (train_end, test_start, test_end) triples. Train is always
    "everything up to train_end" (expanding window); test is the following
    `test_days`. Returns [] rather than raising when there isn't enough
    history for even one fold — callers must check.

    `test_days` and `step_days` default equal (42/42, halved from the first
    run's 21/21) and MUST be changed together. When `step_days == test_days`,
    fold N+1's test window starts exactly where fold N's ends — every day in
    range gets tested exactly once. Raising `step_days` alone (leaving
    `test_days` behind) opens a `step_days - test_days` gap between every
    pair of folds: on the real 8-city panel, step_days=42/test_days=21 tests
    only 48% of the calendar (verified directly against this function's own
    output), silently dropping the other half from walk_forward_skill_median,
    the OOF interval calibration, and quiet_vs_event — exactly the kind of
    evaluation blind spot spec principle 8 exists to catch. Moving both
    together halves the fold count (cheaper) while keeping the same 96%
    contiguous coverage the original 21/21 config had."""
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


def _run_one_loso_fold(panel: pd.DataFrame, horizons: list[int], feature_cols: list[str],
                        fires: pd.DataFrame | None, held_out: str,
                        num_threads: int | None) -> dict | None:
    """One spatial-LOSO fold's actual work, factored out so both the
    sequential loop and the parallel (ProcessPoolExecutor) path in
    spatial_loso call the exact same code -- the two paths must be provably
    identical, not just similar, since only one of them is what every
    existing test exercises."""
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
        return None
    test_frame = _align_city(test_frame, train_frame.city.cat.categories)
    models = train_quantile_models(train_frame, feature_cols, num_boost_round=200,
                                    num_threads=num_threads)
    pred = predict_quantiles(models, test_frame, feature_cols)
    truth = test_frame["y"].values
    rmse = float(np.sqrt(np.nanmean((truth - pred["pm25_p50"].values) ** 2)))
    # Persistence baseline ("PM2.5 stays what it last was"), scored on the
    # SAME held-out rows -- lag_0 is already a feature, so this costs
    # nothing extra to compute. spatial_loso_rmse has had no comparator:
    # 40.51 alone doesn't say whether the model is doing anything a naive
    # guess wouldn't already do just as well.
    baseline_rmse = float(np.sqrt(np.nanmean((truth - test_frame["lag_0"].values) ** 2)))
    return {"held_out": held_out, "rmse": round(rmse, 2), "n": len(test_frame),
            "truth": truth, "pred": pred["pm25_p50"].values,
            "baseline_pred": test_frame["lag_0"].values, "baseline_rmse": round(baseline_rmse, 2)}


# Set once per worker PROCESS by _init_loso_worker, read by _loso_worker_task.
# ProcessPoolExecutor's initializer runs once when a worker starts (not once
# per submitted task), so the panel/horizons/etc. get pickled to each worker
# ONE time no matter how many folds that worker goes on to run -- submitting
# 57 tasks each carrying the full panel would re-pickle a multi-hundred-MB
# object 57 times for no reason.
_LOSO_WORKER_STATE: dict = {}


def _init_loso_worker(panel, horizons, feature_cols, fires, num_threads):
    _LOSO_WORKER_STATE.update(panel=panel, horizons=horizons, feature_cols=feature_cols,
                               fires=fires, num_threads=num_threads)


def _loso_worker_task(held_out: str) -> dict | None:
    s = _LOSO_WORKER_STATE
    return _run_one_loso_fold(s["panel"], s["horizons"], s["feature_cols"], s["fires"],
                               held_out, s["num_threads"])


def spatial_loso(panel: pd.DataFrame, horizons: list[int], feature_cols: list[str],
                  fires: pd.DataFrame | None = None,
                  max_workers: int | None = None, threads_per_fold: int = 2,
                  checkpoint_dir: str | None = None) -> dict:
    """Direct analog of fusion.py::loso_validation, applied to the
    forecaster (spec 5.2). For each real station: retrain excluding it
    entirely from training, forecast its cell using composite-only features
    with itself excluded from its own composite (spec 3.1's self-exclusion
    rule via features.build_features's loso_exclude), score against its
    real held-out readings.

    `max_workers`: None (default) runs every fold sequentially in THIS
    process, one after another -- unchanged from before this parameter
    existed, and what every existing test exercises (monkeypatching
    train_quantile_models/predict_quantiles only affects the current
    process; a ProcessPoolExecutor worker re-imports the module fresh and
    would silently see the REAL functions instead, breaking that test
    coverage). Pass an integer to run that many folds concurrently in
    separate processes instead -- train_and_promote does, for the real
    training run. Each concurrent fold's LightGBM calls are capped to
    `threads_per_fold` threads (default 2) so N concurrent folds don't
    oversubscribe the machine the way N folds each grabbing every core
    would. Measured on real 4-city data: one fold is ~7 min wall-clock
    (~104s feature build, ~313s LightGBM training on ~8M rows, ~4s
    predict) -- with 57 real stations across delhi/chennai/bengaluru/
    mumbai, sequential is ~6.5-7h; this is the dominant cost of the whole
    training run.

    `checkpoint_dir`: None (default) disables checkpointing -- identical to
    before this parameter existed. When set, each held-out station's result
    is cached to `checkpoint_dir/spatial_loso/<station>.pkl` as it completes
    and re-loaded (never recomputed) on a later call with the SAME
    checkpoint_dir -- the resume path for a killed-and-relaunched training
    Job. See checkpoint.py's own docstring for what does and doesn't
    survive a Job restart."""
    station_cells = sorted(panel[panel.pm25_station.notna()].cell.unique())
    if not station_cells:
        return {"overall_rmse": float("nan"), "baseline_rmse": float("nan"),
                "per_station": {}, "n_stations": 0}

    per_station, all_true, all_pred, all_baseline = {}, [], [], []

    def _record(result):
        if result is None:
            return
        per_station[result["held_out"]] = {"rmse": result["rmse"], "n": result["n"],
                                            "baseline_rmse": result["baseline_rmse"]}
        all_true.extend(result["truth"])
        all_pred.extend(result["pred"])
        all_baseline.extend(result["baseline_pred"])

    if max_workers is None or max_workers <= 1:
        for i, held_out in enumerate(station_cells, 1):
            found, cached = load_fold(checkpoint_dir, "spatial_loso", held_out)
            if found:
                print(f"[spatial_loso] fold {i}/{len(station_cells)}: {held_out} -- resumed from checkpoint")
                _record(cached)
                continue
            # Each fold rebuilds features for the WHOLE panel (composite
            # fidelity requires full-city context), so this loop is the
            # dominant cost of the whole training run on a real multi-city
            # panel and can run silently for a long time with no other
            # output. One line per fold is the difference between "still
            # running" and "looks hung".
            print(f"[spatial_loso] fold {i}/{len(station_cells)}: holding out {held_out}")
            t0 = time.perf_counter()
            result = _run_one_loso_fold(panel, horizons, feature_cols, fires, held_out,
                                         num_threads=None)
            print(f"[spatial_loso] fold {i}/{len(station_cells)} done in "
                  f"{time.perf_counter() - t0:.0f}s")
            save_fold(checkpoint_dir, "spatial_loso", held_out, result)
            _record(result)
    else:
        # Checkpointed stations are resolved BEFORE submission, not filtered
        # out of the results after -- a resumed station has no reason to pay
        # for a worker process and a full feature rebuild just to be thrown
        # away.
        pending = []
        for held_out in station_cells:
            found, cached = load_fold(checkpoint_dir, "spatial_loso", held_out)
            if found:
                print(f"[spatial_loso] {held_out} -- resumed from checkpoint")
                _record(cached)
            else:
                pending.append(held_out)
        if pending:
            print(f"[spatial_loso] {len(pending)}/{len(station_cells)} stations pending, "
                  f"{max_workers} concurrent workers x {threads_per_fold} threads/fold "
                  f"(cpu_count={os.cpu_count()})")
            done = 0
            t_start = time.perf_counter()
            with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_loso_worker,
                                      initargs=(panel, horizons, feature_cols, fires,
                                                threads_per_fold)) as ex:
                futures = {ex.submit(_loso_worker_task, held_out): held_out
                           for held_out in pending}
                # as_completed, not the submission order -- folds finish whenever
                # they finish, and blocking on submission order would waste every
                # worker that raced ahead while we wait on a slow one specifically.
                for future in as_completed(futures):
                    done += 1
                    held_out = futures[future]
                    print(f"[spatial_loso] fold {done}/{len(pending)} complete: {held_out} "
                          f"(elapsed {time.perf_counter() - t_start:.0f}s total)")
                    result = future.result()
                    save_fold(checkpoint_dir, "spatial_loso", held_out, result)
                    _record(result)

    overall = float(np.sqrt(np.nanmean((np.array(all_true) - np.array(all_pred)) ** 2))) if all_true else float("nan")
    # Persistence RMSE across the SAME pooled held-out rows the model's own
    # overall_rmse is scored on -- the direct answer to "would a naive guess
    # have done just as well", which spatial_loso_rmse alone cannot answer.
    baseline_overall = (float(np.sqrt(np.nanmean((np.array(all_true) - np.array(all_baseline)) ** 2)))
                        if all_true else float("nan"))
    return {"overall_rmse": round(overall, 2), "baseline_rmse": round(baseline_overall, 2),
            "per_station": per_station, "n_stations": len(per_station)}


def _run_one_city_loso_fold(panels_by_city: dict[str, pd.DataFrame], horizons: list[int],
                             feature_cols: list[str], fires_by_city: dict,
                             held_out: str, train_cities: list[str],
                             num_threads: int | None) -> dict | None:
    """One city-LOSO fold, factored out for the same reason as
    _run_one_loso_fold: the sequential loop and the parallel path must call
    provably identical code."""
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
    # Climatology built ONCE, from train_panel only, and reused for the test
    # build below -- the held-out city is entirely absent from train_panel
    # (it's in train_cities' complement by construction), so every one of
    # its cell/ward/city identifiers will correctly miss every lookup and
    # come back NaN, exactly the same "genuinely never seen this place"
    # signal a real deployment to a new city would produce. Without this,
    # build_features' default path builds climatology from the TEST call's
    # own panel (the held-out city's own full history), so a (cell,dow-hour)
    # or (cell,month) bucket's median is partly computed from that same
    # row's own future/co-located readings -- the identical leak class
    # spatial_loso already guards against via loso_exclude (see
    # build_climatology's own docstring) and run_walk_forward guards against
    # via its explicit train-only rebuild. city_loso had never gotten the
    # same treatment; caught by an explicit pre-retrain audit, not by the
    # (insufficient) existing test coverage, which only checked that every
    # city appears in the output, never that its climatology features were
    # leak-free the way test_spatial_loso_test_frame_sees_the_other_
    # stations_not_its_own_answer already does for spatial_loso.
    clim_tables = build_climatology(train_panel)
    train_frame = mask_unknown_city(
        build_features(train_panel, horizons,
                       fires=pd.concat(train_fires, ignore_index=True) if train_fires else None,
                       restrict_to_station_cells=True, clim_tables=clim_tables)
        .dropna(subset=["y"]))
    test_panel = station_cells_only(panels_by_city[held_out])
    test_frame = build_features(test_panel, horizons, fires=fires_by_city.get(held_out),
                                 restrict_to_station_cells=True, clim_tables=clim_tables)
    if train_frame.empty or test_frame.dropna(subset=["y"]).empty:
        return None
    test_frame = _align_city(test_frame, train_frame.city.cat.categories, relabel_unknown=True)
    models = train_quantile_models(train_frame, feature_cols, num_boost_round=200,
                                    num_threads=num_threads)
    scored = test_frame.dropna(subset=["y"])
    pred = predict_quantiles(models, scored, feature_cols)
    rmse = float(np.sqrt(np.nanmean((scored["y"].values - pred["pm25_p50"].values) ** 2)))
    return {"held_out": held_out, "rmse": round(rmse, 2), "n": len(scored)}


_CITY_LOSO_WORKER_STATE: dict = {}


def _init_city_loso_worker(panels_by_city, horizons, feature_cols, fires_by_city, num_threads):
    _CITY_LOSO_WORKER_STATE.update(panels_by_city=panels_by_city, horizons=horizons,
                                    feature_cols=feature_cols, fires_by_city=fires_by_city,
                                    num_threads=num_threads)


def _city_loso_worker_task(args: tuple[str, list[str]]) -> dict | None:
    held_out, train_cities = args
    s = _CITY_LOSO_WORKER_STATE
    return _run_one_city_loso_fold(s["panels_by_city"], s["horizons"], s["feature_cols"],
                                    s["fires_by_city"], held_out, train_cities, s["num_threads"])


def run_city_loso(panels_by_city: dict[str, pd.DataFrame], horizons: list[int],
                   feature_cols: list[str],
                   fires_by_city: dict[str, pd.DataFrame] | None = None,
                   max_workers: int | None = None, threads_per_fold: int = 2,
                   checkpoint_dir: str | None = None) -> dict:
    """Train on N-1 cities, test on the held-out city's real stations, as
    if the model had never seen that city (spec 5.2).

    LIMITATION, stated so nobody reads this as a stronger guarantee than it
    is: the held-out city's test frame is built from that city's OWN panel,
    so its spatial features (composite lags, positional block, nearest-station
    distance) are computed from that same city's remaining stations. This is
    zero information about the held-out CITY's learned behaviour — which is
    what this split measures — but it is NOT zero information about the
    held-out city's stations. Cell-level independence is spatial_loso's job,
    not this one's.

    `max_workers`: same contract as spatial_loso's -- None (default) runs
    sequentially in this process (what every existing test exercises); an
    integer runs that many city-folds concurrently in separate processes,
    each fold's LightGBM capped to `threads_per_fold` threads.

    `checkpoint_dir`: same contract as spatial_loso's -- see that docstring
    and checkpoint.py."""
    fires_by_city = fires_by_city or {}
    splits = [(h, t) for h, t in city_loso_splits(list(panels_by_city)) if t]
    per_city = {}

    def _record(result):
        if result is not None:
            per_city[result["held_out"]] = {"rmse": result["rmse"], "n": result["n"]}

    if max_workers is None or max_workers <= 1:
        for held_out, train_cities in splits:
            found, cached = load_fold(checkpoint_dir, "city_loso", held_out)
            if found:
                print(f"[run_city_loso] {held_out} -- resumed from checkpoint")
                _record(cached)
                continue
            print(f"[run_city_loso] holding out {held_out}, training on {train_cities}")
            t0 = time.perf_counter()
            result = _run_one_city_loso_fold(panels_by_city, horizons, feature_cols,
                                              fires_by_city, held_out, train_cities,
                                              num_threads=None)
            print(f"[run_city_loso] {held_out} done in {time.perf_counter() - t0:.0f}s")
            save_fold(checkpoint_dir, "city_loso", held_out, result)
            _record(result)
    else:
        pending = []
        for held_out, train_cities in splits:
            found, cached = load_fold(checkpoint_dir, "city_loso", held_out)
            if found:
                print(f"[run_city_loso] {held_out} -- resumed from checkpoint")
                _record(cached)
            else:
                pending.append((held_out, train_cities))
        if pending:
            print(f"[run_city_loso] {len(pending)}/{len(splits)} cities pending, "
                  f"{max_workers} concurrent workers x {threads_per_fold} threads/fold")
            t_start = time.perf_counter()
            with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_city_loso_worker,
                                      initargs=(panels_by_city, horizons, feature_cols,
                                                fires_by_city, threads_per_fold)) as ex:
                futures = {ex.submit(_city_loso_worker_task, split): split[0] for split in pending}
                done = 0
                for future in as_completed(futures):
                    done += 1
                    held_out = futures[future]
                    print(f"[run_city_loso] fold {done}/{len(pending)} complete: {held_out} "
                          f"(elapsed {time.perf_counter() - t_start:.0f}s total)")
                    result = future.result()
                    save_fold(checkpoint_dir, "city_loso", held_out, result)
                    _record(result)

    return {"per_city": per_city}


def _run_one_walk_forward_fold(full_panel: pd.DataFrame, frame: pd.DataFrame,
                                feature_cols: list[str], num_cols: list[str],
                                train_end: pd.Timestamp, test_start: pd.Timestamp,
                                test_end: pd.Timestamp, num_threads: int | None) -> dict | None:
    """One walk-forward fold, factored out for the same reason as the two
    LOSO functions' per-fold helpers: sequential and parallel must call
    provably identical code, not just similar code."""
    # Climatology from TRAIN ONLY, re-attached to both sides of the fold.
    # build_features built it from the whole panel, which means every test
    # row's clim_dow_hour/clim_month was partly computed from that same
    # row's own future readings — the deleted forecast.py called this out
    # explicitly ("from TRAIN ONLY. The tough baseline.") and the
    # discipline has to survive the rewrite. Re-attaching is a merge, not
    # a second feature build: the expensive spatial work is untouched.
    clim = build_climatology(full_panel[full_panel.ts <= train_end])
    tr = attach_climatology(frame[frame.ts <= train_end], clim).dropna(subset=["y"])
    te = attach_climatology(
        frame[(frame.ts > test_start) & (frame.ts <= test_end)], clim).dropna(subset=["y"])
    if tr.empty or te.empty:
        return None
    # mask_unknown_city AFTER the train/test split, applied ONLY to tr —
    # matching _run_one_loso_fold/_run_one_city_loso_fold's own pattern
    # (mask_unknown_city(train_frame), test_frame never touched). Train.py
    # used to mask the whole pooled frame ONCE before any fold existed, so
    # ~5% of every fold's TEST rows carried an artificially withheld city
    # label too — a real serving call always knows its own city, so this
    # was measuring skill on a task strictly harder than the one being
    # shipped, for a random 5% of every fold's test set. _align_city gives
    # `te` tr's expanded category set (now including "unknown") WITHOUT
    # relabelling any of its real values, so LightGBM's categorical code
    # mapping stays consistent between train and test — the same reason
    # the two LOSO folds call it on their own test frames.
    tr = mask_unknown_city(tr)
    te = _align_city(te, tr.city.cat.categories)
    n_train = len(tr)
    models = train_quantile_models(tr, feature_cols, num_boost_round=300, num_threads=num_threads)
    pred = predict_quantiles(models, te, feature_cols)
    skill = skill_vs_baseline(te["y"].values, pred["pm25_p50"].values, te["lag_0"].values)
    # The linear ceiling baseline is fit per fold on a capped sample:
    # sklearn's QuantileRegressor solves an LP whose cost grows fast in n,
    # and a median line does not need millions of rows to be identified.
    ceil_fit = tr.sample(min(len(tr), 50_000), random_state=0)
    ceil = train_ceiling_baseline(ceil_fit, num_cols)
    ceil_pred = ceil.predict(te[num_cols].select_dtypes(include=[np.number]).fillna(0.0))
    oof_frame = pd.DataFrame({
        "y": te["y"].values, "p10": pred["pm25_p10"].values,
        "p50": pred["pm25_p50"].values, "p90": pred["pm25_p90"].values,
        "ceiling": ceil_pred, "fires_6h": te["fires_6h"].fillna(0).values,
        # Needed to stratify quiet_vs_event by city -- pooled-only reporting
        # is exactly what hid the fires_6h city-mix confound (event-rich
        # cities happened to be the easy, low-RMSE ones).
        "city": te["city"].astype(str).values,
        # This fold's training set size, broadcast to every row -- lets a
        # caller select OOF rows from folds whose model was trained on a
        # data volume close to the FINAL full-data model's, instead of
        # pooling early folds (barely-trained) with late ones (well-trained)
        # as if they were equally representative. See train.py's interval
        # calibration, which uses exactly this to fix the walk-forward's
        # fold-immaturity bias in the p10/p90 coverage calibration.
        "n_train": n_train,
    })
    return {"skill": skill, "oof": oof_frame, "n_train": n_train}


_WF_WORKER_STATE: dict = {}


def _init_wf_worker(full_panel, frame, feature_cols, num_cols, num_threads):
    _WF_WORKER_STATE.update(full_panel=full_panel, frame=frame, feature_cols=feature_cols,
                             num_cols=num_cols, num_threads=num_threads)


def _wf_worker_task(bounds: tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]) -> dict | None:
    train_end, test_start, test_end = bounds
    s = _WF_WORKER_STATE
    return _run_one_walk_forward_fold(s["full_panel"], s["frame"], s["feature_cols"],
                                       s["num_cols"], train_end, test_start, test_end,
                                       s["num_threads"])


def run_walk_forward(full_panel: pd.DataFrame, frame: pd.DataFrame, feature_cols: list[str],
                      num_cols: list[str],
                      folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
                      max_workers: int | None = None, threads_per_fold: int = 2,
                      checkpoint_dir: str | None = None) -> dict:
    """Runs every walk_forward_folds() fold and returns
    {"fold_skills": [...], "oof": DataFrame|None}. Each fold is independent
    (a different expanding-window train/test split, no fold depends on
    another's output), same shape as spatial_loso/run_city_loso.

    `max_workers`: same contract as the two LOSO functions -- None
    (default) runs sequentially in this process (what every existing test
    exercises); an integer runs that many folds concurrently in separate
    processes, each capped to `threads_per_fold` LightGBM threads.

    `checkpoint_dir`: same contract as spatial_loso's -- see that docstring
    and checkpoint.py. Keyed on `train_end.isoformat()`, unique per fold by
    construction (walk_forward_folds never repeats a train_end)."""
    fold_skills, oof = [], []

    def _record(result):
        if result is not None:
            fold_skills.append(result["skill"])
            oof.append(result["oof"])

    if max_workers is None or max_workers <= 1:
        for fi, (train_end, test_start, test_end) in enumerate(folds, 1):
            key = train_end.isoformat()
            found, cached = load_fold(checkpoint_dir, "walk_forward", key)
            if found:
                extra = f", n_train={cached['n_train']:,}" if cached else " (empty)"
                print(f"[walk_forward] fold {fi}/{len(folds)}: train_end={train_end.date()} "
                      f"-- resumed from checkpoint{extra}")
                _record(cached)
                continue
            print(f"[walk_forward] fold {fi}/{len(folds)}: train_end={train_end.date()}")
            t0 = time.perf_counter()
            result = _run_one_walk_forward_fold(full_panel, frame, feature_cols, num_cols,
                                                 train_end, test_start, test_end,
                                                 num_threads=None)
            if result is None:
                print(f"[walk_forward] fold {fi}/{len(folds)} empty, skipped "
                      f"({time.perf_counter() - t0:.0f}s)")
            else:
                print(f"[walk_forward] fold {fi}/{len(folds)} done in "
                      f"{time.perf_counter() - t0:.0f}s (train_end={train_end.date()}, "
                      f"n_train={result['n_train']:,})")
            save_fold(checkpoint_dir, "walk_forward", key, result)
            _record(result)
    else:
        pending = []
        for bounds in folds:
            train_end = bounds[0]
            key = train_end.isoformat()
            found, cached = load_fold(checkpoint_dir, "walk_forward", key)
            if found:
                extra = f", n_train={cached['n_train']:,}" if cached else " (empty)"
                print(f"[walk_forward] train_end={train_end.date()} -- resumed from checkpoint{extra}")
                _record(cached)
            else:
                pending.append(bounds)
        if pending:
            print(f"[walk_forward] {len(pending)}/{len(folds)} folds pending, "
                  f"{max_workers} concurrent workers x {threads_per_fold} threads/fold")
            t_start = time.perf_counter()
            with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_wf_worker,
                                      initargs=(full_panel, frame, feature_cols, num_cols,
                                                threads_per_fold)) as ex:
                futures = {ex.submit(_wf_worker_task, bounds): bounds for bounds in pending}
                done = 0
                for future in as_completed(futures):
                    done += 1
                    train_end = futures[future][0]
                    result = future.result()
                    extra = f", n_train={result['n_train']:,}" if result else ""
                    print(f"[walk_forward] fold {done}/{len(pending)} complete: "
                          f"train_end={train_end.date()}{extra} "
                          f"(elapsed {time.perf_counter() - t_start:.0f}s total)")
                    save_fold(checkpoint_dir, "walk_forward", train_end.isoformat(), result)
                    _record(result)

    o = pd.concat(oof, ignore_index=True) if oof else None
    return {"fold_skills": fold_skills, "oof": o}


if __name__ == "__main__":
    frame = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=24 * 300, freq="h", tz="UTC")})
    print("folds:", walk_forward_folds(frame))
    print("city-loso:", city_loso_splits(["bengaluru", "delhi", "chennai"]))
