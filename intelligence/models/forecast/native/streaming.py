"""Generalized per-fold-unit disk-streaming primitives. Formalizes the
pattern proven today (scratch_disk_streamed_city_loso.py,
scratch_final_fit_disk_streamed.py): build one fold-unit's features (a
city, a held-out station, a time-window), stream its float32 array
straight to disk, free it from pandas/Python, move to the next unit. At
combine time, memmap each unit's file (OS pages it in from disk on
demand -- never fully resident) and concatenate into one array -- this
needs only ~1x the final array's size in RAM, not the ~2x pandas'
dropna().reset_index() needs during its own internal block consolidation
(the actual OOM this whole package exists to route around; see
docs/superpowers/specs/2026-08-21-local-native-training-pipeline-design.md
section 1)."""
import gc
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import intelligence.models.forecast.features as features_module
from intelligence.models.forecast.climatology import build_climatology
from intelligence.models.forecast.features import (
    build_features, downcast_panel, station_cells_only,
)
from intelligence.models.forecast.model import (
    mask_unknown_city, UNKNOWN_CITY, PARAMS, QUANTILES,
)
from intelligence.models.forecast.native.native_composite import composite_grid_native
from intelligence.models.forecast.validation import _align_city, city_loso_splits, event_weights

_real_composite_grid = features_module.composite_grid


@contextmanager
def _native_composite_grid():
    """Monkeypatches features_module's composite_grid reference (build_
    features does `from spatial import composite_grid`, a name binding in
    ITS OWN namespace -- patching spatial.composite_grid directly would not
    reach build_features' already-bound reference) for the duration of the
    `with` block, then restores the real pandas/NumPy implementation on
    exit -- including on an exception, via the `finally`.

    Not restoring this (an earlier version rebound composite_grid
    permanently and never put it back) meant every build_features call
    later in the SAME process -- including one a caller intended to run the
    real pandas path -- would silently run the Numba kernel instead. That
    is a trapdoor: test_city_loso_parity.py's ordering (pandas path called
    BEFORE the native path) happened to save it, but a harmless-looking
    reordering of two test calls would have made the pandas assertion
    compare native output against itself and pass vacuously. The context
    manager removes the ordering dependency entirely: each call to either
    orchestrator below is bracketed by its own install/restore."""
    features_module.composite_grid = composite_grid_native
    try:
        yield
    finally:
        features_module.composite_grid = _real_composite_grid


def _encode(frame: pd.DataFrame, feature_columns: list[str], label_col: str,
            city_codes: list[str] | None = None) -> np.ndarray:
    """Encodes `feature_columns` + `label_col` from `frame` into one float32
    array -- the shared logic behind both `stream_unit_to_disk` (writes it to
    disk) and the held-out test frame scoring in `run_city_loso_native`
    (keeps it in memory). `city_codes`, when given, fixes the integer
    encoding for the `city` column (so codes agree across every unit/frame
    encoded this way) -- required whenever `feature_columns` includes
    "city"."""
    out = np.empty((len(frame), len(feature_columns) + 1), dtype=np.float32)
    for i, col in enumerate(feature_columns):
        if col == "city":
            if city_codes is None:
                raise ValueError("city_codes is required when 'city' is a feature column")
            codes = pd.Categorical(frame["city"].astype(str), categories=city_codes).codes
            if (codes < 0).any():
                unmapped = sorted(set(frame["city"].astype(str)[codes < 0]))
                raise ValueError(f"city value(s) not present in city_codes: {unmapped}")
            out[:, i] = codes.astype(np.float32)
        else:
            out[:, i] = frame[col].to_numpy(dtype=np.float32)
    out[:, -1] = frame[label_col].to_numpy(dtype=np.float32)
    return out


def stream_unit_to_disk(frame: pd.DataFrame, path: Path,
                         feature_columns: list[str], label_col: str = "y",
                         city_codes: list[str] | None = None) -> int:
    """Writes `feature_columns` + `label_col` as one float32 .npy array to
    `path`. `city_codes`, when given, fixes the integer encoding for the
    `city` column (so codes agree across every unit written this way) --
    required whenever `feature_columns` includes "city". A city value not
    present in `city_codes` raises (see `_encode`) rather than silently
    encoding as -1, which LightGBM would otherwise read as a missing
    value instead of a mapping bug."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = _encode(frame, feature_columns, label_col, city_codes)
    np.save(path, out)
    n = len(out)
    del out
    gc.collect()
    return n


def combine_streamed_units(paths: list[Path]) -> np.ndarray:
    """Memmap-loads every path in `paths` and concatenates into one float32
    array. Deletes each input file after combining. The mmap handles are
    explicitly dropped (`del mmaps; gc.collect()`) before unlinking --
    without this, Windows raises PermissionError on a file still mapped
    (hit this exact error in today's proof-of-concept)."""
    mmaps = [np.load(p, mmap_mode="r") for p in paths]
    total_rows = sum(m.shape[0] for m in mmaps)
    n_cols = mmaps[0].shape[1]
    combined = np.empty((total_rows, n_cols), dtype=np.float32)
    pos = 0
    for m in mmaps:
        combined[pos:pos + m.shape[0]] = m
        pos += m.shape[0]

    # Explicitly close memmap handles to release file locks on Windows
    for m in mmaps:
        if hasattr(m, '_mmap'):
            m._mmap.close()
    del mmaps
    gc.collect()
    for p in paths:
        try:
            p.unlink()
        except PermissionError:
            pass  # Windows mmap handle lingering; harmless, not worth failing the run over

    return combined


def run_city_loso_native(panels_by_city: dict, horizons: list[int],
                          feature_cols: list[str],
                          fires_by_city: dict | None = None,
                          num_threads: int | None = None) -> dict:
    """Same return shape as validation.run_city_loso:
    {"per_city": {city: {"rmse": float, "n": int}}}. Streams each
    training city's features to disk one at a time instead of pooling all
    N-1 cities into one pandas frame -- see this module's own docstring
    for why.

    Calls `station_cells_only` on every city's panel (training AND
    held-out) before building features -- matching validation.run_city_
    loso's own robustness contract (it tolerates receiving raw, unfiltered
    panels). Without this, a caller that hands in unfiltered panels (a
    reasonable assumption given run_city_loso's own contract) would make
    this function build composite/positional features over every
    non-station cell in each city's full grid before discarding them --
    directly undermining the memory-bounding goal disk-streaming exists
    for (found in review, not by a test -- the parity test's own
    `_load_panels` helper happened to pre-filter, masking the gap).

    Uses `validation.city_loso_splits` for the (held_out, train_cities)
    pairs, same as validation.run_city_loso, instead of re-deriving
    train_cities inline -- a single-city `panels_by_city` (or any input
    that leaves a city with no training peers) then yields an empty
    `train_cities` for that split, and the fold is skipped (excluded from
    `per_city`, matching run_city_loso's own `if t` filter on its splits)
    rather than crashing on `pd.concat([])` or `combine_streamed_units([])`.

    `num_threads`: None (default) leaves LightGBM to auto-detect, same as
    before this parameter existed. Pass an explicit value to pin it --
    needed when comparing this function's RMSE against validation.
    run_city_loso's for parity: the two paths build their lgb.Dataset from
    different objects (a raw float32 ndarray here vs. a pandas DataFrame
    there), which can carry different auto-detected thread defaults, and
    LightGBM's histogram reduction order (hence its exact numeric output)
    is not guaranteed identical across thread counts even with the same
    data. MEASURED (real 3-city run, review fix round): pinning both paths
    to num_threads=4, then again to num_threads=1 (fully removing thread
    count as a variable), gave the IDENTICAL per-city RMSE delta as the
    unpinned baseline in both cases -- thread-count divergence is not a
    contributor here (see this package's parity test / the task report
    for the full before/after numbers).

    Each training city's build_features call below runs on THAT CITY's
    panel alone, unlike validation.run_city_loso's pooled build (all N-1
    cities concatenated into one panel first) -- so composite_grid's
    station pool for e.g. a Chennai cell draws only from Chennai stations
    here, vs. Chennai+Hyderabad+Ahmedabad stations (at effectively-zero
    weight) in the pandas path. QUANTIFIED, not assumed: composite_grid's
    weight is align*decay, decay=exp(-dist_km/DIST_DECAY_KM) with
    DIST_DECAY_KM=2.0 (spatial.py). At the real chennai-hyderabad distance
    (515km, the closest of this test's 3 city pairs), even the
    best-case/most-favorable weight (align=1.0, i.e. perfectly downwind)
    at an unrealistically extreme 20 m/s sustained surface wind is
    ~4.7e-6; at a genuinely realistic wind speed the weight is <1e-10.
    Real in-city station weights at real intra-city distances (0-50km)
    are O(0.01-1). The cross-city contribution is therefore many orders
    of magnitude below any feature's float32 precision floor and
    genuinely negligible, not merely assumed so (see
    scratch_out/finding2_composite_weight_check.py for the computation)."""
    fires_by_city = fires_by_city or {}
    cities = sorted(panels_by_city)
    city_codes = sorted(set(cities) | {UNKNOWN_CITY})
    per_city = {}

    with _native_composite_grid():
        for held_out, train_cities in city_loso_splits(cities):
            if not train_cities:
                continue
            clim_tables = build_climatology(downcast_panel(pd.concat(
                [station_cells_only(panels_by_city[c]) for c in train_cities],
                ignore_index=True)))

            work_dir = Path("scratch_out") / "native_city_loso" / held_out
            paths = []
            for i, city in enumerate(train_cities):
                frame = mask_unknown_city(
                    build_features(station_cells_only(panels_by_city[city]), horizons,
                                    fires=fires_by_city.get(city),
                                    restrict_to_station_cells=True,
                                    clim_tables=clim_tables).dropna(subset=["y"]),
                    seed=i,
                )
                path = work_dir / f"{city}.npy"
                stream_unit_to_disk(frame, path, feature_cols, city_codes=city_codes)
                paths.append(path)
                del frame
                gc.collect()

            if not paths:
                continue

            combined = combine_streamed_units(paths)
            city_col_idx = feature_cols.index("city")
            X, y = combined[:, :-1], combined[:, -1]

            params = {**PARAMS, "alpha": 0.5}
            if num_threads is not None:
                params["num_threads"] = num_threads
            ds = lgb.Dataset(X, label=y, categorical_feature=[city_col_idx])
            model = lgb.train(params, ds, num_boost_round=200)
            del combined, X, y, ds
            gc.collect()

            test_frame = build_features(station_cells_only(panels_by_city[held_out]), horizons,
                                         fires=fires_by_city.get(held_out),
                                         restrict_to_station_cells=True,
                                         clim_tables=clim_tables)
            test_frame = _align_city(test_frame, city_codes, relabel_unknown=True)
            scored = test_frame.dropna(subset=["y"])
            test_arr = _encode(scored, feature_cols, "y", city_codes)

            pred = model.predict(test_arr[:, :-1])
            rmse = float(np.sqrt(np.nanmean((test_arr[:, -1] - pred) ** 2)))
            per_city[held_out] = {"rmse": round(rmse, 2), "n": len(scored)}

    return {"per_city": per_city}


def run_final_fit_native(panels_by_city: dict, horizons: list[int],
                          feature_cols: list[str],
                          fires_by_city: dict | None = None) -> dict:
    """Same return shape as train.py's train_and_promote `final_models`
    dict: one lgb.Booster per quantile in QUANTILES, weighted the same
    way (event_weights, 4x boost on rows with real trailing FIRMS
    activity) as the pandas served model there. Unlike city-LOSO, there
    is no held-out city here -- climatology is built once from ALL
    cities pooled and every city streams into ONE combined training set.

    POSITIONAL ENCODING, not name-realigned: unlike the pandas-served
    path (`intelligence/models/forecast/__init__.py::_predict_field`),
    which hands LightGBM a pandas Categorical column that gets realigned
    by category NAME at predict time, this function builds its
    lgb.Dataset from a raw float32 ndarray with `city` pre-encoded to an
    integer via `city_codes`. The trained booster therefore has no
    name-based recovery: whatever code later calls .predict() on it MUST
    encode `city` using this exact same `city_codes` ordering
    (`sorted(set(cities) | {UNKNOWN_CITY})`, matching the serving path's
    own convention in `__init__.py`'s `city_categories`) or predictions
    silently corrupt rather than raise.

    Calls `station_cells_only` on every city's panel before building
    features -- both for the pooled climatology build and for each
    city's own feature frame. Same fix run_city_loso_native needed (see
    its docstring): a caller handing in an unfiltered panel would
    otherwise build composite/positional features over every non-station
    cell in a city's full grid before discarding them, defeating the
    memory-bounding point of streaming to disk in the first place.
    station_cells_only is a pure subset filter, so this is idempotent
    when the caller already filtered (as this module's own parity
    test's `_load_panels` helper does)."""
    fires_by_city = fires_by_city or {}
    cities = sorted(panels_by_city)
    city_codes = sorted(set(cities) | {UNKNOWN_CITY})

    with _native_composite_grid():
        pooled_station_panel = downcast_panel(pd.concat(
            [station_cells_only(panels_by_city[c]) for c in cities], ignore_index=True))
        clim_tables = build_climatology(pooled_station_panel)
        del pooled_station_panel
        gc.collect()

        work_dir = Path("scratch_out") / "native_final_fit"
        paths = []
        for i, city in enumerate(cities):
            frame = mask_unknown_city(
                build_features(station_cells_only(panels_by_city[city]), horizons,
                                fires=fires_by_city.get(city),
                                restrict_to_station_cells=True,
                                clim_tables=clim_tables).dropna(subset=["y"]),
                seed=i,
            )
            path = work_dir / f"{city}.npy"
            stream_unit_to_disk(frame, path, feature_cols, city_codes=city_codes)
            paths.append(path)
            del frame
            gc.collect()

        combined = combine_streamed_units(paths)
        city_col_idx = feature_cols.index("city")
        fires_col_idx = feature_cols.index("fires_6h")
        X, y = combined[:, :-1], combined[:, -1]
        # Same weighting train.py's train_and_promote applies to the served
        # model -- reused via event_weights rather than reimplemented inline,
        # so a future change to the boost factor or fire column name doesn't
        # have to be kept in sync by hand across the pandas and native paths.
        weight = event_weights(pd.DataFrame({"fires_6h": X[:, fires_col_idx]})).astype(np.float32)

        ds = lgb.Dataset(X, label=y, weight=weight, categorical_feature=[city_col_idx])
        # Force binning up front, single-threaded -- if left lazy, two threads
        # calling train() on the same not-yet-constructed Dataset at once
        # would race on that first construction step (same reasoning as
        # train.py's train_and_promote for its own concurrent quantile fits).
        ds.construct()
        threads_per_quantile = max(1, (os.cpu_count() or 4) // len(QUANTILES))

        def _fit_one(q):
            params = {**PARAMS, "alpha": q, "num_threads": threads_per_quantile}
            return q, lgb.train(params, ds, num_boost_round=500)

        final_models = {}
        with ThreadPoolExecutor(max_workers=len(QUANTILES)) as ex:
            for q, model in ex.map(_fit_one, QUANTILES):
                final_models[q] = model

        del combined, X, y, ds
        gc.collect()

    return final_models


def run_spatial_loso_native(panel: pd.DataFrame, horizons: list[int],
                             feature_cols: list[str],
                             fires: pd.DataFrame | None = None,
                             num_threads: int | None = None) -> dict:
    """Same return shape as validation.spatial_loso:
    {"overall_rmse": float, "baseline_rmse": float,
     "per_station": {cell: {"rmse", "n", "baseline_rmse"}}, "n_stations": int}.

    Unlike city-LOSO (whose training pool is N-1 CITIES), spatial-LOSO's
    training pool is ALL 8 cities minus ONE STATION -- so every fold needs
    every city's features, not a small per-city subset. This still
    decomposes per-city, exactly like Phase 0: each of the (up to) 8
    cities' build_features call is independent and small; only the ONE
    city containing the held-out station passes `loso_exclude`. Climatology
    is rebuilt from the pooled (all-cities, held-out-cell-EXCLUDED) station
    panel once per fold -- cheap, since it's pre-horizon-expansion -- and
    shared into every city's build_features call via `clim_tables`, exactly
    matching build_climatology's own documented exclude_cell contract (it
    must drop that cell from all three scopes, not just its own, to avoid
    a single-station ward's climatology being that station's own history
    in disguise -- see climatology.py's own docstring)."""
    panel = downcast_panel(station_cells_only(panel))
    station_cells = sorted(panel[panel.pm25_station.notna()].cell.unique())
    if not station_cells:
        return {"overall_rmse": float("nan"), "baseline_rmse": float("nan"),
                "per_station": {}, "n_stations": 0}

    cities = sorted(panel.city.unique())
    city_codes = sorted(set(cities) | {UNKNOWN_CITY})
    per_station = {}
    all_true, all_pred, all_baseline = [], [], []

    for fold_i, held_out in enumerate(station_cells):
        held_out_city = panel.loc[panel.cell == held_out, "city"].iloc[0]

        with _native_composite_grid():
            clim_tables = build_climatology(panel, exclude_cell=held_out)

            work_dir = Path("scratch_out") / "native_spatial_loso" / held_out
            paths = []
            for i, city in enumerate(cities):
                city_panel = panel[panel.city == city]
                loso_exclude = held_out if city == held_out_city else None
                built = build_features(city_panel, horizons, loso_exclude=loso_exclude,
                                        fires=fires, restrict_to_station_cells=True,
                                        clim_tables=clim_tables)
                # frame.cell != held_out, matching _run_one_loso_fold's own
                # train_frame filter exactly -- loso_exclude only excludes
                # the held-out cell from its OWN composite feature (self-
                # exclusion), it does not drop that cell's rows from the
                # output. Without this filter the held-out station's real
                # historical y labels stream straight into training, and
                # the model is scored against data it was trained on --
                # caught by the fast single-city sanity check the plan's
                # own review step called for (RMSE undershot pandas by
                # 6-10 points on a real ahmedabad 2-station check, far past
                # any float32/mask_unknown_city noise floor).
                frame = mask_unknown_city(
                    built[built.cell != held_out].dropna(subset=["y"]),
                    seed=i,
                )
                path = work_dir / f"{city}.npy"
                stream_unit_to_disk(frame, path, feature_cols, city_codes=city_codes)
                paths.append(path)
                del frame
                gc.collect()

            combined = combine_streamed_units(paths)
            city_col_idx = feature_cols.index("city")
            X, y = combined[:, :-1], combined[:, -1]

            params = {**PARAMS, "alpha": 0.5}
            if num_threads is not None:
                params["num_threads"] = num_threads
            ds = lgb.Dataset(X, label=y, categorical_feature=[city_col_idx])
            model = lgb.train(params, ds, num_boost_round=200)
            del combined, X, y, ds
            gc.collect()

            # Test frame: the held-out city's own panel, same clim_tables,
            # loso_exclude applied -- matches _run_one_loso_fold's own
            # "one build, filter after" discipline (features.py's
            # composite_grid needs the held-out station's neighbours
            # present to compose a real, non-degenerate self-exclusion
            # value; slicing to that one cell BEFORE building would leave
            # every spatial feature NaN).
            held_out_city_panel = panel[panel.city == held_out_city]
            test_frame_full = build_features(held_out_city_panel, horizons,
                                              loso_exclude=held_out, fires=fires,
                                              restrict_to_station_cells=True,
                                              clim_tables=clim_tables)
            test_frame = test_frame_full[test_frame_full.cell == held_out]

        if test_frame.empty:
            continue
        test_frame = test_frame.copy()
        test_arr = _encode(test_frame, feature_cols, "y", city_codes)
        pred = model.predict(test_arr[:, :-1])
        truth = test_arr[:, -1]
        rmse = float(np.sqrt(np.nanmean((truth - pred) ** 2)))
        baseline_pred = test_frame["lag_0"].to_numpy(dtype=np.float32)
        baseline_rmse = float(np.sqrt(np.nanmean((truth - baseline_pred) ** 2)))
        per_station[held_out] = {"rmse": round(rmse, 2), "n": len(test_frame),
                                  "baseline_rmse": round(baseline_rmse, 2)}
        all_true.extend(truth.tolist())
        all_pred.extend(pred.tolist())
        all_baseline.extend(baseline_pred.tolist())

        print(f"[spatial_loso_native] fold {fold_i + 1}/{len(station_cells)}: "
              f"{held_out} rmse={per_station.get(held_out, {}).get('rmse')}")

    overall = (float(np.sqrt(np.nanmean((np.array(all_true) - np.array(all_pred)) ** 2)))
               if all_true else float("nan"))
    baseline_overall = (float(np.sqrt(np.nanmean((np.array(all_true) - np.array(all_baseline)) ** 2)))
                        if all_true else float("nan"))
    return {"overall_rmse": round(overall, 2), "baseline_rmse": round(baseline_overall, 2),
            "per_station": per_station, "n_stations": len(per_station)}
