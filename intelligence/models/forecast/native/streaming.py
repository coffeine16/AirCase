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
from pathlib import Path

import numpy as np
import pandas as pd


def stream_unit_to_disk(frame: pd.DataFrame, path: Path,
                         feature_columns: list[str], label_col: str = "y",
                         city_codes: list[str] | None = None) -> int:
    """Writes `feature_columns` + `label_col` as one float32 .npy array to
    `path`. `city_codes`, when given, fixes the integer encoding for the
    `city` column (so codes agree across every unit written this way) --
    required whenever `feature_columns` includes "city"."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.empty((len(frame), len(feature_columns) + 1), dtype=np.float32)
    for i, col in enumerate(feature_columns):
        if col == "city":
            if city_codes is None:
                raise ValueError("city_codes is required when 'city' is a feature column")
            codes = pd.Categorical(frame["city"].astype(str), categories=city_codes).codes
            out[:, i] = codes.astype(np.float32)
        else:
            out[:, i] = frame[col].to_numpy(dtype=np.float32)
    out[:, -1] = frame[label_col].to_numpy(dtype=np.float32)
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


import intelligence.models.forecast.features as features_module
from intelligence.models.forecast.climatology import build_climatology
from intelligence.models.forecast.features import (
    build_features, downcast_panel, station_cells_only,
)
from intelligence.models.forecast.model import mask_unknown_city, UNKNOWN_CITY
from intelligence.models.forecast.validation import _align_city
from intelligence.models.forecast.native.native_composite import composite_grid_native

_real_composite_grid = features_module.composite_grid
_CHUNK = 2000  # rows of the (n_t, n_q, n_s)-equivalent working set per call
# Both currently unused by run_city_loso_native itself -- kept rather than
# deleted because Task 5's run_final_fit_native appends to this SAME file
# and its brief was not available to check when this comment was written
# (see the plan's progress.md pre-flight conflict scan, "4 -> 5" row).
# Delete if Task 5 lands without needing them.


def _install_native_composite_grid():
    """Monkeypatches features_module's composite_grid reference (build_
    features does `from spatial import composite_grid`, a name binding in
    ITS OWN namespace -- patching spatial.composite_grid directly would not
    reach build_features' already-bound reference). Idempotent: safe to
    call more than once."""
    features_module.composite_grid = composite_grid_native


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
    _install_native_composite_grid()
    fires_by_city = fires_by_city or {}
    cities = sorted(panels_by_city)
    city_codes = cities + [UNKNOWN_CITY]
    per_city = {}

    for held_out in cities:
        train_cities = [c for c in cities if c != held_out]
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

        combined = combine_streamed_units(paths)
        city_col_idx = feature_cols.index("city")
        X, y = combined[:, :-1], combined[:, -1]

        import lightgbm as lgb
        from intelligence.models.forecast.model import PARAMS
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
        test_arr = np.empty((len(scored), len(feature_cols) + 1), dtype=np.float32)
        for i, col in enumerate(feature_cols):
            if col == "city":
                codes = pd.Categorical(scored["city"].astype(str), categories=city_codes).codes
                test_arr[:, i] = codes.astype(np.float32)
            else:
                test_arr[:, i] = scored[col].to_numpy(dtype=np.float32)
        test_arr[:, -1] = scored["y"].to_numpy(dtype=np.float32)

        pred = model.predict(test_arr[:, :-1])
        rmse = float(np.sqrt(np.mean((test_arr[:, -1] - pred) ** 2)))
        per_city[held_out] = {"rmse": round(rmse, 2), "n": len(scored)}

    return {"per_city": per_city}


import os
from concurrent.futures import ThreadPoolExecutor

from intelligence.models.forecast.validation import event_weights
from intelligence.models.forecast.model import PARAMS, QUANTILES


def run_final_fit_native(panels_by_city: dict, horizons: list[int],
                          feature_cols: list[str],
                          fires_by_city: dict | None = None) -> dict:
    """Same return shape as train.py's train_and_promote `final_models`
    dict: one lgb.Booster per quantile in QUANTILES, weighted the same
    way (event_weights, 4x boost on rows with real trailing FIRMS
    activity) as the pandas served model there. Unlike city-LOSO, there
    is no held-out city here -- climatology is built once from ALL
    cities pooled and every city streams into ONE combined training set.

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
    _install_native_composite_grid()
    fires_by_city = fires_by_city or {}
    cities = sorted(panels_by_city)
    city_codes = cities + [UNKNOWN_CITY]

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

    import lightgbm as lgb
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
