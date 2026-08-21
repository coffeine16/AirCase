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


def _install_native_composite_grid():
    """Monkeypatches features_module's composite_grid reference (build_
    features does `from spatial import composite_grid`, a name binding in
    ITS OWN namespace -- patching spatial.composite_grid directly would not
    reach build_features' already-bound reference). Idempotent: safe to
    call more than once."""
    features_module.composite_grid = composite_grid_native


def run_city_loso_native(panels_by_city: dict, horizons: list[int],
                          feature_cols: list[str],
                          fires_by_city: dict | None = None) -> dict:
    """Same return shape as validation.run_city_loso:
    {"per_city": {city: {"rmse": float, "n": int}}}. Streams each
    training city's features to disk one at a time instead of pooling all
    N-1 cities into one pandas frame -- see this module's own docstring
    for why."""
    _install_native_composite_grid()
    fires_by_city = fires_by_city or {}
    cities = sorted(panels_by_city)
    city_codes = cities + [UNKNOWN_CITY]
    per_city = {}

    for held_out in cities:
        train_cities = [c for c in cities if c != held_out]
        clim_tables = build_climatology(downcast_panel(pd.concat(
            [panels_by_city[c] for c in train_cities], ignore_index=True)))

        work_dir = Path("scratch_out") / "native_city_loso" / held_out
        paths = []
        for i, city in enumerate(train_cities):
            frame = mask_unknown_city(
                build_features(panels_by_city[city], horizons,
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
        ds = lgb.Dataset(X, label=y, categorical_feature=[city_col_idx])
        model = lgb.train({**PARAMS, "alpha": 0.5}, ds, num_boost_round=200)
        del combined, X, y, ds
        gc.collect()

        test_frame = build_features(panels_by_city[held_out], horizons,
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
