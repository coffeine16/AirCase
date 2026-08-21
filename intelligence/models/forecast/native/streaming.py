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
