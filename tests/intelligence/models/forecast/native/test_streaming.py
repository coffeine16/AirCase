from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from intelligence.models.forecast.native.streaming import (
    stream_unit_to_disk, combine_streamed_units,
)

FEATURE_COLS = ["a", "b", "city"]


def _fake_frame(n, city_name):
    return pd.DataFrame({
        "a": np.arange(n, dtype=float),
        "b": np.arange(n, dtype=float) * 2.0,
        "city": [city_name] * n,
        "y": np.arange(n, dtype=float) * 10.0,
    })


def test_stream_unit_to_disk_writes_expected_row_count(tmp_path):
    frame = _fake_frame(50, "delhi")
    path = tmp_path / "delhi.npy"
    n = stream_unit_to_disk(frame, path, FEATURE_COLS, city_codes=["delhi", "mumbai", "unknown"])
    assert n == 50
    assert path.exists()
    arr = np.load(path)
    assert arr.shape == (50, len(FEATURE_COLS) + 1)  # features + label


def test_stream_unit_to_disk_encodes_city_as_its_fixed_code(tmp_path):
    frame = _fake_frame(5, "mumbai")
    path = tmp_path / "mumbai.npy"
    codes = ["delhi", "mumbai", "unknown"]
    stream_unit_to_disk(frame, path, FEATURE_COLS, city_codes=codes)
    arr = np.load(path)
    city_col_idx = FEATURE_COLS.index("city")
    assert (arr[:, city_col_idx] == 1.0).all()  # "mumbai" is codes[1]


def test_combine_streamed_units_matches_manual_concat(tmp_path):
    codes = ["delhi", "mumbai", "unknown"]
    paths = []
    for i, city in enumerate(["delhi", "mumbai"]):
        frame = _fake_frame(10 + i, city)
        path = tmp_path / f"{city}.npy"
        stream_unit_to_disk(frame, path, FEATURE_COLS, city_codes=codes)
        paths.append(path)

    combined = combine_streamed_units(paths)
    assert combined.shape[0] == 10 + 11
    assert combined.shape[1] == len(FEATURE_COLS) + 1
    # input files are deleted after combining
    for p in paths:
        assert not p.exists()
