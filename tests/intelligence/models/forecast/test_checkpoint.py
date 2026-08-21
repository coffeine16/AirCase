import numpy as np
import pandas as pd

from intelligence.models.forecast.checkpoint import load_fold, save_fold


def test_load_fold_not_found_when_never_saved(tmp_path):
    found, result = load_fold(str(tmp_path), "spatial_loso", "cell_A")
    assert found is False
    assert result is None


def test_save_then_load_round_trips_a_result(tmp_path):
    payload = {"held_out": "cell_A", "rmse": 12.5, "n": 40}
    save_fold(str(tmp_path), "spatial_loso", "cell_A", payload)
    found, result = load_fold(str(tmp_path), "spatial_loso", "cell_A")
    assert found is True
    assert result == payload


def test_save_then_load_distinguishes_cached_none_from_never_attempted(tmp_path):
    # A legitimately empty fold (e.g. no valid rows) returns None -- that
    # must be cached too, or a resumed run retries it forever, unable to
    # tell "not yet tried" apart from "tried, found nothing".
    save_fold(str(tmp_path), "walk_forward", "2024-06-01T00:00:00+00:00", None)
    found, result = load_fold(str(tmp_path), "walk_forward", "2024-06-01T00:00:00+00:00")
    assert found is True
    assert result is None


def test_checkpoint_dir_none_disables_checkpointing(tmp_path):
    # Every real call site defaults to checkpoint_dir=None -- must be a
    # true no-op, not "look in the current directory".
    save_fold(None, "spatial_loso", "cell_A", {"rmse": 1.0})
    found, result = load_fold(None, "spatial_loso", "cell_A")
    assert found is False
    assert result is None


def test_fold_key_with_colons_is_filesystem_safe(tmp_path):
    # A walk-forward fold's key is an ISO timestamp ("2024-06-01T00:00:00+00:00"),
    # which contains characters Windows filenames reject outright.
    key = "2024-06-01T00:00:00+00:00"
    save_fold(str(tmp_path), "walk_forward", key, {"skill": 0.1})
    found, result = load_fold(str(tmp_path), "walk_forward", key)
    assert found is True
    assert result == {"skill": 0.1}


def test_handles_a_dataframe_payload_like_a_real_walk_forward_fold(tmp_path):
    # Real walk-forward fold results carry a pandas DataFrame (the OOF
    # frame) -- the reason this module uses pickle, not JSON.
    payload = {"skill": 0.12, "oof": pd.DataFrame({"y": [1.0, 2.0], "p50": [1.1, 2.2]}),
               "n_train": 500}
    save_fold(str(tmp_path), "walk_forward", "fold-1", payload)
    found, result = load_fold(str(tmp_path), "walk_forward", "fold-1")
    assert found is True
    assert result["skill"] == 0.12
    pd.testing.assert_frame_equal(result["oof"], payload["oof"])
