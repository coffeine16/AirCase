"""Per-fold checkpointing for train_and_promote's three CV stages (spec
5.1-5.3): walk-forward, spatial-LOSO, city-LOSO. Each stage is dozens to
low-hundreds of independent folds and is the dominant cost of a real
training run (hours) -- losing all of it to a killed job (HF credit
exhausted, preemption) means re-paying for compute already done.

This module only handles the LOCAL half: save/load one fold's result to/
from disk, keyed by (stage, fold_key). Local disk alone does NOT survive a
killed-and-relaunched HF Job -- a Job's filesystem is deleted the moment it
exits (see training/train_and_upload.py's own docstring) -- making that
durable is train_and_upload.py's job, via periodic Hub sync of the same
directory these functions write to. `checkpoint_dir=None` (the default
everywhere this is threaded through) disables checkpointing entirely --
every existing caller (tests, ad-hoc local runs) is unaffected.

Pickle, not JSON: a walk-forward fold's result carries a pandas DataFrame
(the OOF frame) plus numpy arrays, neither JSON-serialisable without a
custom encoder for every fold shape across all three stages. Accepted
here specifically because every checkpoint is FIRST-PARTY -- written by
this same training process, read back only by a later invocation of the
same job's own resume path, round-tripped through a PRIVATE Hub repo this
job already authenticates to with its own token (see train_and_upload.py's
create_repo(..., private=True)). Not a general-purpose deserialiser for
untrusted input; do not point this at anything else."""
import pickle
from pathlib import Path
from typing import Any


def _fold_path(checkpoint_dir: str, stage: str, fold_key: str) -> Path:
    # fold_key can carry characters unsafe in a filename (a walk-forward
    # fold's key is an ISO timestamp, which contains ":") -- keep the
    # escaping in ONE place so save/load can never disagree on it.
    safe_key = str(fold_key).replace(":", "-").replace("/", "_").replace(" ", "_")
    return Path(checkpoint_dir) / stage / f"{safe_key}.pkl"


def load_fold(checkpoint_dir: str | None, stage: str, fold_key: str) -> tuple[bool, Any]:
    """Returns (found, result). found=False means "never attempted, compute
    it now"; found=True means "already attempted, before this process even
    started" and `result` is whatever that attempt returned -- including
    None for a fold that was legitimately empty (e.g. no station in that
    cell's window). None must be cached too: without a found=True mask, a
    resumed run would retry an always-empty fold forever, unable to tell
    "not yet tried" apart from "tried, found nothing"."""
    if checkpoint_dir is None:
        return False, None
    path = _fold_path(checkpoint_dir, stage, fold_key)
    if not path.exists():
        return False, None
    with open(path, "rb") as f:
        return True, pickle.load(f)["result"]


def save_fold(checkpoint_dir: str | None, stage: str, fold_key: str, result: Any) -> None:
    """Atomic write (temp file + rename) -- a checkpoint file is never
    observed half-written by the periodic Hub-sync thread reading the same
    directory concurrently, or left corrupt by a kill landing mid-write,
    which is exactly the scenario this module exists to defend against."""
    if checkpoint_dir is None:
        return
    path = _fold_path(checkpoint_dir, stage, fold_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"result": result}, f)
    tmp.replace(path)
