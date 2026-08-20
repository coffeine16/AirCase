"""Entrypoint for the HF Jobs training container.

A Job's filesystem is deleted the moment it exits — there is no persistent
disk. So this script does three things in order: (1) seed the PRIOR model
(if one was already committed to the repo's models/ folder and baked into
this image) so the promotion gate has something real to compare against,
not just "no prior, always promote"; (2) run training; (3) push whichever
model is now actually SERVED (the freshly promoted one, or the still-active
prior if this run was refused) to a Hugging Face Hub model repo, since that
Hub repo is the only durable thing this Job can write to. The GH Actions
workflow that launched this Job then downloads that Hub repo's contents
back into models/ and commits them to git — this script does not touch git
at all, it only knows about the Hub.
"""
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

MODELS_DIR = Path("models")                        # baked into the image from the repo
OUT_DIR = Path("data/outputs/_forecast_models")     # what intelligence.models.forecast.run() reads/writes

# EDIT THIS to your own Hugging Face username/org before the first real run.
# The repo is created automatically (private) on first use if it doesn't exist.
HF_MODEL_REPO = "MagmaCubes1133/aircase-forecast-models"

DEFAULT_CITIES = ["bengaluru", "delhi", "chennai", "mumbai", "kolkata", "hyderabad", "pune", "ahmedabad"]

# The GH Actions workflow passes this through via `hf jobs run --env
# CITIES=...` when its `cities` input is non-blank; unset/blank means all 8.
CITIES = [c.strip() for c in os.environ["CITIES"].split(",")] if os.environ.get("CITIES") else DEFAULT_CITIES

# Checkpointing (per-fold resume if this Job dies mid-run, e.g. HF credit
# runs out and gets recharged just enough to relaunch). A Job's filesystem
# is deleted the moment it exits (this module's own docstring above), so
# CHECKPOINT_DIR alone does not survive a kill -- durability comes from
# periodically syncing it to HF_MODEL_REPO's "checkpoints/" prefix (see
# upload_checkpoints/start_checkpoint_sync below) and pulling it back down
# on the next launch (download_checkpoints).
#
# Namespaced by GIT_SHA + the exact city list, NOT just a fixed path: a
# checkpoint is only safe to resume if it came from the byte-identical code
# and city selection that's running now -- reusing a stale checkpoint from
# a DIFFERENT commit (different feature logic, different fold geometry)
# would silently corrupt the run it "resumes". GIT_SHA defaults to
# "unknown" if the launch workflow didn't pass it (e.g. an ad-hoc `hf jobs
# run` invocation) -- checkpointing still works within that one run, it
# just won't distinguish it from another "unknown"-tagged run.
GIT_SHA = os.environ.get("GIT_SHA", "unknown")
RUN_FINGERPRINT = f"{GIT_SHA}-{'-'.join(sorted(CITIES))}"
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_HUB_PREFIX = f"checkpoints/{RUN_FINGERPRINT}"
CHECKPOINT_SYNC_INTERVAL_S = 600   # 10 minutes -- "periodic", not per-fold:
# a Hub upload is a real network round-trip and 82+19+8=109 folds at maybe
# a few minutes each would otherwise mean well over a hundred API calls
# over a multi-hour run for no real durability gain between them.


def download_checkpoints() -> None:
    """Pull any checkpoints a PRIOR, killed attempt at this same run (same
    RUN_FINGERPRINT) already pushed to the Hub -- the actual resume path.
    No matching prefix (first attempt, or a genuinely different run) is not
    an error, just nothing to resume."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=HF_MODEL_REPO, allow_patterns=f"{CHECKPOINT_HUB_PREFIX}/**",
                           local_dir=".")
    except Exception as e:   # noqa: BLE001 -- no checkpoints to resume is not fatal
        print(f"[train_and_upload] no checkpoints to resume ({type(e).__name__}: {e})")
        return
    src = Path(CHECKPOINT_HUB_PREFIX)
    if not src.exists():
        print("[train_and_upload] no prior checkpoints found for this run — starting fresh")
        return
    n = sum(1 for _ in src.rglob("*.pkl"))
    if n == 0:
        print("[train_and_upload] no prior checkpoints found for this run — starting fresh")
        return
    # Downloaded under the fingerprinted Hub path; build_features'/
    # validation.py's own checkpoint_dir contract just wants ONE flat
    # local directory (see checkpoint.py) -- move it into place rather
    # than thread the fingerprint through every fold-runner call.
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)
    shutil.move(str(src), str(CHECKPOINT_DIR))
    print(f"[train_and_upload] resumed {n} fold checkpoint(s) from a prior attempt at this run")


def upload_checkpoints() -> None:
    """Best-effort sync of the local checkpoint directory to the Hub. Never
    raises into the caller -- a failed sync should not take down a training
    run that's otherwise progressing fine; it just means less is saved if
    THIS particular sync's window is also when the Job gets killed."""
    if not CHECKPOINT_DIR.exists() or not any(CHECKPOINT_DIR.rglob("*.pkl")):
        return
    try:
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=str(CHECKPOINT_DIR), path_in_repo=CHECKPOINT_HUB_PREFIX,
                               repo_id=HF_MODEL_REPO, commit_message="checkpoint sync")
    except Exception as e:   # noqa: BLE001 -- best-effort; retried next interval
        print(f"[train_and_upload] checkpoint sync failed ({type(e).__name__}: {e}) -- will retry")


def start_checkpoint_sync() -> tuple[threading.Thread, threading.Event]:
    """Background thread, periodic upload of whatever's checkpointed
    locally so far. Runs CONCURRENTLY with the blocking `run()` call below
    -- without this, nothing durable exists until training finishes, which
    defeats the entire point of checkpointing something that can take
    hours. Daemon thread + a stop Event the caller sets in a `finally`, so
    a normal exit (or an exception inside `run()`) still gets one final
    sync of whatever's left, not just whatever happened to land on the
    last periodic tick."""
    stop_event = threading.Event()

    def _loop():
        while not stop_event.wait(CHECKPOINT_SYNC_INTERVAL_S):
            upload_checkpoints()
        upload_checkpoints()   # final sync on stop

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread, stop_event


def clear_checkpoints() -> None:
    """Delete this run's checkpoints from the Hub once it completes
    normally (promoted or refused -- either way the run finished, there is
    nothing left to resume). Otherwise a LATER, unrelated run that happens
    to share this exact RUN_FINGERPRINT (same commit re-launched on
    purpose, e.g. a deliberate re-run) would download and skip folds
    that were real for THIS run's attempt but should be recomputed fresh
    for that one. Best-effort: a failed delete just means the next launch
    downloads and resumes from a run that already finished, which is
    wasteful (it will re-verify already-good folds) but not incorrect,
    since the fingerprint still requires byte-identical code+cities."""
    try:
        from huggingface_hub import HfApi
        HfApi().delete_folder(path_in_repo=CHECKPOINT_HUB_PREFIX, repo_id=HF_MODEL_REPO)
        print("[train_and_upload] cleared this run's checkpoints (completed normally)")
    except Exception as e:   # noqa: BLE001 -- best-effort cleanup only
        print(f"[train_and_upload] couldn't clear checkpoints ({type(e).__name__}: {e}) -- harmless")


def seed_prior_model():
    """Copy models/ (whatever the repo last committed, baked into this image
    at build time) into data/outputs/_forecast_models/ so train_and_promote
    sees it as `prior_manifest` and actually gates against it. First-ever
    run: models/ has nothing but .gitkeep, so there's no prior — expected,
    not an error."""
    manifest_path = MODELS_DIR / "manifest.json"
    if not manifest_path.exists():
        print("[train_and_upload] no prior model in models/ — first run, or "
              "every prior run was refused before ever promoting anything")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(manifest_path, OUT_DIR / "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    version_dir = MODELS_DIR / manifest["version"]
    if version_dir.exists():
        shutil.copytree(version_dir, OUT_DIR / manifest["version"], dirs_exist_ok=True)
    print(f"[train_and_upload] seeded prior model {manifest['version']} "
          f"(spatial-LOSO RMSE {manifest['eval'].get('spatial_loso_rmse')})")


def served_manifest(run_result: dict) -> dict | None:
    """Which manifest is actually served after this run — the just-promoted
    one, or (if refused) whatever was seeded as the prior above. Mirrors the
    exact same logic intelligence.models.forecast.run() uses internally to
    decide what forecast.json gets built from."""
    if run_result["promoted"]:
        return run_result
    manifest_path = OUT_DIR / "manifest.json"
    # train_and_promote only WRITES this file inside its `if promoted:`
    # branch, so when refused it still holds whatever seed_prior_model()
    # copied there — i.e. exactly the model still being served.
    return json.loads(manifest_path.read_text()) if manifest_path.exists() else None


def main():
    from huggingface_hub import HfApi
    api = HfApi()
    # Created HERE, not just before the final upload below -- the periodic
    # checkpoint sync needs the repo to exist from the very start of
    # training, not only once training finishes. exist_ok=True makes this
    # safe to call again later too (unnecessary now, left the single call).
    api.create_repo(HF_MODEL_REPO, repo_type="model", exist_ok=True, private=True)

    seed_prior_model()
    download_checkpoints()

    # Background periodic sync runs CONCURRENTLY with the blocking run()
    # call below -- see start_checkpoint_sync's own docstring for why. The
    # `finally` guarantees one last sync attempt whether run() returns
    # normally or raises (e.g. this Job gets SIGTERM'd for running out of
    # HF credit mid-training) -- that last sync is the whole point.
    sync_thread, sync_stop = start_checkpoint_sync()
    try:
        from intelligence.models.forecast import run
        result = run(cities=CITIES, checkpoint_dir=str(CHECKPOINT_DIR))
    finally:
        sync_stop.set()
        sync_thread.join(timeout=120)

    # run() returned without raising -- every fold in every CV stage ran to
    # completion (whether or not the result was PROMOTED), so there is
    # nothing left for a future launch to resume regardless of what happens
    # below.
    clear_checkpoints()

    print("=== FINAL MANIFEST ===")
    print(json.dumps(result, indent=2))

    served = served_manifest(result)
    if served is None:
        print("[train_and_upload] no usable model exists after this run — "
              "first-ever run AND it was refused (see eval numbers above for "
              "why). Nothing to upload.", file=sys.stderr)
        sys.exit(1)

    version = served["version"]
    print(f"[train_and_upload] served version: {version} "
          f"(promoted THIS run: {result['promoted']})")

    api.upload_file(path_or_fileobj=str(OUT_DIR / "manifest.json"),
                     path_in_repo="manifest.json", repo_id=HF_MODEL_REPO)
    api.upload_folder(folder_path=str(OUT_DIR / version),
                       path_in_repo=version, repo_id=HF_MODEL_REPO)
    print(f"[train_and_upload] uploaded to https://huggingface.co/{HF_MODEL_REPO}")


if __name__ == "__main__":
    main()
