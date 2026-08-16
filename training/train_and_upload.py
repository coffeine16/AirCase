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
    seed_prior_model()

    from intelligence.models.forecast import run
    result = run(cities=CITIES)
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

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(HF_MODEL_REPO, repo_type="model", exist_ok=True, private=True)
    api.upload_file(path_or_fileobj=str(OUT_DIR / "manifest.json"),
                     path_in_repo="manifest.json", repo_id=HF_MODEL_REPO)
    api.upload_folder(folder_path=str(OUT_DIR / version),
                       path_in_repo=version, repo_id=HF_MODEL_REPO)
    print(f"[train_and_upload] uploaded to https://huggingface.co/{HF_MODEL_REPO}")


if __name__ == "__main__":
    main()
