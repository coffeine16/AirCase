# Training image for HF Jobs — built and pushed by
# .github/workflows/train-forecast.yml, run via `hf jobs run`.
#
# Build context is the REPO ROOT (not this directory), so this only pulls
# in what intelligence/models/forecast/ actually depends on: shared/config.py
# + shared/grid.py (not the whole shared/ surface), the forecast package
# itself, and the pre-built historical panels. Not the full repo — no app/,
# no ingestion collectors, no .git history, nothing that would bloat the
# image or drag in dependencies training never touches.
FROM python:3.11-slim

WORKDIR /app

# LightGBM's compiled lib_lightgbm.so is linked against libgomp (OpenMP);
# python:3.11-slim doesn't ship it, so importing lightgbm fails with
# "OSError: libgomp.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY training/requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY shared/__init__.py shared/config.py shared/grid.py shared/
COPY intelligence/__init__.py intelligence/__init__.py
COPY intelligence/models/__init__.py intelligence/models/__init__.py
COPY intelligence/models/forecast/ intelligence/models/forecast/
COPY data/historical/ data/historical/
COPY training/train_and_upload.py .

CMD ["python", "train_and_upload.py"]
