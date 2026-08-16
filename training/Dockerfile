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

COPY training/requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY shared/__init__.py shared/config.py shared/grid.py shared/
COPY intelligence/__init__.py intelligence/__init__.py
COPY intelligence/models/__init__.py intelligence/models/__init__.py
COPY intelligence/models/forecast/ intelligence/models/forecast/
COPY data/historical/ data/historical/
COPY training/train_and_upload.py .

CMD ["python", "train_and_upload.py"]
