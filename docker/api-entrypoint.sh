#!/bin/sh
set -e
# The dataset is deterministic from configs/ecosystem.yaml, so a missing one is regenerated
# rather than shipped in the image.
if [ ! -f data/generated/default/manifest.json ]; then
    echo "No dataset found; generating it (one-off)..."
    python scripts/generate_data.py
fi
exec uvicorn backend.api.app:app --host 0.0.0.0 --port 8000
