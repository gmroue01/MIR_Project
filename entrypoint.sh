#!/bin/sh
set -e

export HF_HOME=/app/data
export TRANSFORMERS_CACHE=/app/data

DATA_DIR="/app/data"
SENTINEL="$DATA_DIR/.download_complete"

if [ ! -f "$SENTINEL" ]; then
    echo "First boot — downloading data from HuggingFace..."
    for i in 1 2 3 4 5; do
        echo "Attempt $i..."
        timeout 600 python -c "
import os
os.environ['HF_HOME'] = '/app/data'
from huggingface_hub import snapshot_download
snapshot_download(
    'GabrielMroue/MirProject',
    local_dir='/app/data',
    repo_type='dataset',
    token='$HF_TOKEN',
    max_workers=2
)
" && touch "$SENTINEL" && break || echo "Attempt $i failed, retrying..."
        sleep 10
    done
else
    echo "Data already present at /app/data, skipping download."
fi

echo "Creating symlinks..."
# Le code cherche /app/Flickr8K, /app/indexes_faiss, /app/dataset
# Les données sont dans /app/data/ (volume Railway)
ln -sfn /app/data/Flickr8k    /app/Flickr8K    2>/dev/null || true
ln -sfn /app/data/indexes_faiss /app/indexes_faiss 2>/dev/null || true
ln -sfn /app/data/indexes      /app/indexes      2>/dev/null || true
ln -sfn /app/data/dataset      /app/dataset      2>/dev/null || true

echo "Starting app..."
exec python run.py