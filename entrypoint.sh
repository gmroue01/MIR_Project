#!/bin/sh
set -e

echo "=== Downloading data from HuggingFace... ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download('GabrielMroue/MirProject', local_dir='/app/data', repo_type='dataset')
"

echo "=== Download complete. Dataset structure: ==="
ls -la /app/data/

echo "=== Creating symlinks... ==="
ln -sfn /app/data /app/dataset       && echo "  /app/dataset OK" || echo "  /app/dataset FAILED"
ln -sfn /app/data /app/indexes_faiss && echo "  /app/indexes_faiss OK" || echo "  /app/indexes_faiss FAILED"
ln -sfn /app/data /app/indexes       && echo "  /app/indexes OK" || echo "  /app/indexes FAILED"
mkdir -p /app/Flickr8K
ln -sfn /app/data/Images    /app/Flickr8K/Images    2>/dev/null && echo "  Flickr8K/Images OK"    || echo "  Flickr8K/Images FAILED (maybe no Images subdir)"
ln -sfn /app/data/captions.txt /app/Flickr8K/captions.txt 2>/dev/null && echo "  Flickr8K/captions.txt OK" || echo "  Flickr8K/captions.txt FAILED"

echo "=== Starting Python app... ==="
exec python run.py
