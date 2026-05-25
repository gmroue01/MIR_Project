#!/bin/sh
set -e

echo "Downloading data from HuggingFace..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('GabrielMroue/MirProject', local_dir='/app', repo_type='dataset')
"

echo "Data ready, starting app..."
exec python run.py