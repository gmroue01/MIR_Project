#!/bin/sh
set -e

echo "Downloading data from HuggingFace..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('GabrielMroue/MirProject', local_dir='/app/data', repo_type='dataset')
"

echo "Creating symlinks..."
ln -sfn /app/data /app/dataset
ln -sfn /app/data /app/indexes_faiss
ln -sfn /app/data /app/indexes
mkdir -p /app/Flickr8K
ln -sfn /app/data/Images /app/Flickr8K/Images
ln -sfn /app/data/captions.txt /app/Flickr8K/captions.txt

echo "Starting app..."
exec python run.py
