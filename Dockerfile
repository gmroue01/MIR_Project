# ── Stage 1 : build du frontend React ────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build
# Vite outDir = ../app/static  →  résultat dans /app/static

# ── Stage 2 : runtime Python ──────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Dépendances système pour OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# torch CPU (beaucoup plus léger que la version CUDA par défaut)
RUN pip install --no-cache-dir \
        torch==2.6.0+cpu \
        torchvision==0.21.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Reste des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code applicatif
COPY app/ ./app/
COPY run.py .

# Frontend buildé (copié depuis le stage 1)
COPY --from=frontend-builder /app/static ./app/static/

# Les gros dossiers de données sont montés en volume (voir docker-compose.yml) :
#   /app/dataset        — dataset voitures
#   /app/indexes_faiss  — index FAISS (CLIP + futur ALIGN)
#   /app/Flickr8K       — images Flickr8K

EXPOSE 8000

CMD ["python", "run.py"]
