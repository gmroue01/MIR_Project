import os
import time
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

from app.searcher import search, get_indexing_metrics, get_all_images, _load_meta
from app.evaluation.metrics import compute_all
from app.similarity.measures import MEASURES
from app.config import DATASET_DIR, STATIC_DIR

app = FastAPI(title="MIR - Image Search Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve dataset images
app.mount("/images", StaticFiles(directory=DATASET_DIR), name="images")

# Serve React build (production)
if os.path.isdir(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")


# ---------- Schemas ----------

class SearchRequest(BaseModel):
    query_index: int
    descriptors: List[str]
    measure: str
    top_k: int = 50


# ---------- Routes ----------

@app.get("/api/images")
def list_images(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    class_filter: Optional[str] = None,
):
    images = get_all_images()
    if class_filter:
        images = [img for img in images if class_filter.lower() in img["class"].lower()]
    total = len(images)
    start = (page - 1) * page_size
    end = start + page_size
    return {"total": total, "page": page, "page_size": page_size, "images": images[start:end]}


@app.get("/api/classes")
def list_classes():
    _, classes = _load_meta()
    unique = sorted(set(classes.tolist()))
    counts = {c: int(np.sum(classes == c)) for c in unique}
    return {"classes": unique, "counts": counts}


@app.get("/api/descriptors")
def list_descriptors():
    return {"descriptors": ["color_histogram", "hog", "mobilenetv2", "resnet50", "vit_base", "dinov2", "sift", "orb"]}


@app.get("/api/measures")
def list_measures():
    return {"measures": list(MEASURES.keys()), "hamming_only_orb": True}


@app.post("/api/search")
def search_endpoint(req: SearchRequest):
    valid_descriptors = {"color_histogram", "hog", "mobilenetv2", "resnet50", "vit_base", "dinov2", "sift", "orb"}
    for d in req.descriptors:
        if d not in valid_descriptors:
            raise HTTPException(400, f"Unknown descriptor: {d}")
    if req.measure not in MEASURES:
        raise HTTPException(400, f"Unknown measure: {req.measure}")
    if req.measure == "hamming" and not (len(req.descriptors) == 1 and req.descriptors[0] == "orb"):
        raise HTTPException(400, "Hamming distance is only available with ORB as sole descriptor.")
    if req.top_k not in (20, 50):
        raise HTTPException(400, "top_k must be 20 or 50.")

    filenames, classes = _load_meta()
    if req.query_index < 0 or req.query_index >= len(filenames):
        raise HTTPException(400, "Invalid query_index.")

    query_class = str(classes[req.query_index])
    total_relevant = int(np.sum(classes == query_class)) - 1

    t0 = time.perf_counter()
    results_top50 = search(req.query_index, req.descriptors, req.measure, top_k=50)
    search_time = time.perf_counter() - t0

    results_top20 = results_top50[:20]

    metrics_20 = compute_all(results_top20, query_class, total_relevant, k_values=[20])
    metrics_50 = compute_all(results_top50, query_class, total_relevant, k_values=[50])

    return {
        "query_filename": str(filenames[req.query_index]),
        "query_class": query_class,
        "results_top20": results_top20,
        "results_top50": results_top50,
        "metrics": {
            "top20": {
                "precision": metrics_20.get("precision@20", 0),
                "recall": metrics_20.get("recall@20", 0),
                "average_precision": metrics_20["average_precision"],
                "r_precision": metrics_20["r_precision"],
            },
            "top50": {
                "precision": metrics_50.get("precision@50", 0),
                "recall": metrics_50.get("recall@50", 0),
                "average_precision": metrics_50["average_precision"],
                "r_precision": metrics_50["r_precision"],
            },
            "total_relevant_in_db": total_relevant,
        },
        "search_time_s": round(search_time, 4),
    }


class MapRequest(BaseModel):
    descriptors: List[str]
    measure: str
    top_k: int = 50
    max_queries: int = 100  # cap to keep response time reasonable


@app.post("/api/map")
def compute_map(req: MapRequest):
    """
    Compute Mean Average Precision over a sample of queries (one per class).
    Returns per-class AP and overall MAP.
    """
    valid_descriptors = {"color_histogram", "hog", "mobilenetv2", "resnet50", "vit_base", "dinov2", "sift", "orb"}
    for d in req.descriptors:
        if d not in valid_descriptors:
            raise HTTPException(400, f"Unknown descriptor: {d}")
    if req.measure not in MEASURES:
        raise HTTPException(400, f"Unknown measure: {req.measure}")
    if req.top_k not in (20, 50):
        raise HTTPException(400, "top_k must be 20 or 50.")

    filenames, classes = _load_meta()
    unique_classes = sorted(set(classes.tolist()))

    # Pick one representative query per class (first image)
    class_to_idx = {}
    for i, c in enumerate(classes.tolist()):
        if c not in class_to_idx:
            class_to_idx[c] = i

    query_indices = [class_to_idx[c] for c in unique_classes]
    if req.max_queries and len(query_indices) > req.max_queries:
        query_indices = query_indices[:req.max_queries]
        unique_classes = unique_classes[:req.max_queries]

    t0 = time.perf_counter()
    ap_per_class = {}

    for qidx, qclass in zip(query_indices, unique_classes):
        total_relevant = int(np.sum(classes == qclass)) - 1
        results = search(qidx, req.descriptors, req.measure, top_k=req.top_k)
        m = compute_all(results, qclass, total_relevant, k_values=[req.top_k])
        ap_per_class[qclass] = round(m["average_precision"], 4)

    map_score = float(np.mean(list(ap_per_class.values())))
    elapsed = time.perf_counter() - t0

    return {
        "map": round(map_score, 4),
        "num_queries": len(query_indices),
        "ap_per_class": ap_per_class,
        "top_k": req.top_k,
        "elapsed_s": round(elapsed, 2),
    }


@app.get("/api/indexing-metrics")
def indexing_metrics():
    m = get_indexing_metrics()
    if not m:
        raise HTTPException(503, "Indexes not built yet. Run the indexer first.")
    return m


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Catch-all: serve React index.html for client-side routing (production)
@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"detail": "Frontend not built. Run: cd frontend && npm run build"}
