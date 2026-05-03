import axios from "axios";

const BASE = "/api";

export const getImages = (page = 1, pageSize = 50, classFilter = "") =>
  axios.get(`${BASE}/images`, {
    params: { page, page_size: pageSize, ...(classFilter && { class_filter: classFilter }) },
  }).then((r) => r.data);

export const getClasses = () =>
  axios.get(`${BASE}/classes`).then((r) => r.data);

export const search = (queryIndex, descriptors, measure, topK) =>
  axios.post(`${BASE}/search`, {
    query_index: queryIndex,
    descriptors,
    measure,
    top_k: topK,
  }).then((r) => r.data);

export const getIndexingMetrics = () =>
  axios.get(`${BASE}/indexing-metrics`).then((r) => r.data);

export const computeMap = (descriptors, measure, topK, maxQueries = 46) =>
  axios.post(`${BASE}/map`, {
    descriptors,
    measure,
    top_k: topK,
    max_queries: maxQueries,
  }).then((r) => r.data);

// ── CLIP / Flickr8K ────────────────────────────────────────────────────────

export const getClipImages = (page = 1, pageSize = 30) =>
  axios.get(`${BASE}/clip/images`, { params: { page, page_size: pageSize } }).then((r) => r.data);

export const clipTextToImage = (query, topK = 10) =>
  axios.post(`${BASE}/clip/text-to-image`, { query, top_k: topK }).then((r) => r.data);

export const clipImageToText = (imageIdx, topK = 10) =>
  axios.post(`${BASE}/clip/image-to-text`, { image_idx: imageIdx, top_k: topK }).then((r) => r.data);

export const clipEvaluate = (imageIndices, textQueries, topK = 10) =>
  axios.post(`${BASE}/clip/evaluate`, {
    image_indices: imageIndices,
    text_queries: textQueries,
    top_k: topK,
  }).then((r) => r.data);
