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
