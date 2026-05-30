import React, { useState, useEffect } from "react";
import { getIndexingMetrics } from "../api";
import styles from "./BenchmarkPage.module.css";

const LABELS = {
  color_histogram: "Histo. Couleur",
  mobilenetv2:    "MobileNetV2",
  resnet50:       "ResNet50",
  vit_base:       "ViT Base",
  dinov2:         "DinoV2",
  sift:           "SIFT",
};

const COLORS = {
  color_histogram: "#8b5cf6",
  mobilenetv2:    "#f59e0b",
  resnet50:       "#34d399",
  vit_base:       "#f472b6",
  dinov2:         "#60a5fa",
  sift:           "#a78bfa",
};

function Bar({ value, max, color }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className={styles.bar}>
      <div className={styles.barFill} style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

export default function BenchmarkPage() {
  const [metrics, setMetrics] = useState(null);
  const [error,   setError]   = useState("");

  useEffect(() => {
    getIndexingMetrics()
      .then(setMetrics)
      .catch(() => setError("Index non disponible — lancez d'abord l'indexeur."));
  }, []);

  if (error) return (
    <div className={styles.page}>
      <div className={styles.errorBox}>{error}</div>
    </div>
  );

  if (!metrics) return (
    <div className={styles.page}>
      <div className={styles.loading}>
        <div className={styles.orbit}><div className={styles.orbitDot} /></div>
        <span>Chargement des métriques…</span>
      </div>
    </div>
  );

  const entries   = Object.entries(metrics);
  const numImages = entries[0]?.[1]?.num_images ?? "?";
  const maxTime   = Math.max(...entries.map(([, v]) => v.indexing_time_s));
  const maxSize   = Math.max(...entries.map(([, v]) => v.descriptor_size_mb));
  const maxAvg    = Math.max(...entries.map(([, v]) => v.avg_search_time_s));
  const maxDim    = Math.max(...entries.map(([, v]) => v.descriptor_dim));

  return (
    <div className={styles.page}>
      <div className={styles.pageHead}>
        <h1 className={styles.pageTitle}><span className={styles.star}>✦</span> Benchmark des descripteurs</h1>
        <p className={styles.pageSub}>Métriques mesurées sur {numImages} images</p>
      </div>

      {/* Cards grid */}
      <div className={styles.cards}>
        {entries.map(([key, m]) => {
          const color = COLORS[key] || "#8b5cf6";
          return (
            <div key={key} className={styles.card} style={{ "--c": color }}>
              <div className={styles.cardHead}>
                <span className={styles.cardDot} />
                <span className={styles.cardName}>{LABELS[key] || key}</span>
                <span className={styles.cardDim}>{m.descriptor_dim}D</span>
              </div>

              <div className={styles.stats}>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Indexation</span>
                  <span className={styles.statVal} style={{ color }}>{m.indexing_time_s.toFixed(1)} s</span>
                  <Bar value={m.indexing_time_s} max={maxTime} color={color} />
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Taille index</span>
                  <span className={styles.statVal} style={{ color }}>{m.descriptor_size_mb.toFixed(2)} MB</span>
                  <Bar value={m.descriptor_size_mb} max={maxSize} color={color} />
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Temps moy. / img</span>
                  <span className={styles.statVal} style={{ color }}>{(m.avg_search_time_s * 1000).toFixed(1)} ms</span>
                  <Bar value={m.avg_search_time_s} max={maxAvg} color={color} />
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Dimension</span>
                  <span className={styles.statVal} style={{ color }}>{m.descriptor_dim}</span>
                  <Bar value={m.descriptor_dim} max={maxDim} color={color} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Comparison table */}
      <div className={styles.tableSection}>
        <h2 className={styles.sectionTitle}>Vue comparative</h2>
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Descripteur</th>
                <th>Dimension</th>
                <th>Indexation</th>
                <th className={styles.barCol}></th>
                <th>Taille (MB)</th>
                <th className={styles.barCol}></th>
                <th>Tps / img</th>
                <th className={styles.barCol}></th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([key, m]) => {
                const color = COLORS[key] || "#8b5cf6";
                return (
                  <tr key={key}>
                    <td>
                      <span className={styles.tdDot} style={{ background: color }} />
                      {LABELS[key] || key}
                    </td>
                    <td className={styles.num}>{m.descriptor_dim}</td>
                    <td className={styles.num}>{m.indexing_time_s.toFixed(1)} s</td>
                    <td className={styles.barCell}><Bar value={m.indexing_time_s} max={maxTime} color={color} /></td>
                    <td className={styles.num}>{m.descriptor_size_mb.toFixed(2)}</td>
                    <td className={styles.barCell}><Bar value={m.descriptor_size_mb} max={maxSize} color={color} /></td>
                    <td className={styles.num}>{(m.avg_search_time_s * 1000).toFixed(1)} ms</td>
                    <td className={styles.barCell}><Bar value={m.avg_search_time_s} max={maxAvg} color={color} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
