import React, { useState } from "react";
import { computeMap } from "../api";
import styles from "./Step4Metrics.module.css";

function Bar({ value, color = "#8b5cf6" }) {
  const pct = Math.min(100, value * 100);
  return (
    <div className={styles.bar}>
      <div className={styles.barFill} style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

function MetricRow({ label, value, color }) {
  const pct = (value * 100).toFixed(1);
  return (
    <div className={styles.metricRow}>
      <div className={styles.metricTop}>
        <span className={styles.metricLabel}>{label}</span>
        <span className={styles.metricValue} style={{ color }}>{pct}%</span>
      </div>
      <Bar value={value} color={color} />
    </div>
  );
}

function MetricGroup({ title, data }) {
  return (
    <div className={styles.group}>
      <div className={styles.groupTitle}>{title}</div>
      <MetricRow label="Precision"       value={data.precision}          color="#8b5cf6" />
      <MetricRow label="Recall"          value={data.recall}             color="#22d3ee" />
      <MetricRow label="Avg. Precision"  value={data.average_precision}  color="#f59e0b" />
      <MetricRow label="R-Precision"     value={data.r_precision}        color="#34d399" />
    </div>
  );
}

export default function Step4Metrics({ result, config, onNewSearch }) {
  const [mapResult, setMapResult]   = useState(null);
  const [computing, setComputing]   = useState(false);
  const [mapError, setMapError]     = useState("");

  if (!result) return null;

  const { metrics } = result;

  const handleMap = async () => {
    setComputing(true);
    setMapError("");
    setMapResult(null);
    try {
      const data = await computeMap(config.descriptors, config.measure, config.topK, 46);
      setMapResult(data);
    } catch (e) {
      setMapError(e.response?.data?.detail || "Erreur lors du calcul.");
    } finally {
      setComputing(false);
    }
  };

  const mapEntries = mapResult
    ? Object.entries(mapResult.ap_per_class).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <div className={styles.step}>
      {/* Header */}
      <div className={styles.head}>
        <div className={styles.stepId}>04</div>
        <div>
          <h2 className={styles.title}>Métriques d'évaluation</h2>
          <p className={styles.sub}>
            Classe : <strong className={styles.cls}>{result.query_class.replace(/_/g, " ")}</strong>
            <span className={styles.sep}>·</span>
            {metrics.total_relevant_in_db} pertinents dans la base
            <span className={styles.sep}>·</span>
            {(result.search_time_s * 1000).toFixed(0)} ms
          </p>
        </div>
        <button className={styles.newSearchBtn} onClick={onNewSearch}>↺ Nouvelle recherche</button>
      </div>

      {/* Body */}
      <div className={styles.body}>
        {/* Left: metric groups */}
        <div className={styles.metricsPanel}>
          <MetricGroup title="Top 20" data={metrics.top20} />
          <MetricGroup title="Top 50" data={metrics.top50} />

          {/* Legend */}
          <div className={styles.legend}>
            {[
              { c: "#8b5cf6", l: "Precision" },
              { c: "#22d3ee", l: "Recall" },
              { c: "#f59e0b", l: "Avg. Precision" },
              { c: "#34d399", l: "R-Precision" },
            ].map(({ c, l }) => (
              <div key={l} className={styles.legendItem}>
                <span className={styles.legendDot} style={{ background: c }} />
                {l}
              </div>
            ))}
          </div>
        </div>

        {/* Right: MAP panel */}
        <div className={styles.mapPanel}>
          <div className={styles.mapHeader}>
            <span className={styles.mapTitle}>Mean Average Precision</span>
            <span className={styles.mapSub}>1 requête/classe · {config.topK === 50 ? "Top 50" : "Top 20"}</span>
          </div>

          {!mapResult && !computing && (
            <div className={styles.mapCta}>
              <p className={styles.mapDesc}>
                Calcule la MAP sur l'ensemble des classes<br />
                avec la même configuration de recherche.
              </p>
              <div className={styles.configPill}>
                {config.descriptors.join(" + ")} · {config.measure} · Top {config.topK}
              </div>
              <button className={styles.mapBtn} onClick={handleMap}>
                Calculer MAP
              </button>
            </div>
          )}

          {computing && (
            <div className={styles.mapLoading}>
              <div className={styles.orbit}>
                <div className={styles.orbitDot} />
              </div>
              <span>Calcul en cours…</span>
            </div>
          )}

          {mapError && <p className={styles.error}>{mapError}</p>}

          {mapResult && (
            <div className={styles.mapResult}>
              <div className={styles.mapScore}>
                <span className={styles.mapScoreValue}>{(mapResult.map * 100).toFixed(1)}<span className={styles.mapScoreUnit}>%</span></span>
                <span className={styles.mapScoreLbl}>MAP@{mapResult.top_k} — {mapResult.num_queries} classes — {mapResult.elapsed_s}s</span>
              </div>
              <div className={styles.apList}>
                {mapEntries.map(([cls, ap]) => (
                  <div key={cls} className={styles.apRow}>
                    <span className={styles.apCls}>{cls.replace(/_/g, " ")}</span>
                    <div className={styles.apBarWrap}>
                      <div
                        className={styles.apBar}
                        style={{
                          width: `${ap * 100}%`,
                          background: ap > .5 ? "#34d399" : ap > .2 ? "#f59e0b" : "#f87171",
                        }}
                      />
                    </div>
                    <span className={styles.apVal}>{(ap * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className={styles.foot}>
        <button className={styles.newSearchBtnFoot} onClick={onNewSearch}>← Nouvelle recherche</button>
      </div>
    </div>
  );
}
