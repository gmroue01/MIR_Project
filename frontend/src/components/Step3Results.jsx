import React, { useState } from "react";
import styles from "./Step3Results.module.css";

export default function Step3Results({ result, onNext, onNewSearch }) {
  const [view, setView] = useState(50);
  if (!result) return null;

  const results = view === 20 ? result.results_top20 : result.results_top50;
  const relevant = results.filter(r => r.class === result.query_class).length;

  return (
    <div className={styles.step}>
      {/* Header */}
      <div className={styles.head}>
        <div className={styles.queryBox}>
          <img src={`/images/${result.query_filename}`} alt="query" className={styles.queryImg} />
          <div>
            <span className={styles.queryLbl}>Requête</span>
            <span className={styles.queryClass}>{result.query_class.replace(/_/g, " ")}</span>
          </div>
        </div>

        <div className={styles.headCenter}>
          <div className={styles.stepId}>03</div>
          <div>
            <h2 className={styles.title}>Résultats de recherche</h2>
            <p className={styles.sub}>
              <span className={styles.statRelevant}>{relevant} pertinents</span>
              <span className={styles.statSep}>/</span>
              <span className={styles.statTotal}>{results.length} résultats</span>
            </p>
          </div>
        </div>

        <div className={styles.headRight}>
          <div className={styles.toggle}>
            {[20, 50].map(k => (
              <button
                key={k}
                className={`${styles.toggleBtn} ${view === k ? styles.toggleOn : ""}`}
                onClick={() => setView(k)}
              >
                Top {k}
              </button>
            ))}
          </div>
          <button className={styles.newSearchBtn} onClick={onNewSearch}>↺ Nouvelle</button>
        </div>
      </div>

      {/* Grid */}
      <div className={styles.body}>
        <div className={styles.grid}>
          {results.map(r => {
            const ok = r.class === result.query_class;
            return (
              <div key={r.index} className={`${styles.card} ${ok ? styles.cardOk : styles.cardKo}`} title={`${r.class.replace(/_/g, " ")}\nd = ${r.distance.toFixed(4)}`}>
                <span className={styles.rank}>#{r.rank}</span>
                <img src={`/images/${r.filename}`} alt={r.filename} loading="lazy" className={styles.thumb} />
                <span className={`${styles.dot} ${ok ? styles.dotOk : styles.dotKo}`} />
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer */}
      <div className={styles.foot}>
        <div className={styles.legend}>
          <span className={styles.legendDotOk} /> Pertinent
          <span className={styles.legendDotKo} style={{ marginLeft: "1.5rem" }} /> Non pertinent
        </div>
        <button className={styles.nextBtn} onClick={onNext}>
          Voir les métriques <span>→</span>
        </button>
      </div>
    </div>
  );
}
