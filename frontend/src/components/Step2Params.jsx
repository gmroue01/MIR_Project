import React, { useState } from "react";
import { search } from "../api";
import styles from "./Step2Params.module.css";

const DESCRIPTORS = [
  { id: "color_histogram", label: "Histo. Couleur" },
  { id: "hog",             label: "HOG" },
  { id: "mobilenetv2",     label: "MobileNetV2" },
  { id: "resnet50",        label: "ResNet50" },
  { id: "vit_base",        label: "ViT Base" },
  { id: "dinov2",          label: "DinoV2" },
  { id: "sift",            label: "SIFT" },
  { id: "orb",             label: "ORB" },
];

const MEASURES = [
  { id: "euclidean",  label: "Euclidienne" },
  { id: "cosine",     label: "Cosinus" },
  { id: "chi_square", label: "Chi-carré" },
  { id: "jensen",     label: "Jensen-Shannon" },
  { id: "hamming",    label: "Hamming", orbOnly: true },
];

export default function Step2Params({ queryImage, config, onChange, onSearch, searching, setSearching, onBack }) {
  const [error, setError] = useState("");

  const { descriptors, measure, topK } = config;
  const orbOnly = descriptors.length === 1 && descriptors[0] === "orb";

  const toggleDesc = (id) => {
    const next = descriptors.includes(id)
      ? descriptors.filter(d => d !== id)
      : [...descriptors, id];
    if (next.length === 0) return;
    onChange({ ...config, descriptors: next, measure: next.length === 1 && next[0] === "orb" ? measure : measure === "hamming" ? "euclidean" : measure });
  };

  const handleSearch = async () => {
    if (!queryImage) { setError("Aucune image sélectionnée."); return; }
    if (measure === "hamming" && !orbOnly) { setError("Hamming est disponible uniquement avec ORB seul."); return; }
    setError("");
    setSearching(true);
    try {
      const result = await search(queryImage.index, descriptors, measure, topK);
      onSearch(result);
    } catch (e) {
      setError(e.response?.data?.detail || "Erreur lors de la recherche.");
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className={styles.step}>
      <div className={styles.head}>
        <div className={styles.stepId}>02</div>
        <div>
          <h2 className={styles.title}>Configurer la recherche</h2>
          <p className={styles.sub}>Choisissez les descripteurs et la mesure de similarité</p>
        </div>
      </div>

      <div className={styles.body}>
        {/* Left: image preview */}
        <div className={styles.preview}>
          {queryImage ? (
            <>
              <div className={styles.imgWrap}>
                <img src={`/images/${queryImage.filename}`} alt="query" className={styles.img} />
                <div className={styles.imgGlow} />
              </div>
              <div className={styles.imgInfo}>
                <span className={styles.imgLabel}>Image requête</span>
                <span className={styles.imgClass}>{queryImage.class.replace(/_/g, " ")}</span>
                <button className={styles.changeBtn} onClick={onBack}>← Changer</button>
              </div>
            </>
          ) : (
            <div className={styles.noImg}>
              <span>Aucune image sélectionnée</span>
              <button className={styles.changeBtn} onClick={onBack}>← Retour</button>
            </div>
          )}
        </div>

        {/* Right: config */}
        <div className={styles.config}>
          <div className={styles.section}>
            <label className={styles.label}>
              Descripteurs
              {descriptors.length > 1 && <span className={styles.badge}>concaténés ×{descriptors.length}</span>}
            </label>
            <div className={styles.chips}>
              {DESCRIPTORS.map(d => (
                <button
                  key={d.id}
                  className={`${styles.chip} ${descriptors.includes(d.id) ? styles.chipOn : ""}`}
                  onClick={() => toggleDesc(d.id)}
                >
                  {d.label}
                </button>
              ))}
            </div>
          </div>

          <div className={styles.section}>
            <label className={styles.label}>Mesure de similarité</label>
            <div className={styles.chips}>
              {MEASURES.map(m => {
                const disabled = m.orbOnly && !orbOnly;
                return (
                  <button
                    key={m.id}
                    className={`${styles.chip} ${measure === m.id ? styles.chipOn : ""} ${disabled ? styles.chipOff : ""}`}
                    onClick={() => !disabled && onChange({ ...config, measure: m.id })}
                    title={disabled ? "Hamming disponible uniquement avec ORB seul" : ""}
                    disabled={disabled}
                  >
                    {m.label}
                  </button>
                );
              })}
            </div>
          </div>

          <div className={styles.section}>
            <label className={styles.label}>Résultats</label>
            <div className={styles.chips}>
              {[20, 50].map(k => (
                <button
                  key={k}
                  className={`${styles.chip} ${topK === k ? styles.chipOn : ""}`}
                  onClick={() => onChange({ ...config, topK: k })}
                >
                  Top {k}
                </button>
              ))}
            </div>
          </div>

          {error && <p className={styles.error}>{error}</p>}
        </div>
      </div>

      <div className={styles.foot}>
        <div className={styles.configSummary}>
          <span className={styles.summaryItem}>{descriptors.map(d => DESCRIPTORS.find(x => x.id === d)?.label).join(" + ")}</span>
          <span className={styles.summaryDot}>·</span>
          <span className={styles.summaryItem}>{MEASURES.find(m => m.id === measure)?.label}</span>
          <span className={styles.summaryDot}>·</span>
          <span className={styles.summaryItem}>Top {topK}</span>
        </div>
        <button
          className={styles.searchBtn}
          onClick={handleSearch}
          disabled={searching || !queryImage}
        >
          {searching ? (
            <><span className={styles.spinner} /> Recherche en cours…</>
          ) : (
            <>Lancer la recherche <span className={styles.arrow}>→</span></>
          )}
        </button>
      </div>
    </div>
  );
}
