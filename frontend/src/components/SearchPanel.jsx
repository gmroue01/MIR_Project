import React, { useState } from "react";
import { search } from "../api";
import styles from "./SearchPanel.module.css";

const DESCRIPTORS = [
  { id: "color_histogram", label: "Histo. Couleur" },
  { id: "hog", label: "HOG" },
  { id: "mobilenetv2", label: "MobileNetV2" },
  { id: "resnet50", label: "ResNet50" },
  { id: "vit_base", label: "ViT Base" },
  { id: "dinov2", label: "DinoV2" },
  { id: "sift", label: "SIFT" },
  { id: "orb", label: "ORB" },
];

const MEASURES = [
  { id: "euclidean", label: "Euclidienne", orbOnly: false },
  { id: "cosine", label: "Cosinus", orbOnly: false },
  { id: "chi_square", label: "Chi-square", orbOnly: false },
  { id: "jensen", label: "Jensen-Shannon", orbOnly: false },
  { id: "hamming", label: "Hamming", orbOnly: true },
];

export default function SearchPanel({ queryImage, onSearch, setSearching, searching }) {
  const [selectedDescriptors, setSelectedDescriptors] = useState(["color_histogram"]);
  const [measure, setMeasure] = useState("euclidean");
  const [topK, setTopK] = useState(50);
  const [error, setError] = useState("");

  const orbOnly = selectedDescriptors.length === 1 && selectedDescriptors[0] === "orb";

  const toggleDescriptor = (id) => {
    setSelectedDescriptors((prev) => {
      const next = prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id];
      return next.length === 0 ? prev : next;
    });
    // If we change descriptors and Hamming was selected, reset measure
    if (measure === "hamming" && id !== "orb") {
      setMeasure("euclidean");
    }
  };

  const handleSearch = async () => {
    if (!queryImage) {
      setError("Veuillez sélectionner une image requête.");
      return;
    }
    if (measure === "hamming" && !orbOnly) {
      setError("Hamming nécessite uniquement ORB.");
      return;
    }
    setError("");
    setSearching(true);
    try {
      const result = await search(queryImage.index, selectedDescriptors, measure, topK);
      onSearch(result);
    } catch (e) {
      setError(e.response?.data?.detail || "Erreur lors de la recherche.");
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className={styles.panel}>
      <h2 className={styles.title}>Paramètres</h2>

      <div className={styles.section}>
        <label className={styles.label}>Image requête</label>
        {queryImage ? (
          <div className={styles.queryPreview}>
            <img src={`/images/${queryImage.filename}`} alt="query" className={styles.queryThumb} />
            <span className={styles.queryName}>{queryImage.class.replace(/_/g, " ")}</span>
          </div>
        ) : (
          <p className={styles.hint}>Sélectionnez une image dans la grille →</p>
        )}
      </div>

      <div className={styles.section}>
        <label className={styles.label}>Descripteurs</label>
        <div className={styles.chips}>
          {DESCRIPTORS.map((d) => (
            <button
              key={d.id}
              className={`${styles.chip} ${selectedDescriptors.includes(d.id) ? styles.chipActive : ""}`}
              onClick={() => toggleDescriptor(d.id)}
            >
              {d.label}
            </button>
          ))}
        </div>
        {selectedDescriptors.length > 1 && (
          <p className={styles.hint}>Concaténation : {selectedDescriptors.length} descripteurs</p>
        )}
      </div>

      <div className={styles.section}>
        <label className={styles.label}>Mesure de similarité</label>
        <div className={styles.chips}>
          {MEASURES.map((m) => {
            const disabled = m.id === "hamming" ? !orbOnly : false;
            return (
              <button
                key={m.id}
                className={`${styles.chip} ${measure === m.id ? styles.chipActive : ""} ${disabled ? styles.chipDisabled : ""}`}
                onClick={() => !disabled && setMeasure(m.id)}
                title={disabled ? "Hamming disponible uniquement avec ORB seul" : ""}
              >
                {m.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className={styles.section}>
        <label className={styles.label}>Résultats</label>
        <div className={styles.topkRow}>
          {[20, 50].map((k) => (
            <button
              key={k}
              className={`${styles.chip} ${topK === k ? styles.chipActive : ""}`}
              onClick={() => setTopK(k)}
            >
              Top {k}
            </button>
          ))}
        </div>
      </div>

      {error && <p className={styles.error}>{error}</p>}

      <button
        className={styles.searchBtn}
        onClick={handleSearch}
        disabled={searching || !queryImage}
      >
        {searching ? "Recherche en cours..." : "Lancer la recherche"}
      </button>
    </div>
  );
}
