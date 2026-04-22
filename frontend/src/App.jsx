import React, { useState } from "react";
import styles from "./App.module.css";
import ImageBrowser from "./components/ImageBrowser";
import SearchPanel from "./components/SearchPanel";
import ResultsPanel from "./components/ResultsPanel";
import MetricsPanel from "./components/MetricsPanel";
import BenchmarkPanel from "./components/BenchmarkPanel";
import MapPanel from "./components/MapPanel";

const TABS = ["Recherche", "MAP", "Benchmark"];

export default function App() {
  const [activeTab, setActiveTab] = useState("Recherche");
  const [queryImage, setQueryImage] = useState(null);
  const [searchResult, setSearchResult] = useState(null);
  const [searching, setSearching] = useState(false);

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <h1 className={styles.logo}>MIR <span>Image Search</span></h1>
          <nav className={styles.tabs}>
            {TABS.map((t) => (
              <button
                key={t}
                className={`${styles.tab} ${activeTab === t ? styles.tabActive : ""}`}
                onClick={() => setActiveTab(t)}
              >
                {t}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className={styles.main}>
        {activeTab === "Recherche" && (
          <div className={styles.searchLayout}>
            <aside className={styles.sidebar}>
              <SearchPanel
                queryImage={queryImage}
                onSearch={(result) => setSearchResult(result)}
                setSearching={setSearching}
                searching={searching}
              />
              {searchResult && (
                <MetricsPanel metrics={searchResult.metrics} searchTime={searchResult.search_time_s} />
              )}
            </aside>

            <section className={styles.content}>
              {!searchResult ? (
                <ImageBrowser onSelect={setQueryImage} selectedImage={queryImage} />
              ) : (
                <ResultsPanel
                  result={searchResult}
                  onBack={() => setSearchResult(null)}
                  onSelectNew={(img) => { setSearchResult(null); setQueryImage(img); }}
                />
              )}
            </section>
          </div>
        )}

        {activeTab === "MAP" && <MapPanel />}
        {activeTab === "Benchmark" && <BenchmarkPanel />}
      </main>
    </div>
  );
}
