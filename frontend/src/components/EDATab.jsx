// src/components/EDATab.jsx
import React, { useState,useEffect } from "react";

export default function EDATab({
  uploaded,
  reportUrl,
  handleGenerateEda, // optional: if you have a "Generate ydata" button elsewhere
  css = {},
}) {
  const [kpis, setKpis] = useState(null);
  const [kpiLoading, setKpiLoading] = useState(false);
  const [kpiError, setKpiError] = useState("");

const fetchKpis = async () => {
  try {
    setKpiError("");
    setKpiLoading(true);

    const res = await fetch("http://localhost:8000/eda/profile/ydata/kpis", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        dataset_id: "default",
        title: "EDA Report"
      })
    });

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`Server responded with ${res.status}: ${errorText}`);
    }

    const data = await res.json();
    setKpis(data);  // ✅ correct

  } catch (err) {
    console.error("KPI fetch failed:", err);
    setKpiError("❌ Failed to fetch KPIs. Please check backend and try again.");
  } finally {
    setKpiLoading(false);
  }
};




  const styles = {
    card: { padding: 16, borderRadius: 12, border: "1px solid #e5e7eb", background: "#fff", ...css.card },
    hgroup: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, ...css.hgroup },
    subttl: { fontSize: 18, fontWeight: 700, ...css.subttl },
    subtle: { color: "#6b7280", fontSize: 13.5, ...css.subtle },
    ghostBtn: {
      border: "1px solid #d1d5db",
      background: "#fff",
      padding: "8px 12px",
      borderRadius: 8,
      fontSize: 14,
      cursor: "pointer",
      ...css.ghostBtn,
    },
    iframeWrap: { height: 560, borderRadius: 12, overflow: "hidden", border: "1px solid #e5e7eb", ...css.iframeWrap },
    iframe: { width: "100%", height: "100%", border: "none", ...css.iframe },
  };

  return (
    <section style={styles.card}>
      <div style={styles.hgroup}>
        <div>
          <div style={styles.subttl}>Exploratory Data Analysis Report</div>
          <div style={styles.subtle}>
            Auto-generate a <b>ydata-profiling</b> report + AI insights for your dataset.
          </div>
        </div>

        {/* Optional: show a generate button if you wire it up from parent */}
        {handleGenerateEda && uploaded && (
          <button style={styles.ghostBtn} onClick={handleGenerateEda}>
            Generate ydata
          </button>
        )}
      </div>

      {!uploaded ? (
        <div style={{ textAlign: "center", color: "#6b7280" }}>
          Upload a CSV to enable reports.
        </div>
      ) : (
        <>
          <button
            style={{ ...styles.ghostBtn, marginBottom: 10 }}
            onClick={fetchKpis}                // ✅ only runs on click
            disabled={!reportUrl || kpiLoading} // block until report exists
            title={!reportUrl ? "Generate the report first" : undefined}
          >
            {kpiLoading ? "Loading..." : "Load Insights"}
          </button>

          <div
            style={{
              border: "1px solid #d1d5db",
              padding: 16,
              borderRadius: 8,
              background: "#ffffffff",
              marginBottom: 16,
            }}
          >
            {kpiLoading ? (
              <div>Loading KPIs…</div>
            ) : kpiError ? (
              <div style={{ color: "#991b1b" }}>{kpiError}</div>
            ) : kpis ? (
              <>
                <div style={{ fontWeight: 700, marginBottom: 6 }}>Business Insights</div>
                <ul style={{ fontSize: 13.5, paddingLeft: 20, marginBottom: 10 }}>
                  {/* <li><b>Rows:</b> {kpis?.summary?.rows}</li>
                  <li><b>Columns:</b> {kpis?.summary?.cols}</li>
                  <li><b>Missing Cells %:</b> {kpis?.summary?.missing_cells_pct}%</li>
                  <li><b>Duplicate Rows %:</b> {kpis?.summary?.duplicate_rows_pct}%</li>
                  <li><b>Health Score:</b> {kpis?.summary?.health_score}/100</li> */}
                </ul>
                {kpis?.insights && (
                  <div
                    style={{
                      // padding: 12,
                      // background: "#f0fdfa",
                      // border: "1px solid #ccfbf1",
                      // borderRadius: 10,
                    }}
                  >
                    {/* <div style={{ fontWeight: 600, marginBottom: 6 }}> Business Insights</div> */}
                    <div style={{ fontSize: 14, whiteSpace: "pre-line",lineHeight: 1.6, color: "#374151" }}>
                      {kpis.insights}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div style={{ color: "#6b7280" }}>No KPIs loaded yet.</div>
            )}
          </div>
        </>
      )}

      {reportUrl ? (
        <div style={styles.iframeWrap}>
          <iframe title="EDA Report" src={reportUrl} style={styles.iframe} />
        </div>
      ) : (
        <div
          style={{
            border: "1px dashed #ccc",
            padding: 20,
            textAlign: "center",
            background: "#fafafa",
            borderRadius: 8,
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 6 }}>No report yet</div>
          <div style={styles.subtle}>
            Click <b>Generate ydata</b> to build your report.
          </div>
        </div>
      )}
    </section>
  );
}
