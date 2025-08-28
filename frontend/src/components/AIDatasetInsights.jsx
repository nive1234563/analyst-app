
// components/AIDatasetInsights.jsx
import React, { useState } from "react";

const styles = {
  card: { background: "#fff", border: "1px solid #e5e7eb", borderRadius: 16, padding: 16 },
  h1: { fontWeight: 700, fontSize: 16, marginBottom: 8 },
  subtle: { color: "#6b7280", fontSize: 12.5, marginBottom: 12 },
  row: { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", margin: "8px 0 16px" },
  ghostBtn: { padding: "8px 12px", borderRadius: 10, border: "1px solid #e5e7eb", background: "#fff", cursor: "pointer", fontSize: 13 },
  chip: (ok) => ({
    padding: "6px 10px",
    borderRadius: 999,
    border: "1px solid " + (ok ? "#86efac" : "#fecaca"),
    background: ok ? "#ecfdf5" : "#fff1f2",
    color: ok ? "#166534" : "#991b1b",
    fontSize: 12.5
  }),
  sectionTtl: { fontWeight: 600, margin: "14px 0 8px 0" },
  codebox: { background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, padding: 12, fontSize: 12.5, overflowX: "auto" },
  list: { margin: 0, paddingLeft: 16 },
  tableWrap: { border: "1px solid #e5e7eb", borderRadius: 10, overflow: "hidden" },
  table: { width: "100%", borderCollapse: "collapse", fontSize: 12.5 },
  th: { textAlign: "left", padding: "8px 10px", background: "#f8fafc", borderBottom: "1px solid #e5e7eb" },
  td: { padding: "8px 10px", borderBottom: "1px solid #f1f5f9" },
};

export default function AIDatasetInsights({ uploaded }) {
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [res, setRes] = useState(null);

  const runAI = async () => {
    try {
      setLoading(true);
      setErr("");
      setRes(null);
      const r = await fetch("http://localhost:8000/ai/dataset_insights", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: "default", max_rows: 5 })
      });
      if (!r.ok) throw new Error(await r.text());
      const json = await r.json();
      setRes(json);
    } catch (e) {
      setErr(typeof e === "string" ? e : e.message || "Failed to run AI");
    } finally {
      setLoading(false);
    }
  };

  const feasibility = res?.feasibility || {};
  const sampleRows = res?.sample_rows || [];
  const ai = res?.ai_json;
  const aiText = res?.ai_text;

  return (
    <section style={styles.card}>
      <div style={styles.h1}>AI Dataset Insights</div>
      <div style={styles.subtle}>Uploads a 5-row sample to AI to identify dataset type and recommend analyses.</div>

      <div style={styles.row}>
        <button
          style={{ ...styles.ghostBtn, opacity: uploaded ? 1 : 0.5 }}
          disabled={!uploaded || loading}
          onClick={runAI}
        >
          {loading ? "Analyzing…" : "Run AI on 5 Rows"}
        </button>

        {res && (
          <>
            <span style={styles.chip(feasibility?.forecast?.ok)}>Forecast: {feasibility?.forecast?.ok ? "OK" : "No"}</span>
            <span style={styles.chip(feasibility?.eda?.ok)}>EDA: {feasibility?.eda?.ok ? "OK" : "No"}</span>
            <span style={styles.chip(feasibility?.historical?.ok)}>Historical: {feasibility?.historical?.ok ? "OK" : "No"}</span>
          </>
        )}
      </div>

      {err && <div style={{ color: "#b91c1c", marginBottom: 8 }}>{err}</div>}

      {res && ai && (
        <>
          {ai.summary && (
            <>
              <div style={styles.sectionTtl}>📘 Dataset Summary</div>
              <p>{ai.summary}</p>
            </>
          )}

          {ai.feasibility && (
            <>
              <div style={styles.sectionTtl}>🧪 Feasibility</div>
              <ul style={styles.list}>
                <li><b>Forecast:</b> {ai.feasibility.forecast.ok ? "✅ Yes" : "❌ No"} — {ai.feasibility.forecast.reason}</li>
                <li><b>EDA:</b> {ai.feasibility.eda.ok ? "✅ Yes" : "❌ No"} — {ai.feasibility.eda.reason}</li>
                <li><b>Historical:</b> {ai.feasibility.historical.ok ? "✅ Yes" : "❌ No"} — {ai.feasibility.historical.reason}</li>
              </ul>
            </>
          )}

          {ai.forecast_guidance && (
            <>
              <div style={styles.sectionTtl}>📈 Forecast Guidance</div>
              <ul style={styles.list}>
                <li><b>Time Column:</b> {ai.forecast_guidance.time_col || "Not specified"}</li>
                <li><b>Target Column:</b> {ai.forecast_guidance.target_col || "Not specified"}</li>
                {Array.isArray(ai.forecast_guidance.modifications) && ai.forecast_guidance.modifications.length > 0 && (
                  <li><b>Modifications:</b>
                    <ul>{ai.forecast_guidance.modifications.map((m, i) => <li key={i}>{m}</li>)}</ul>
                  </li>
                )}
              </ul>
            </>
          )}

          {ai.suggested_kpis?.length > 0 && (
            <>
              <div style={styles.sectionTtl}>📊 Suggested KPIs</div>
              <ul style={styles.list}>{ai.suggested_kpis.map((t, i) => <li key={i}>{t}</li>)}</ul>
            </>
          )}

          {ai.suggested_charts?.length > 0 && (
            <>
              <div style={styles.sectionTtl}>📉 Suggested Charts</div>
              <ul style={styles.list}>{ai.suggested_charts.map((t, i) => <li key={i}>{t}</li>)}</ul>
            </>
          )}

          {ai.insight_potential?.length > 0 && (
            <>
              <div style={styles.sectionTtl}>💡 Insight Potential</div>
              <ul style={styles.list}>{ai.insight_potential.map((t, i) => <li key={i}>{t}</li>)}</ul>
            </>
          )}
        </>
      )}

      {/* Always show raw AI response too */}
      {aiText && (
        <>
          <div style={styles.sectionTtl}>AI Output (Raw)</div>
          <pre style={styles.codebox}>{aiText}</pre>
        </>
      )}

      {sampleRows.length > 0 && (
        <>
          <div style={styles.sectionTtl}>🧾 5-row Sample Preview</div>
          <div style={styles.tableWrap}>
            <table style={styles.table}>
              <thead><tr>{Object.keys(sampleRows[0]).map((c) => <th key={c} style={styles.th}>{c}</th>)}</tr></thead>
              <tbody>
                {sampleRows.map((r, i) => (
                  <tr key={i}>{Object.keys(r).map((c) => <td key={c} style={styles.td}>{String(r[c] ?? "")}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}
