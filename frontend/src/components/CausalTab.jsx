// src/components/CausalTab.jsx
import React, { useState } from "react";

export default function CausalTab({ datasetId = "default", outcome = "SALES" }) {
  const [modelInfo, setModelInfo] = useState(null);
  const [points, setPoints] = useState([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const runCausal = async () => {
    setLoading(true);
    setErr("");
    setPoints([]);
    try {
      // Call the /causal/train endpoint with hardcoded config
      const trainRes = await fetch("http://localhost:8000/causal/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: datasetId,
          title: "Causal Model",
          outcome_col: "SALES",                     // replace if needed
          treatment_col: "DISCOUNT",                // replace if needed
          feature_cols: ["REGION", "PRODUCT", "STOCK", "INCOME"] // replace with your columns
        })
      });
      if (!trainRes.ok) throw new Error(await trainRes.text());
      const trained = await trainRes.json();

      setModelInfo({ id: trained.model_name, est: "DRLearner" });

      // Get summary points using the same model name
      const sumRes = await fetch("http://localhost:8000/causal/summary_points", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: trained.model_name,
          k: 6
        })
      });
      if (!sumRes.ok) throw new Error(await sumRes.text());
      const summary = await sumRes.json();
      setPoints(summary.points || []);
    } catch (e) {
      setErr(typeof e === "string" ? e : e.message || "Failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 16, background: "white", borderRadius: 12, border: "1px solid #e5e7eb" }}>
      <div style={{ fontWeight: 700, marginBottom: 6 }}>Causal Analysis</div>
      {modelInfo && (
        <div style={{ marginBottom: 10, color: "#475569", fontSize: 13 }}>
          Model: <b>{modelInfo.est}</b> · outcome: <code>{outcome}</code>
        </div>
      )}
      <button
        onClick={runCausal}
        disabled={loading}
        style={{ padding: "8px 16px", borderRadius: 8, background: "#3b82f6", color: "white", border: "none" }}
      >
        {loading ? "Training & Summarizing…" : "Run Causal Analysis"}
      </button>

      {err && <div style={{ marginTop: 12, color: "#b91c1c" }}>{err}</div>}

      {points.length > 0 && (
        <div style={{ marginTop: 16 }}>
          {points.map((msg, i) => (
            <div key={i} style={{ padding: 12, borderRadius: 8, background: "white" }}>
              {msg}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
