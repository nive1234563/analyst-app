// src/components/CausalInsights.jsx
import React, { useEffect, useMemo, useState } from "react";

const box = {
  card: { background: "white", border: "1px solid #e5e7eb", borderRadius: 12, padding: 16 },
  row: { display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" },
  input: { padding: "8px 10px", border: "1px solid #e5e7eb", borderRadius: 8, fontSize: 13, minWidth: 180 },
  btn: { padding: "8px 12px", borderRadius: 10, border: "1px solid #e5e7eb", background: "white", cursor: "pointer", fontSize: 13 },
  primary: { padding: "8px 12px", borderRadius: 10, border: "1px solid #93c5fd", background: "#eff6ff", cursor: "pointer", fontSize: 13 },
  label: { fontSize: 12.5, color: "#6b7280" },
  title: { fontWeight: 700, marginBottom: 6 },
  subtle: { fontSize: 12.5, color: "#6b7280" },
};

export default function CausalInsights() {
  const [model, setModel] = useState(null);
  const [segmentCol, setSegmentCol] = useState("");
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [insights, setInsights] = useState("");
  const [facts, setFacts] = useState(null);

  const cacheKey = useMemo(() => {
    const mid = model?.model_id || "none";
    const seg = segmentCol || "none";
    return `causal_insights::${mid}::${seg}`;
  }, [model, segmentCol]);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch("http://localhost:8000/causal/latest_model");
        if (res.ok) setModel(await res.json());
        else setModel(null);
      } catch { setModel(null); }
    };
    load();
  }, []);

  useEffect(() => {
    const cached = localStorage.getItem(cacheKey);
    if (cached) {
      const parsed = JSON.parse(cached);
      setInsights(parsed.insights || "");
      setFacts(parsed.facts || null);
    }
  }, [cacheKey]);

  const run = async () => {
    setError("");
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/causal/insights", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: model?.model_id || null,
          segment_col: segmentCol || null,
          k_cards: 6,
          include_deciles: true,
          custom_prompt: prompt || null
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setInsights(data.insights || "");
      setFacts(data.facts || null);
      localStorage.setItem(cacheKey, JSON.stringify({ insights: data.insights, facts: data.facts }));
    } catch (e) {
      setError(typeof e === "string" ? e : e.message || "Failed to generate insights");
    } finally { setLoading(false); }
  };

  const reset = () => { setInsights(""); setFacts(null); localStorage.removeItem(cacheKey); };

  return (
    <div style={box.card}>
      <div style={box.title}>AI Insights (Causal)</div>
      <div style={box.subtle}>Combines effects, segment uplift, deciles & top‑K cards. Persists until you regenerate.</div>
      <div style={{ height: 10 }} />

      <div style={box.row}>
        <div>
          <div style={box.label}>Model</div>
          <div style={box.subtle}>
            {model?.model_id ? (<><b>{model.model_id}</b> · {model.estimator} · treat: <code>{model.treatment_col}</code> · outcome: <code>{model.outcome_col}</code></>) : "No model available"}
          </div>
        </div>
      </div>

      <div style={{ height: 8 }} />
      <div style={box.row}>
        <div>
          <div style={box.label}>Segment column (optional)</div>
          <input placeholder="e.g., COUNTRY or CUSTOMER_SEGMENT" value={segmentCol} onChange={(e)=>setSegmentCol(e.target.value)} style={box.input}/>
        </div>
        <div>
          <div style={box.label}>Custom prompt (optional)</div>
          <input placeholder="Add business context" value={prompt} onChange={(e)=>setPrompt(e.target.value)} style={{...box.input, minWidth: 280}}/>
        </div>

        <button onClick={run} disabled={loading} style={box.primary}>{loading ? "Generating…" : "Generate AI Insights"}</button>
        <button onClick={()=>navigator.clipboard.writeText(insights || "")} disabled={!insights} style={box.btn}>Copy</button>
        <button onClick={reset} disabled={!insights} style={box.btn}>Reset</button>
      </div>

      {error && (<><div style={{ height: 10 }} /><div style={{ color: "#b91c1c", fontSize: 13 }}>{error}</div></>)}

      {insights && (<><div style={{ height: 16 }} /><div style={{ ...box.card, border: "1px solid #eef2ff", background: "#f8fafc" }}>
        <div style={{ ...box.label, marginBottom: 6 }}>AI Summary</div><div style={{ fontSize: 14 }}>{insights}</div></div></>)}

      {facts && (<><div style={{ height: 12 }} /><details><summary style={{ cursor: "pointer", fontSize: 13, color: "#475569" }}>Show raw facts (debug)</summary>
        <pre style={{ fontSize: 12.5, whiteSpace: "pre-wrap" }}>{JSON.stringify(facts, null, 2)}</pre></details></>)}
    </div>
  );
}
