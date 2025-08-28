import React, { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  LineChart, Line, Legend
} from "recharts";

export default function SlicesPanel({ uploaded }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeIdx, setActiveIdx] = useState(0);

  const runAnalysis = async () => {
    try {
      setLoading(true);
      setErr("");
      const res = await fetch("http://localhost:8000/analyze/slices", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
      });
      if (!res.ok) throw new Error(await res.text());
      const json = await res.json();
      setData(json);
      setActiveIdx(0);
    } catch (e) {
      console.error(e);
      setErr("Failed to run global analysis. Check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  const dims = data?.dimensions || [];
  const active = dims[activeIdx] || {};
  const barData = active.bar || [];
  const line = active.line || { labels: [], series: [] };

  // reshape for Recharts LineChart
  const lineRows = (line.labels || []).map((ds, i) => {
    const row = { ds };
    (line.series || []).forEach(s => {
      row[s.name] = s.data[i];
    });
    return row;
  });

  return (
  <div style={{ padding: "20px ", background: "#ffffffff", border: "1px solid #e5e7eb" }}>
    <h2 style={{ fontSize: "22px", fontWeight: "bold", marginBottom: "20px" }}>
      Global Historical Analysis
    </h2>

    {/* Start Analysis Button */}
    <div style={{ background: "#fff", borderRadius: "10px", padding: "18px", border: "1px solid #e5e7eb", marginBottom: "20px" }}>
      <button
        onClick={runAnalysis}
        disabled={!uploaded || loading}
        style={{
          padding: "10px 20px",
          border: "none",
          borderRadius: "6px",
          fontSize: "14px",
          cursor: (!uploaded || loading) ? "not-allowed" : "pointer",
          background: (!uploaded || loading) ? "#9ca3af" : "#374d6fff",
          color: "white"
        }}
      >
        {loading ? "Analyzing..." : "Start Analysis"}
      </button>
      {err && <div style={{ marginTop: "10px", color: "#dc2626" }}>{err}</div>}
    </div>

    {/* Show charts only after analysis */}
    {data && (
      <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
        
        {/* Tabs */}
        <div style={{ background: "#fff", borderRadius: "10px", padding: "2px 4px" }}>
          <div style={{ display: "flex", gap: "10px" }}>
            {dims.map((d, idx) => (
              <button
                key={d.name}
                onClick={() => setActiveIdx(idx)}
                style={{
                  padding: "8px 14px",
                  border: "none",
                  borderRadius: "6px",
                  cursor: "pointer",
                  background: idx === activeIdx ? "#374d6fff" : "#e5e7eb",
                  color: idx === activeIdx ? "#fff" : "#000",
                  transition: "background 0.2s"
                }}
              >
                {d.name}
              </button>
            ))}
          </div>
        </div>

        <div style={{ background: "#ffffffff", borderRadius: "10px", padding: "18px", border: "1px solid #e5e7eb" }}>
          <div style={{color: "#000", fontWeight: "bold", marginBottom: "0px" }}>Insights</div>
          {Array.isArray(active.insights) && active.insights.length > 0 ? (
            <ul style={{ paddingLeft: "0px" , color: "#000", fontSize: "14px", marginBottom: "30px" }}>
              {active.insights.map((t, i) => <p key={i}>{t}</p>)}
            </ul>
          ) : (
            <p style={{ color: "#096691ff", fontSize: "14px" }}>No AI insights available.</p>
          )}
        

        {/* Charts Section */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", padding : "15px" }}>
          <div style={{ background: "#fff", borderRadius: "10px", padding: "18px" }}>
            <div style={{ fontWeight: "600", marginBottom: "20px" }}>
              Top {barData.length} by {active?.name}
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={barData}>
                <CartesianGrid vertical={false} strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="total" fill="#0094c9ff" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Line Chart Box */}
            <div style={{ background: "#fff", borderRadius: "10px", padding: "18px"}}>
            <div style={{ fontWeight: "600", marginBottom: "20px" }}>
                {active?.name} over time ({active?.freq})
            </div>
            <ResponsiveContainer width="100%" height={300}>
                <LineChart data={lineRows}>
                <CartesianGrid vertical={false} strokeDasharray="3 3" />
                <XAxis dataKey="ds" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Legend />

                {(line.series || []).slice(0, 10).map((s, i) => {
                    const palette = [
                    "#4177a3ff", "#58b4ffff", "#F59E0B", "#EF4444", "#3B82F6",
                    "#8B5CF6", "#EC4899", "#14B8A6", "#F97316", "#84CC16"
                    ];
                    return (
                    <Line
                        key={s.name}
                        type="monotone"
                        dataKey={s.name}
                        stroke={palette[i % palette.length]}
                        strokeWidth={1.1}
                        dot={{ r: 2 }}   // markers enabled
                    />
                    );
                })}
                </LineChart>
            </ResponsiveContainer>
            </div>
            </div>

        </div>

        {/* Insights Section */}
        
      </div>
    )}
  </div>
);


}