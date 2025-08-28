import React, { useState } from "react";

export default function AggregationChartsPanel() {
  const [loading, setLoading] = useState(false);
  const [pairs, setPairs] = useState([]);
  const [aggregations, setAggregations] = useState([]);
  const [error, setError] = useState("");
  const [details, setDetails] = useState({});

  const container = { padding: 16, background: "#fff" };
  const box = {
    border: "1px solid #e7ebf4",
    borderRadius: 12,
    padding: 18,
    background: "#fff",
    boxShadow: "0 2px 10px rgba(0,0,0,0.04)",
    marginLeft: 40,
    marginRight: 40,
    marginBottom: 16,
  };
  const btn = {
    padding: "8px 12px",
    borderRadius: 8,
    border: "1px solid #4b6397",
    background: "#e8eefa",
    color: "#2d3f64",
    cursor: "pointer",
  };
  const kpiCard = {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    border: "1px solid #e7ebf4",
    borderRadius: 8,
    padding: 12,
    background: "#fff",
    boxShadow: "0 2px 10px rgba(0,0,0,0.04)",
    minHeight: 96,
    textAlign: "center",
  };

  const startAnalysis = async () => {
    setError("");
    setLoading(true);
    setDetails({});
    setAggregations([]);
    try {
      const res = await fetch("http://localhost:8000/compare/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: "default" }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(typeof body === "string" ? body : JSON.stringify(body));

      const fetchedPairs = body.pairs || [];
      const fetchedAggregations = body.aggregations || [];
      console.log("Aggregations from server:", fetchedAggregations);
      setAggregations(fetchedAggregations);   // <-- KPIs rendered immediately
      setPairs(fetchedPairs);

      // Begin loading insights in the background (charts already visible with placeholders)
      const insightPromises = fetchedPairs.map((p) =>
        fetch(`http://localhost:8000/compare/insight/${p.id}`)
          .then((r) => r.json())
          .then((d) => ({ id: p.id, data: d }))
          .catch((e) => ({ id: p.id, data: { chart: null, insight: "Failed to load insight." } }))
      );

      const all = await Promise.all(insightPromises);
      const next = all.reduce((acc, cur) => { acc[cur.id] = cur.data; return acc; }, {});
      setDetails(next);
    } catch (e) {
      console.error(e);
      setError("Failed to start analysis. Check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  // --- simple SVG charts (same as before) ---
  const Chart = ({ chart }) => {
    if (!chart) return null;
    const t = String(chart.type || "").toLowerCase();
    if (t === "bar") return <BarChart data={chart} />;
    if (t === "line") return <LineChart data={chart} />;
    if (t === "scatter") return <ScatterChart data={chart} />;
    return <div style={{ color: "#5b7fc8", marginTop: 6 }}>Chart type “{chart.type}” not supported.</div>;
  };
  const BarChart = ({ data }) => {
    const W = 560, H = 220, pad = 30;
    const xs = data.x || [];
    const ys = (data.y || []).map((v) => (typeof v === "number" && !isNaN(v) ? v : 0));
    const maxY = Math.max(1, ...ys);
    const bw = (W - pad * 2) / Math.max(1, xs.length);
    return (
      <svg width={W} height={H} style={{ background: "#f5f7f9", borderRadius: 4 }}>
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#ccc" />
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#ccc" />
        {ys.map((y, i) => {
          const h = (y / maxY) * (H - pad * 2);
          return (
            <g key={i}>
              <rect x={pad + i * bw + 4} y={H - pad - h} width={bw - 8} height={h} fill="#0ae4ec" />
              <text x={pad + i * bw + bw / 2} y={H - pad + 12} fontSize="10" textAnchor="middle" fill="#9ca3ad">
                {String(xs[i]).slice(0, 10)}
              </text>
            </g>
          );
        })}
        <text x={pad} y={14} fontSize="12" fill="#111827">{data.yLabel}</text>
      </svg>
    );
  };
  const LineChart = ({ data }) => {
    const W = 560, H = 220, pad = 30;
    const xs = data.x || [];
    const ys = (data.y || []).map((v) => (typeof v === "number" && !isNaN(v) ? v : 0));
    const maxY = Math.max(1, ...ys);
    const step = (W - pad * 2) / Math.max(1, xs.length - 1);
    const points = ys.map((y, i) => {
      const px = pad + i * step;
      const py = H - pad - (y / maxY) * (H - pad * 2);
      return `${px},${py}`;
    }).join(" ");
    return (
      <svg width={W} height={H} style={{ background: "#f5f7f9", borderRadius: 4 }}>
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#ccc" />
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#ccc" />
        <polyline fill="none" stroke="#30e3ff" strokeWidth="2" points={points} />
        {ys.map((y, i) => {
          const px = pad + i * step;
          const py = H - pad - (y / maxY) * (H - pad * 2);
          return <circle key={i} cx={px} cy={py} r="3" fill="#077fc5" />;
        })}
        <text x={pad} y={14} fontSize="12" fill="#111827">{data.yLabel}</text>
      </svg>
    );
  };
  const ScatterChart = ({ data }) => {
    const W = 560, H = 220, pad = 30;
    const xs = (data.x || []).map((v) => (typeof v === "number" ? v : 0));
    const ys = (data.y || []).map((v) => (typeof v === "number" ? v : 0));
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const sx = (x) => pad + ((x - minX) / (maxX - minX || 1)) * (W - pad * 2);
    const sy = (y) => H - pad - ((y - minY) / (maxY - minY || 1)) * (H - pad * 2);
    return (
      <svg width={W} height={H} style={{ background: "#f5f7f9", borderRadius: 4 }}>
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#ccc" />
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#ccc" />
        {xs.map((x, i) => <circle key={i} cx={sx(x)} cy={sy(ys[i])} r="3" fill="#f5ab43" />)}
        <text x={pad} y={14} fontSize="12" fill="#111827">{data.xLabel} vs {data.yLabel}</text>
      </svg>
    );
  };

  return (
    <div style={container}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
        <button style={btn} onClick={startAnalysis} disabled={loading}>
          {loading ? "Analyzing..." : "Start Analysis"}
        </button>
        {error && <div style={{ color: "#b91c1c", marginLeft: 12 }}>{error}</div>}
      </div>

      {/* KPI grid: fixed 4 per row, always shown once we have the array */}
      <div style={{ ...box, marginTop: 4 }}>
        <h3 style={{ margin: "0 0 12px 0", color: "#303c4b", fontSize: 16 }}>Key Metrics</h3>
        {aggregations && aggregations.length > 0 ? (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, minmax(180px, 1fr))",
              gap: 16,
            }}
          >
            {aggregations.slice(0, 8).map((agg, idx) => (
              <div key={idx} style={kpiCard}>
                <h4 style={{ margin: 0, fontSize: 14, color: "#303c4b" }}>{agg.name}</h4>
                <p style={{ margin: "8px 0 0 0", fontSize: 24, fontWeight: "bold", color: "#2d3f64" }}>
                  {agg.value !== null && agg.value !== undefined ? Number(agg.value).toLocaleString() : "N/A"}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ color: "#6b7280" }}>
            {loading ? "Deriving KPIs…" : "No KPIs yet. Click Start Analysis."}
          </div>
        )}
      </div>

      {/* Chart cards */}
      {pairs.length === 0 && !loading && (
        <div style={{ ...box, textAlign: "center", color: "#6b7280" }}>
          Click <b>Start Analysis</b>
        </div>
      )}

      <div style={{ display: "flex", flexWrap: "wrap", gap: 16 }}>
        {pairs.map((p) => {
          const d = details[p.id];
          return (
            <div key={p.id} style={{ ...box, flex: "1 1 calc(50% - 16px)" }}>
              <h3 style={{ fontSize: 16, color: "#303c4b" }}>{p.title}</h3>
              <div style={{ color: "#9fa4ae", fontSize: 12, marginBottom: 18 }}>
                {p.x_col} vs {p.y_col} · chart: <b>{p.chart_type}</b> · name: {p.chart_name}
              </div>
              <div style={{ marginBottom: 20, color: "#586071" }}>{p.reason}</div>
              <div style={{ marginTop: 14, display: "flex", gap: 16 }}>
                <div style={{ flex: "1 1 50%" }}>{d?.chart && <Chart chart={d.chart} />}</div>
                <div style={{ flex: "1 1 50%" }}>
                  <p style={{ font: "12pt", color: "#586071" }}>{d?.insight || "Loading..."}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
