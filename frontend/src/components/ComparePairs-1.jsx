import React, { useState } from "react";

export default function ComparePairsPanel() {
  const [loading, setLoading] = useState(false);
  const [pairs, setPairs] = useState([]);
  const [error, setError] = useState("");
  const [openId, setOpenId] = useState(null);
  const [insightData, setInsightData] = useState({}); // pairId -> {chart, insight}

  const box = {
    border: "1px solid #e5e7eb",
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    background: "#fff",
    boxShadow: "0 2px 10px rgba(0,0,0,0.04)",
  };
  const btn = {
    padding: "8px 12px",
    borderRadius: 8,
    border: "1px solid #111827",
    background: "#111827",
    color: "#fff",
    cursor: "pointer",
  };
  const ghost = {
    padding: "8px 12px",
    borderRadius: 8,
    border: "1px solid #e5e7eb",
    background: "#f9fafb",
    color: "#111827",
    cursor: "pointer",
    marginLeft: 8,
  };
  const small = { color: "#6b7280", fontSize: 12 };
  const h = { margin: 0, fontSize: 16 };

  const startAnalysis = async () => {
    setError("");
    setLoading(true);
    setOpenId(null);
    setInsightData({});
    try {
      const res = await fetch("http://localhost:8000/compare/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: "default" }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setPairs(data.pairs || []);
    } catch (e) {
      setError("Failed to start analysis. Check backend.");
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const viewInsights = async (id) => {
    if (insightData[id]) {
      setOpenId(openId === id ? null : id);
      return;
    }
    try {
      const res = await fetch(`http://localhost:8000/compare/insight/${id}`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setInsightData((prev) => ({ ...prev, [id]: data }));
      setOpenId(id);
    } catch (e) {
      setError("Failed to load insights.");
      console.error(e);
    }
  };

  // Simple SVG renderers (no dependencies)
  const Chart = ({ chart }) => {
    if (!chart) return null;
    if (chart.type === "bar") return <BarChart data={chart} />;
    if (chart.type === "line") return <LineChart data={chart} />;
    if (chart.type === "scatter") return <ScatterChart data={chart} />;
    return null;
  };

  const BarChart = ({ data }) => {
    const W = 560, H = 220, pad = 30;
    const xs = data.x || [];
    const ys = data.y || [];


    const numericYs = ys.map((v) => (typeof v === "number" && !isNaN(v) ? v : 0));
    const maxY = Math.max(1, ...numericYs);
    const bw = (W - pad * 2) / Math.max(1, xs.length);


    return (
    <svg width={W} height={H} style={{ background: "#fff" }}>
    {/* Axes */}
    <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#ccc" />
    <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#ccc" />


    {/* Bars */}
    {numericYs.map((y, i) => {
    const h = (y / maxY) * (H - pad * 2);
    return (
    <g key={i}>
    <rect
    x={pad + i * bw + 4}
    y={H - pad - h}
    width={bw - 8}
    height={h}
    fill="#4b5563"
    />
    <text
    x={pad + i * bw + bw / 2}
    y={H - pad + 12}
    fontSize="10"
    textAnchor="middle"
    fill="#374151"
    >
    {String(xs[i]).slice(0, 10)}
    </text>
    </g>
    );
    })}


    {/* Y-axis label */}
    <text x={pad} y={14} fontSize="12" fill="#111827">
    {data.yLabel}
    </text>
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
    <svg width={W} height={H} style={{ background: "#fff" }}>
    <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#ccc" />
    <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#ccc" />


    <polyline fill="none" stroke="#4b5563" strokeWidth="2" points={points} />
    {ys.map((y, i) => {
    const px = pad + i * step;
    const py = H - pad - (y / maxY) * (H - pad * 2);
    return <circle key={i} cx={px} cy={py} r="3" fill="#111827" />;
    })}


    {xs.map((label, i) => (
    i % Math.ceil(xs.length / 8) === 0 && (
    <text
    key={i}
    x={pad + i * step}
    y={H - pad + 12}
    fontSize="10"
    textAnchor="middle"
    fill="#374151"
    >
    {String(label).slice(0, 10)}
    </text>
    )
    ))}


    <text x={pad} y={14} fontSize="12" fill="#111827">
    {data.yLabel}
    </text>
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
        <svg width={W} height={H} style={{ background: "#fff" }}>
        {/* Axes */}
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#ccc" />
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#ccc" />

        {/* Axis labels */}
        <text x={pad - 8} y={pad - 4} fontSize="10" textAnchor="end" fill="#999">{maxY.toFixed(1)}</text>
        <text x={pad - 8} y={H - pad} fontSize="10" textAnchor="end" fill="#999">{minY.toFixed(1)}</text>

        <text x={pad} y={H - pad + 20} fontSize="10" textAnchor="start" fill="#999">{minX.toFixed(1)}</text>
        <text x={W - pad} y={H - pad + 20} fontSize="10" textAnchor="end" fill="#999">{maxX.toFixed(1)}</text>

        {/* Dots */}
        {xs.map((x, i) => (
            <circle key={i} cx={sx(x)} cy={sy(ys[i])} r="3" fill="#ea580c" />
        ))}

        {/* Label */}
        <text x={pad} y={14} fontSize="12" fill="#111827">
            {data.xLabel} vs {data.yLabel}
        </text>
        </svg>
    );
    };


  return (
    <div style={{  padding: 16 }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
        <button style={btn} onClick={startAnalysis} disabled={loading}>
          {loading ? "Analyzing..." : "Start Analysis"}
        </button>
        {/* <span style={small}>Runs one AI call to propose pairs & tiny formulas. No date/sales assumptions.</span> */}
      </div>

      {error && <div style={{ color: "#b91c1c", marginBottom: 8 }}>{error}</div>}

      {pairs.length === 0 && !loading && (
        <div style={{ ...box, textAlign: "center", color: "#6b7280" }}>
          Click <b>Start Analysis</b>
        </div>
      )}

      {pairs.map((p) => {
        const details = insightData[p.id];
        // const isOpen = openId === p.id;
        return (
          <div key={p.id} style={box}>
            <h3 style={h}>{p.title}</h3>
            <div style={{ ...small, marginBottom: 8 }}>
              {p.left} vs {p.right} · chart: {p.chart}
            </div>
            <div style={{ marginBottom: 8, color: "#374151" }}>{p.reason}</div>
            <div>
               <button style={btn} onClick={() => viewInsights(p.id)}>
               { "View insights"} 
              </button> 
              { details && (
                <button style={ghost} onClick={() => setOpenId(null)}>Close</button>
              )}
            </div>

            { details && (
              <div style={{ marginTop: 12 }}>
                <Chart chart={details.chart} />
                <p style={{ marginTop: 8, color: "#111827" }}>{details.insight}</p>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
