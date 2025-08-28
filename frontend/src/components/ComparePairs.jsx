import React, { useState } from "react";

export default function ComparePairsPanel() {
  const [loading, setLoading] = useState(false);
  const [pairs, setPairs] = useState([]);            // [{ id, title, x_col, y_col, chart_type, chart_name, reason }]
  const [error, setError] = useState("");
  const [details, setDetails] = useState({});        // id -> { chart, insight }

  const box = {
    border: "1px solid #e7ebf4ff",
    borderRadius: 12,
    padding: 18,
    marginBottom: 12,
    background: "#fff",
    boxShadow: "0 2px 10px rgba(0,0,0,0.04)",
    
    marginBottom:8,
  };

  const btn = {
  display: "inline-flex",          // makes centering easy
  alignItems: "center",
  justifyContent: "center",
  height: "32px",                  // fixed height for all buttons
  minWidth: "140px",               // optional: consistent width
  padding: "0 16px",               // horizontal padding
  borderRadius: 8,
  border: "1px solid #528af9",     // strong blue border
  background: "#e8eefa",           // light background
  color: "#2d55f0",                // text color
  fontSize: "14px",
  fontWeight: "normal",
  cursor: "pointer",
  transition: "all 0.2s ease-in-out",
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
  const small = { color: "#9fa4aeff", fontSize: 12 };
  const h = { marginTop:22, marginBottom:4, fontSize: 16 , color : "#303c4bff"};

  const startAnalysis = async () => {
    setError("");
    setLoading(true);
    setDetails({});
    try {
      const res = await fetch("http://localhost:8000/compare/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: "default" }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const fetchedPairs = data.pairs || [];

      // Fetch all insights concurrently
      const insightPromises = fetchedPairs.map(p =>
        fetch(`http://localhost:8000/compare/insight/${p.id}`)
          .then(res => res.json())
          .then(insightData => ({ id: p.id, data: insightData }))
          .catch(e => {
            console.error(`Failed to load insight for pair ${p.id}:`, e);
            return { id: p.id, data: { chart: null, insight: "Failed to load insight." } };
          })
      );

      const allInsights = await Promise.all(insightPromises);
      const newDetails = allInsights.reduce((acc, current) => {
        acc[current.id] = current.data;
        return acc;
      }, {});

      setPairs(fetchedPairs);
      setDetails(newDetails);

    } catch (e) {
      setError("Failed to start analysis. Check backend.");
      console.error(e);
    } finally {
      setLoading(false);
    }
  };


  // -------- Minimal SVG chart renderers ----------
  const Chart = ({ chart }) => {
    if (!chart) return null;
    const t = String(chart.type || "").toLowerCase();
    if (t === "bar") return <BarChart data={chart} />;
    if (t === "line") return <LineChart data={chart} />;
    if (t === "scatter") return <ScatterChart data={chart} />;
    return <div style={{ color: "#5b7fc8ff", marginTop: 6 }}>Chart type “{chart.type}” not supported.</div>;
  };

  const BarChart = ({ data }) => {
    const W = 560, H = 220, pad = 30;
    const xs = data.x || [];
    const ys = (data.y || []).map((v) => (typeof v === "number" && !isNaN(v) ? v : 0));
    const maxY = Math.max(1, ...ys);
    const bw = (W - pad * 2) / Math.max(1, xs.length);

    return (
      <svg width={W} height={H} style={{  background:"#f5f7f9ff" , borderRadius : "4pt",padding:"18 8"}}>
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#ccc" />
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#ccc" />
        {ys.map((y, i) => {
          const h = (y / maxY) * (H - pad * 2);
          return (
            <g key={i}>
              <rect x={pad + i * bw + 4} y={H - pad - h} width={bw - 8} height={h} fill="#0ae4ecff" />
              <text x={pad + i * bw + bw / 2} y={H - pad + 12} fontSize="10" textAnchor="middle" fill="#9ca3adff">
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
      <svg width={W} height={H} style={{  background:"#f5f7f9ff" , borderRadius : "4pt",padding:"18 8"}}>
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#ccc" />
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#ccc" />
        <polyline fill="none" stroke="#30e3ffff" strokeWidth="2" points={points} />
        {ys.map((y, i) => {
          const px = pad + i * step;
          const py = H - pad - (y / maxY) * (H - pad * 2);
          return <circle key={i} cx={px} cy={py} r="3" fill="#077fc5ff" />;
        })}
        {xs.map((label, i) =>
          i % Math.ceil(xs.length / 8) === 0 ? (
            <text key={i} x={pad + i * step} y={H - pad + 12} fontSize="10" textAnchor="middle" fill="#374151">
              {String(label).slice(0, 10)}
            </text>
          ) : null
        )}
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
      <svg width={W} height={H} style={{  background:"#f5f7f9ff" , borderRadius : "4pt",padding:"18 8" }}>
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#ccccccff" />
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#ccc" />
        <text x={pad - 8} y={pad - 4} fontSize="10" textAnchor="end" fill="#999">{Number.isFinite(maxY) ? maxY.toFixed(1) : ""}</text>
        <text x={pad - 8} y={H - pad} fontSize="10" textAnchor="end" fill="#999">{Number.isFinite(minY) ? minY.toFixed(1) : ""}</text>
        <text x={pad} y={H - pad + 20} fontSize="10" textAnchor="start" fill="#999">{Number.isFinite(minX) ? minX.toFixed(1) : ""}</text>
        <text x={W - pad} y={H - pad + 20} fontSize="10" textAnchor="end" fill="#999">{Number.isFinite(maxX) ? maxX.toFixed(1) : ""}</text>
        {xs.map((x, i) => <circle key={i} cx={sx(x)} cy={sy(ys[i])} r="3" fill="#f5ab43ff" />)}
        <text x={pad} y={14} fontSize="12" fill="#111827">{data.xLabel} vs {data.yLabel}</text>
      </svg>
    );
  };

  return (
    <div style={{ padding: 16, borderRadius:12 }}>
      <div style={{ background:"#fff", marginBottom:8}}>
      <div ><p><b>Historical Analysis</b></p></div>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
        <button style={btn} onClick={startAnalysis} disabled={loading}>
          {loading ? "Analyzing..." : "Start Analysis"}
        </button>
      </div>
      </div>
      {error && <div style={{ color: "#b91c1c", marginBottom: 8 }}>{error}</div>}

      {pairs.length === 0 && !loading && (
        <div style={{ ...box, textAlign: "center", color: "#6b7280" }}>
          Start Analysis to see Detailed Comparisons
        </div>
      )}
      
      {pairs.map((p) => {
        const d = details[p.id];
        return (
          <div key={p.id} >
            <h3 style={h}>{p.title}</h3>
            <div style={{ ...small, marginBottom: 6 }}>
              {p.x_col} vs {p.y_col} · chart: <b>{p.chart_type}</b> · name: {p.chart_name}
            </div>
            <div style={{ marginBottom: 8, color: "#586071ff" }}>{p.reason}</div>
            <div style={{ ...box }}>
              <div style={{ marginTop: 14, display: 'flex', gap: 16 }}>
                  <div style={{ flex: '1 1 50%' }}>
                    {d?.chart && <Chart chart={d.chart} />}
                  </div>
                  <div style={{ flex: '1 1 50%' }}>
                    <p style={{ font:"12pt", color: "#586071ff" }}>{d?.insight || "Loading..."}</p>
                  </div>
              </div>
            </div>

          </div>
        );
      })}
    </div>
  );
}