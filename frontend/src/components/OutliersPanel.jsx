import React, { useEffect, useMemo, useState } from "react";

const COLORS = {
  primary: "#4a69bd",
  primaryLight: "#87a4e4ff",
  bg: "#f7f9fc",
  card: "#ffffff",
  border: "#e6eaf2",
  text: "#233044",
  subtext: "#5b6b82",
  danger: "#dc2626",
};

export default function OutliersPanel({ datasetId = "default", uploaded }) {
  const [schema, setSchema] = useState({ columns: [], numeric: [], dates: [] });
  const [dateCol, setDateCol] = useState("");
  const [targetCol, setTargetCol] = useState("");
  const [groupCol, setGroupCol] = useState("");
  const [method, setMethod] = useState("iqr"); // iqr | zscore | isoforest

  const [loading, setLoading] = useState(false);
  const [detectRes, setDetectRes] = useState(null); // { count, rows, meta }
  const [seriesRes, setSeriesRes] = useState(null); // { points: [{ds, y, is_outlier}], meta }

  // === Styles ===
  const card = {
    background: COLORS.card,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 14,
    padding: 12,
  };
  const header = {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 10,
  };
  const ttl = { fontWeight: 700, color: COLORS.text };
  const subtle = { color: COLORS.subtext, fontSize: 12.5 };
  const row = { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" };
  const sel = {
    padding: "8px 10px",
    borderRadius: 10,
    border: `1px solid ${COLORS.border}`,
    background: "#fff",
  };
  const btn = {
    padding: "10px 14px",
    borderRadius: 10,
    border: "none",
    background: COLORS.primary,
    color: "#fff",
    fontWeight: 700,
    cursor: "pointer",
  };

  // === Fetch schema (columns) so we can populate dropdowns ===
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await fetch("http://localhost:8000/dataset/schema", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dataset_id: datasetId }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json(); // { columns, numeric, dates }
        if (!cancelled) {
          setSchema(data);
          if (!targetCol && data.numeric?.length) setTargetCol(data.numeric[0]);
          if (!dateCol && data.dates?.length) setDateCol(data.dates[0]);
        }
      } catch (e) {
        console.error("schema fetch failed:", e);
      }
    };
    if (uploaded) load();
    return () => { cancelled = true; };
  }, [uploaded, datasetId]); // eslint-disable-line

  const runDetection = async () => {
    setLoading(true);
    setDetectRes(null);
    setSeriesRes(null);
    try {
      // 1) general outliers for the selected target (IQR/Z/IF)
      const det = await fetch("http://localhost:8000/outliers/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: datasetId,
          target_col: targetCol,
          method,
          group_by: groupCol || null,
        }),
      });
      if (!det.ok) throw new Error(await det.text());
      const detJson = await det.json();
      setDetectRes(detJson);

      // 2) time-series z-score if date column selected
      if (dateCol) {
        const ts = await fetch("http://localhost:8000/outliers/timeseries", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dataset_id: datasetId,
            date_col: dateCol,
            target_col: targetCol,
            z_threshold: 3.0,
            window: 7,
          }),
        });
        if (ts.ok) setSeriesRes(await ts.json());
      }
    } catch (e) {
      console.error(e);
      alert("Outlier detection failed. Check backend.");
    } finally {
      setLoading(false);
    }
  };

  // === Simple compact table for outliers ===
  const OutlierTable = ({ rows }) => {
    if (!rows?.length) return <div style={subtle}>No outliers detected.</div>;
    const cols = Object.keys(rows[0]);
    return (
      <div style={{ overflowX: "auto", border: `1px solid ${COLORS.border}`, borderRadius: 10 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr>
              {cols.map((c) => (
                <th key={c} style={{ textAlign: "left", padding: "8px 10px", borderBottom: `1px solid ${COLORS.border}` }}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 50).map((r, i) => (
              <tr key={i}>
                {cols.map((c) => (
                  <td key={c} style={{ padding: "8px 10px", borderTop: `1px solid ${COLORS.border}` }}>
                    {String(r[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // === Minimal TS chart with red markers ===
  const TimeSeriesChart = ({ points }) => {
    if (!points?.length) return null;
    const W = 720, H = 220, pad = 30;
    const ys = points.map(p => +p.y || 0);
    const maxY = Math.max(1, ...ys), minY = Math.min(...ys);
    const step = (W - pad * 2) / Math.max(1, points.length - 1);
    const x = (i) => pad + i * step;
    const y = (v) => H - pad - ((v - minY) / (maxY - minY || 1)) * (H - pad * 2);

    const path = ys.map((v, i) => `${i ? "L" : "M"} ${x(i)} ${y(v)}`).join(" ");

    return (
      <svg width={W} height={H} style={{ background: "#f5f7f9", borderRadius: 10 }}>
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#d5dbe7" />
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#d5dbe7" />
        <path d={path} fill="none" stroke="#1f83ff" strokeWidth="2" />
        {points.map((p, i) =>
          p.is_outlier ? <circle key={i} cx={x(i)} cy={y(+p.y || 0)} r="3.6" fill={COLORS.danger} /> : null
        )}
        <text x={pad} y={14} fontSize="12" fill={COLORS.text}>
          {dateCol} vs {targetCol} (outliers in red)
        </text>
      </svg>
    );
  };

  return (
    <section style={card}>
      <div style={header}>
        <div style={ttl}>Outlier Analysis</div>
        <div style={subtle}>Detect unusual spikes/drops with IQR, Z-score, or IsolationForest</div>
      </div>

      <div style={{ ...row, marginBottom: 12 }}>
        <select style={sel} value={method} onChange={(e) => setMethod(e.target.value)}>
          <option value="iqr">IQR</option>
          <option value="zscore">Z-Score</option>
          <option value="isoforest">IsolationForest</option>
        </select>
        <select style={sel} value={targetCol} onChange={(e) => setTargetCol(e.target.value)}>
          <option value="">Select numeric column</option>
          {schema.numeric.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <select style={sel} value={dateCol} onChange={(e) => setDateCol(e.target.value)}>
          <option value="">(optional) date column</option>
          {schema.dates.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <select style={sel} value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
          <option value="">(optional) group by</option>
          {schema.columns.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <button style={{ ...btn, opacity: uploaded ? 1 : 0.6 }} disabled={!uploaded || loading} onClick={runDetection}>
          {loading ? "Detecting…" : "Run Outlier Detection"}
        </button>
      </div>

      {detectRes && (
        <>
          <div style={{ ...row, margin: "8px 0 12px" }}>
            <div style={{ padding: "8px 12px", borderRadius: 10, background: "#f1f5ff", border: `1px solid ${COLORS.border}` }}>
              <b>{detectRes?.count || 0}</b> outliers detected
            </div>
            {detectRes?.meta?.threshold && (
              <div style={{ padding: "8px 12px", borderRadius: 10, background: "#fefbf3", border: `1px solid ${COLORS.border}` }}>
                Threshold: {JSON.stringify(detectRes.meta.threshold)}
              </div>
            )}
          </div>
          <OutlierTable rows={detectRes.rows} />
        </>
      )}

      <div style={{ height: 12 }} />
      {seriesRes?.points?.length ? (
        <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 12 }}>
          <TimeSeriesChart points={seriesRes.points} />
        </div>
      ) : null}
    </section>
  );
}
