import React, { useState, useEffect } from "react";

const STORAGE_KEY = "nat_result_v1";

export default function NatTab({ datasetId = "default" }) {
  const [about, setAbout] = useState("");
  const [files, setFiles] = useState(null);
  const [fileContents, setFileContents] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    try {
      if (!about && !files && !fileContents) {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
        if (saved?.about && saved?.files && saved?.fileContents) {
          setAbout(saved.about);
          setFiles(saved.files);
          setFileContents(saved.fileContents);
        }
      }
    } catch {}
  }, [about, files, fileContents]);

  const fetchFileText = async (url) => {
    const res = await fetch(`http://localhost:8000${url}`);
    if (!res.ok) throw new Error("Failed to fetch file: " + url);
    return await res.text();
  };

  const runNat = async () => {
    setErr("");
    setAbout("");
    setFiles(null);
    setFileContents(null);
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/ai/nat/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: datasetId }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setAbout(data.about || "");
      setFiles(data.files || null);

      const [forecastText, edaText, historicalText] = await Promise.all([
        fetchFileText(data.files.forecast),
        fetchFileText(data.files.eda),
        fetchFileText(data.files.historical),
      ]);

      const allContents = {
        forecast: forecastText,
        eda: edaText,
        historical: historicalText,
      };
      setFileContents(allContents);

      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        about: data.about,
        files: data.files,
        fileContents: allContents,
      }));
    } catch (e) {
      console.error(e);
      setErr("Failed to analyze. Upload a CSV first and check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 16 }}>
      <div style={{
        border: "1px solid #e5e7eb",
        borderRadius: 12,
        padding: 16,
        marginBottom: 16,
        background: "#fff"
      }}>
        <h2 style={{ margin: 0, fontSize: 20 }}>NAT — AI Dataset Setup</h2>
        <p style={{ color: "#6b7280", marginTop: 8 }}>
          Sends a 5-row sample to AI, detects what the data is about, and creates three
          prompt files for Forecast, EDA, and Historical analysis.
        </p>
        <button
          onClick={runNat}
          disabled={loading}
          style={{
            padding: "10px 14px",
            borderRadius: 10,
            border: "1px solid #111827",
            background: loading ? "#e5e7eb" : "#111827",
            color: "#fff",
            cursor: loading ? "not-allowed" : "pointer",
            fontWeight: 600
          }}
        >
          {loading ? "Analyzing…" : "Analyze Sample & Generate Prompts"}
        </button>
        {err && <div style={{ color: "#b91c1c", marginTop: 8 }}>{err}</div>}
      </div>

      {about && (
        <div style={{
          border: "1px solid #e5e7eb",
          borderRadius: 12,
          padding: 16,
          marginBottom: 16,
          background: "#fff"
        }}>
          <div style={{ fontSize: 14, color: "#6b7280" }}>Detected domain</div>
          <div style={{ fontSize: 18, marginTop: 6 }}>{about}</div>
        </div>
      )}

      {files && fileContents && (
        <div style={{ display: "grid", gap: 16 }}>
          {["forecast", "eda", "historical"].map((type) => (
            <div key={type} style={{
              border: "1px solid #e5e7eb",
              borderRadius: 12,
              padding: 16,
              background: "#fff"
            }}>
              <div style={{ fontSize: 14, color: "#6b7280", marginBottom: 6 }}>
                {type.toUpperCase()} Prompt
              </div>
              <a
                href={`http://localhost:8000${files[type]}`}
                target="_blank"
                rel="noreferrer"
                style={{ display: "inline-block", marginBottom: 10 }}
              >
                {type}.txt
              </a>
              <pre style={{
                background: "#f2f8ff",
                border: "#e0e6f2",
                padding: "10px 14px",
                borderRadius: 8,
                fontSize: 13,
                whiteSpace: "pre-wrap",
                overflowX: "auto"
              }}>
                {fileContents[type]}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
