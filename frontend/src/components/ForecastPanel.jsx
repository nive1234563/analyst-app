
import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  forecastTrain,
  forecastPredict,
  forecastPlot,
  forecastInsights,
} from "../api/forecast";

export default function ForecastPanel({ uploaded }) {
  const datasetId = "default";
  const [horizon, setHorizon] = useState(30);
  const [status, setStatus] = useState("");
  const [modelInfo, setModelInfo] = useState(null);
  const [imgHistory, setImgHistory] = useState("");
  const [imgForecast, setImgForecast] = useState("");
  const [imgDecomp, setImgDecomp] = useState("");
  const [insights, setInsights] = useState(() => {
    return JSON.parse(localStorage.getItem("forecast_ai_insights") || "{}");
  });

  const [forecastCheckResult, setForecastCheckResult] = useState(null);
  const [forecastError, setForecastError] = useState("");

  useEffect(() => {
    localStorage.setItem("forecast_ai_insights", JSON.stringify(insights));
  }, [insights]);

  useEffect(() => {
    const runForecastCheck = async () => {
      try {
        const res = await axios.post("http://localhost:8000/forecast/check", {
          datasetId: "default",
        });
        console.log("Forecast Check:", res.data);
        if (res.data.ok && res.data.ds && res.data.y) {
          setForecastCheckResult(res.data);
          setForecastError("");
        } else {
          setForecastCheckResult(null);
          setForecastError("No suitable columns for forecasting.");
        }
      } catch (err) {
        console.error("Forecast check failed:", err);
        setForecastCheckResult(null);
        setForecastError("Forecast check failed.");
      }
    };

    if (uploaded) runForecastCheck();
  }, [uploaded]);

  const onTrain = async () => {
    setStatus("Training (auto-selecting best model)...");
    try {
      const meta = await forecastTrain({ datasetId, dateCol: "ds", targetCol: "y", horizon, folds: 3 });
      setModelInfo(meta);
      setStatus(`Trained. Chosen: \${meta.chosen.toUpperCase()} | Freq: \${meta.freq}`);
      const hist = await forecastPlot(datasetId, "history");
      const fimg = await forecastPlot(datasetId, "forecast");
      const dec = await forecastPlot(datasetId, "decomposition");
      setImgHistory(hist);
      setImgForecast(fimg);
      setImgDecomp(dec);
    } catch (e) {
      setStatus(String(e));
    }
  };

  const onPredict = async () => {
    setStatus("Predicting...");
    try {
      const res = await forecastPredict(datasetId, horizon);
      setStatus(`Predicted \${res.forecast?.length || 0} steps.`);
      const fimg = await forecastPlot(datasetId, "forecast");
      setImgForecast(fimg);
    } catch (e) {
      setStatus(String(e));
    }
  };

  const onInsights = async () => {
    setStatus("Generating insights...");
    setInsights(null);
    try {
      const res = await forecastInsights(datasetId);
      console.log("AI Insight response:", res);
      setInsights(res);
      setStatus(" Insights ready.");
    } catch (e) {
      console.error("Insight error:", e);
      setStatus("❌ Failed to generate insights.");
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {uploaded ? (
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 240 }}>
            {forecastError ? (
              <div style={{ fontSize: 13, color: "red" }}>{forecastError}</div>
            ) : (
              <>
                <label style={{ fontSize: 13, fontWeight: 600 }}>Train Model</label>
                <input
                  type="number"
                  value={horizon}
                  onChange={(e) => setHorizon(parseInt(e.target.value || "30"))}
                  placeholder="Forecast Horizon"
                  style={{ width: "100%", marginTop: 4, padding: 8, borderRadius: 8, border: "1px solid #ccc", marginRight: 14 }}
                />
                <button onClick={onTrain} style={{ marginTop: 8, padding: "8px 12px", borderRadius: 8, border: "1px solid #ddd" }}>
                  Auto-Train
                </button>
              </>
            )}
          </div>
        </div>
      ) : (
        <div style={{ fontSize: 13, color: "#6b7280" }}>Upload a CSV file first to begin forecasting.</div>
      )}

      <div style={{ fontSize: 13, color: "#374151" }}>{status}</div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
        <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#fff" }}>
          <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>History</div>
          {imgHistory ? <img src={imgHistory} alt="history" style={{ width: "100%", borderRadius: 8 }} /> : <div style={{ fontSize: 12, color: "#9ca3af" }}>No image</div>}
        </div>
        <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#fff" }}>
          <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>Forecast</div>
          {imgForecast ? <img src={imgForecast} alt="forecast" style={{ width: "100%", borderRadius: 8 }} /> : <div style={{ fontSize: 12, color: "#9ca3af" }}>No image</div>}
        </div>
        <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#fff" }}>
          <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>Decomposition</div>
          {imgDecomp ? <img src={imgDecomp} alt="decomp" style={{ width: "100%", borderRadius: 8 }} /> : <div style={{ fontSize: 12, color: "#9ca3af" }}>No image</div>}
        </div>
      </div>

      <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 16, background: "#fff" }}>
        <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>AI Insights</div>
        <div style={{ display: "flex", gap: 8, marginTop: 4, marginBottom: 18 }}>
          <button onClick={onInsights} style={{ padding: "8px 12px", borderRadius: 8, border: "1px solid #ddd" }}>Generate Insights</button>
        </div>
        {insights?.ai ? (
          <div className="text-xs text-gray-400 leading-relaxed whitespace-pre-line mt-2">
            {(Array.isArray(insights.ai) ? insights.ai : (insights.ai || "").split("\n"))
              .map((s, i) => s.replace(/^\s*[-•]\s*/, "").trim())
              .filter(Boolean)
              .map((s, i) => <p key={i}>{s}</p>)}
          </div>
        ) : (
          <div style={{ fontSize: 13, color: "#6b7280" }}>
            Click Generate Insights to run the model.
          </div>
        )}
      </div>
    </div>
  );
}
