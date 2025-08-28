import React, { useState } from "react";
import KPIPanel from "./KPIPanel";
import ComparePairsPanel from "./ComparePairs";
import EDATab from "./EDATab";
import ForecastPanel from "./ForecastPanel";
import CausalInsights from "./causal_insights"; // placeholder for Overall Insights later
import OutliersPanel from "./OutliersPanel";

// Colors (flat & minimal)
const COLORS = {
  primary: "#2e86de",
  primaryLight: "#87a4e4ff",
  
  card: "#ffffff",
  border: "#e6ebf2",
  text: "#233044",
  subtext: "#5b6b82",
};

export default function HistoricalWithKPI({ uploaded }) {
  const [tab, setTab] = useState("historical");

  const TabButton = ({ id, label }) => {
    const active = tab === id;
    return (
      <button
        onClick={() => setTab(id)}
        style={{
          height: 32,
          padding: "0 14px",
          border: "none",
          background: active ? COLORS.primary : "#ffffff",
          color: active ? "#fff" : "#1f2937",
          fontWeight: 600,
          borderRadius: 32,
          cursor: "pointer",
        }}
      >
        {label}
      </button>
    );
  };

  return (
    <div style={{ background: COLORS.bg, minHeight: "100%", padding: 0 }}>
      {/* KPI Section */}
      <div
        style={{
          
          overflow: "hidden",
          marginBottom: 16,
          background: "transparent",
          
        }}
      >
        {/* <div
          style={{
            background: "#f3f7ffff",
            color: "#374552ff",
            padding: "10px 14px",
            fontWeight: 700,
            letterSpacing: 0.2,
          }}
        >
          KPIs
        </div> */}
        <div style={{ padding: 0 }}>
          <KPIPanel uploaded={uploaded} />
        </div>
      </div>

      {/* Analysis Tabs */}
      
        {/* Tab header */}
        <div
          style={{
            display: "flex",
            gap: 8,
            borderBottom: `1px solid ${COLORS.border}`,

            padding: "8px 12px",
            // background: COLORS.primaryLight,
          }}
        >
          <TabButton id="historical" label="Historical" />
          <TabButton id="eda" label="EDA – ydata Profiling" />
          <TabButton id="forecast" label="Forecast" />
          <TabButton id="outliers" label="Outliers" />
          <TabButton id="insights" label="Overall Insights" />
          <TabButton id="Report" label="Report" />
          


        </div>
        <div
        style={{
          border: `1px solid ${COLORS.border}`,
          borderRadius: 14,
          boxShadow: "0 1px 3px rgba(2,6,23,0.06)",
          background: COLORS.card,
          marginTop:2
        }}
      >
        {/* Tab content */}
        <div style={{ padding: 16 }}>
          {tab === "historical" && <ComparePairsPanel uploaded={uploaded} />}
          {tab === "eda" && <EDATab uploaded={uploaded} />}
          {tab === "forecast" && <ForecastPanel uploaded={uploaded} />}
          {tab === "outliers" && <OutliersPanel datasetId="default" uploaded={uploaded} />}
          {tab === "insights" && (
            <div style={{ color: COLORS.subtext, fontSize: 14, lineHeight: 1.6 }}>
              🚧 Overall Insights will be added here later.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
