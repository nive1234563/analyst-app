// import React, { useMemo, useState, useEffect ,useCallback } from "react";
// import CausalTab from "./components/CausalTab";
// import EDATab from "./components/EDATab";
// import CausalInsights from "./components/causal_insights";
// import ForecastPanel from "./components/ForecastPanel";
// import SlicesPanel from "./components/slicespanel";
// import AIDatasetInsights from "./components/AIDatasetInsights";
// import ComparePairsPanel from "./components/ComparePairs";
// import NatTab from "./components/NatTab";
// import AggregationChartsPanel from "./components/aggregationCharts";
// import KPIPanel from "./components/KPIPanel";
// import HistoricalWithKPI from "./components/HistoricalWithKPI";
// import OutliersPanel from "./components/OutliersPanel";


// /** ——— Minimal inline styles (no extra deps) ——— */
// const css = {
//   app: {
//     minHeight: "100vh",
//     display: "flex",
//     background: "#e9e9e9ff",
//     color: "#0f172a",
//     fontFamily:
//       "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'",
//   },
//   sidebar: {
//     width: 260,
//     background: "#0b1220",
//     color: "#e5e7eb",
//     display: "flex",
//     flexDirection: "column",
//     borderRight: "1px solid #101827"
//   },
//   brand: {
//     padding: "18px 16px",
//     fontSize: 16,
//     fontWeight: 700,
//     letterSpacing: 0.4,
//     display: "flex",
//     alignItems: "center",
//     gap: 10,
//     borderBottom: "1px solid #111827"
//   },
//   brandLogo: {
//     width: 20,
//     height: 20,
//     borderRadius: 6,
//     background:
//       "linear-gradient(135deg, rgba(99,102,241,.9), rgba(59,130,246,.9))"
//   },
//   searchWrap: { padding: "12px 12px 6px" },
//   search: {
//     width: "100%",
//     padding: "10px 12px",
//     borderRadius: 10,
//     border: "1px solid #1f2937",
//     outline: "none",
//     background: "#0f172a",
//     color: "#e5e7eb",
//     fontSize: 13
//   },
//   tabs: { padding: 8, overflowY: "auto" },
//   tabBtn: (active) => ({
//     width: "100%",
//     padding: "10px 12px",
//     marginBottom: 6,
//     borderRadius: 12,
//     textAlign: "left",
//     border: "1px solid " + (active ? "#334155" : "#111827"),
//     background: active ? "rgba(59,130,246,.14)" : "transparent",
//     color: active ? "#dbeafe" : "#e5e7eb",
//     cursor: "pointer",
//     fontSize: 13.5,
//     transition: "all .12s ease"
//   }),
//   main: { flex: 1, display: "flex", flexDirection: "column" },
//   topbar: {
//     height: 56,
//     display: "flex",
//     alignItems: "center",
//     justifyContent: "space-between",
//     padding: "0 16px",
//     borderBottom: "1px solid #e5e7eb",
//     background: "#ffffffcc",
//     backdropFilter: "saturate(180%) blur(6px)",
//     position: "sticky",
//     top: 0,
//     zIndex: 5,
//   },
//   title: { fontSize: 16, fontWeight: 700 },
//   rightTools: { display: "flex", gap: 8, alignItems: "center" },
//   ghostBtn: {
//     padding: "8px 12px",
//     borderRadius: 10,
//     border: "1px solid #e5e7eb",
//     background: "white",
//     cursor: "pointer",
//     fontSize: 13,
//   },
//   content: {
//     padding: 16,
//     display: "grid",
//     gridTemplateColumns: "minmax(0, 1fr)",
//     gap: 16,
//   },
//   card: {
//     background: "white",
//     border: "1px solid #e5e7eb",
//     borderRadius: 16,
//     padding: 16,
//     boxShadow: "0 1px 0 rgba(0,0,0,.02)",
//   },
//   hgroup: { display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 },
//   subttl: { fontSize: 14, fontWeight: 700, color: "#0f172a" },
//   subtle: { color: "#6b7280", fontSize: 12.5 },
//   row: { display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" },
//   pill: (active) => ({
//     padding: "8px 12px",
//     borderRadius: 999,
//     border: "1px solid " + (active ? "#93c5fd" : "#e5e7eb"),
//     background: active ? "#eff6ff" : "white",
//     cursor: "pointer",
//     fontSize: 12.5,
//   }),
//   media: {
//     width: "100%",
//     borderRadius: 12,
//     border: "1px solid #e5e7eb",
//   },
//   iframeWrap: { width: "100%", height: "80vh", border: "1px solid #e5e7eb", borderRadius: 12, overflow: "hidden" },
//   iframe: { width: "100%", height: "100%", border: "0" },
//   editor: {
//     width: "100%",
//     minHeight: 220,
//     padding: 12,
//     borderRadius: 12,
//     border: "1px solid #e5e7eb",
//     outline: "none",
//     fontSize: 14,
//     lineHeight: 1.5,
//     background: "#fff",
//     whiteSpace: "pre-wrap",
//   },
//   saveBar: { display: "flex", gap: 8, justifyContent: "flex-end" },
// };

// /** ——— Tabs ——— */
// const TAB_LIST = [
//   "KPIs + Historical",
//   "Historical Analysis",
//   "KPIs",
//   "EDA",
//   "Generate Prompts - study",
//   "Outliers",
//   "Health Check",
//   "Causal & Uplift(NA)", 
//   "Historical Analysis(NA)",
//   "Forecast",
// ];

// export default function App() {
//   const [active, setActive] = useState(0);
//   const [q, setQ] = useState("");

//   /** Simple text content for other tabs (editable) */
//   const initialPages = useMemo(
//     () =>
//       TAB_LIST.map((name, i) =>
//         i <= 1
//           ? "" // Forecast & EDA are functional
//           : `# ${name}\n\nThis page is fully editable.\n\n- Double-check content\n- Add notes, to-dos\n- Paste charts/screenshots links\n\nUse the **Edit** button (top right) to modify, then Save.`,
//       ),
//     []
//   );
//   const [pages, setPages] = useState(initialPages);
//   const [editing, setEditing] = useState(false);
//   const [draft, setDraft] = useState(pages[active]);

//   /** Forecast tab state (unchanged) */
//   const [file, setFile] = useState(null);
//   const [uploaded, setUploaded] = useState(false);
//   const [plotTab, setPlotTab] = useState("trend"); // trend | weekly | yearly
//   const plotURL =
//     plotTab === "trend"
//       ? "http://localhost:8000/plot/trend"
//       : plotTab === "weekly"
//       ? "http://localhost:8000/plot/weekly"
//       : "http://localhost:8000/plot/yearly";

//   /** EDA tab state */
//   const [edaLoading, setEdaLoading] = useState(false);
//   const [edaError, setEdaError] = useState("");
//   const [reportUrl, setReportUrl] = useState("");
//   const [engine, setEngine] = useState("ydata"); // ydata | sweetviz
//   const [edaKpis, setEdaKpis] = useState(null);
//   const [kpiLoading, setKpiLoading] = useState(false);
//   const [kpiError, setKpiError] = useState("");

//   const fetchKpis = async () => {
//   try {
//     setKpiError("");
//     setKpiLoading(true);

//     const res = await fetch("http://localhost:8000/eda/profile/ydata/kpis", {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json"
//       },
//       body: JSON.stringify({
//         dataset_id: "default",
//         title: "EDA Report"
//       })
//     });

//     if (!res.ok) {
//       const errorText = await res.text();
//       throw new Error(`Server responded with ${res.status}: ${errorText}`);
//     }

//     const data = await res.json();
//     setEdaKpis(data);
//   } catch (err) {
//     console.error("KPI fetch failed:", err);
//     setKpiError("❌ Failed to fetch KPIs. Please check backend and try again.");
//   } finally {
//     setKpiLoading(false);
//   }
// };








//   const handleFileUpload = async (e) => {
//     const f = e.target.files?.[0];
//     setFile(f || null);
//     setReportUrl("");
//     if (!f) return;

//     const formData = new FormData();
//     formData.append("file", f);

//     try {
//       const res = await fetch("http://localhost:8000/upload-csv/", {
//         method: "POST",
//         body: formData,
//       });
//       if (!res.ok) throw new Error("Upload failed");
//       setUploaded(true);
//     } catch (err) {
//       console.error(err);
//       setUploaded(false);
//       alert("Upload failed. Is the backend running at :8000 ?");
//     }
//   };

//   const generateEda = async (which) => {
//     if (!uploaded) {
//       setEdaError("Upload a CSV first.");
//       return;
//     }
//     setEdaError("");
//     setEdaLoading(true);
//     setEngine(which);

//     try {
//       const res = await fetch(`http://localhost:8000/eda/profile/${which}`, {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ dataset_id: "default", title: "EDA Report" }),
//       });
//       if (!res.ok) throw new Error(await res.text());
//       const data = await res.json();
//       setReportUrl(`http://localhost:8000${data.report_url}?t=${Date.now()}`);
//     } catch (err) {
//       console.error(err);
//       setEdaError("Failed generating report. Check backend logs.");
//     } finally {
//       setEdaLoading(false);
//     }
//   };

// //   useEffect(() => {
// //   const load = async () => {
// //     if (active !== 1 || !uploaded || !file) return;  // <- added !file check
// //     setKpiLoading(true);
// //     setKpiError("");

// //     try {
// //       const res = await fetch("http://localhost:8000/eda/profile/ydata/kpis", {
// //         method: "POST",
// //         headers: { "Content-Type": "application/json" },
// //         body: JSON.stringify({ dataset_id: "default", title: "EDA KPIs" })
// //       });
// //       if (!res.ok) throw new Error(await res.text());
// //       const data = await res.json();
// //       setEdaKpis(data);
// //     } catch (e) {
// //       setKpiError(typeof e === "string" ? e : e.message || "Failed to load KPIs");
// //     } finally {
// //       setKpiLoading(false);
// //     }
// //   };

// //   load();
// // }, [active, uploaded, file]);

  



//   /** Tab switch */
//   const switchTab = (idx) => {
//     if (active > 1 && editing) {
//       const next = [...pages];
//       next[active] = draft;
//       setPages(next);
//       setEditing(false);
//     }
//     setActive(idx);
//     setDraft(pages[idx]);
//   };

//   /** Save/Cancel for text tabs */
//   const onSave = () => {
//     const next = [...pages];
//     next[active] = draft;
//     setPages(next);
//     setEditing(false);
//   };
//   const onCancel = () => {
//     setDraft(pages[active]);
//     setEditing(false);
//   };

//   /** Filter tabs by search */
//   const filteredIdx = TAB_LIST.map((t, i) => ({ i, t }))
//     .filter(({ t }) => t.toLowerCase().includes(q.toLowerCase()));

//   return (
//     <div style={css.app}>
//       {/* Sidebar */}
//       <aside style={css.sidebar}>
//         <div style={css.brand}>
//           <div style={css.brandLogo} />
//           Analyst
//         </div>

        

//         <div style={css.tabs}>
//           {filteredIdx.map(({ i, t }) => (
//             <button
//               key={t}
//               style={css.tabBtn(active === i)}
//               onClick={() => switchTab(i)}
//             >
//               {/* {String(i + 1).padStart(2, "0")} • {t} */}
//               {t}
//             </button>
//           ))}
//         </div>
//       </aside>




// {/* Main */}
   
// <main style={css.main}>
//   {/* Topbar */}
//   <div style={css.topbar}>
//     <div style={css.title}>{TAB_LIST[active]}</div>

//     <div style={css.rightTools}>
      
//         <>
//           <label style={css.ghostBtn}>
//             Upload CSV
//             <input
//               type="file"
//               accept=".csv"
//               onChange={handleFileUpload}
//               style={{ display: "none" }}
//             />
//           </label>
//           {file ? (
//             <span style={css.subtle}>Selected: {file.name}</span>
//           ) : (
//             <span style={css.subtle}>No file selected</span>
//           )}

//           {active === 1 && (
//             <button
//               style={{ ...css.ghostBtn, opacity: uploaded ? 1 : 0.5 }}
//               disabled={!uploaded || edaLoading}
//               onClick={() => generateEda("ydata")}
//             >
//               {edaLoading && engine === "ydata" ? "Generating…" : "Generate ydata"}
//             </button>
//           )}
//         </>
      
//     </div>
//   </div>

//   {/* Content */}
//   <div style={css.content}>
    

//     {/* Forecast tab */}
//     {active === 9 && (
//       <section style={css.card}>
//         <ForecastPanel uploaded={uploaded} />
//       </section>
//     )}


//     {/* EDA tab */}
//     {active === 3 && (
//       <EDATab
//         uploaded={uploaded}
//         edaKpis={edaKpis}
//         kpiLoading={kpiLoading}
//         kpiError={kpiError}
//         edaError={edaError}
//         reportUrl={reportUrl}
//         handleGenerateEda={() => generateEda("ydata")}
//         handleLoadKpis={fetchKpis}
//         css={css}
//       />
//     )}


//     {/* Causal tab */}
//     {active === 7 && (
//       <section style={css.card}>
//         <div style={css.hgroup}>
//           <div>
//             <div style={css.subttl}>Auto Causal Analysis</div>

//           </div>
//         </div>
//         <CausalTab datasetId="default" outcome="SALES" />

//         <div style={{ height: 12 }} />
//         <CausalInsights />
//       </section>
//     )}

//     {active === 8 && <SlicesPanel uploaded={uploaded} />}

//      {/* Insights tab (AI dataset insights) */}
//     {active === 6 && (
//       <AIDatasetInsights uploaded={uploaded} />
//       )}

//     {/* // historical */}
//     {active === 1 && (   
//       <ComparePairsPanel uploaded={uploaded} />  
//       )} 

//     {active === 4 && (
//       <NatTab datasetId="default"/>
//       )}
    
//     {active === 5 && (
//       <OutliersPanel datasetId="default" uploaded={uploaded} />
//       )}

//     {active === 2 && (
//       <KPIPanel datasetId="default" uploaded={uploaded}/>
//       )}

//     {/* Editable content tabs */}
//     {active ===0 && (
//       <HistoricalWithKPI datasetId="default" uploaded={uploaded}/>
//     )}
//   </div>
// </main>

// </div>
    
//   );
// }

import React, { useRef, useState, useEffect } from "react";
import CausalTab from "./components/CausalTab";
import EDATab from "./components/EDATab";
import CausalInsights from "./components/causal_insights";
import ForecastPanel from "./components/ForecastPanel";
import SlicesPanel from "./components/slicespanel";
import AIDatasetInsights from "./components/AIDatasetInsights";
import ComparePairsPanel from "./components/ComparePairs";
import NatTab from "./components/NatTab";
import AggregationChartsPanel from "./components/aggregationCharts";
import KPIPanel from "./components/KPIPanel";
import HistoricalWithKPI from "./components/HistoricalWithKPI";
import OutliersPanel from "./components/OutliersPanel";

// Icons
import { BarChart3, MessageSquare, User, Settings, Bot, X, Send } from "lucide-react";

/** ——— Light mode with fixed vertical navbar + Chat FAB ——— */
const css = {
  app: {
    minHeight: "100vh",
    background: "#f1f4f8ff",
    color: "#0f172a",
    fontFamily:
      "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans'",
    paddingLeft: 80, // space for fixed sidebar
    display: "flex",
    flexDirection: "column",
  },

  // Fixed icon-only sidebar
  vbar: {
    width: 80,
    background: "#ffffff",
    borderRight: "1px solid #e5e7eb",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "12px 0",
    gap: 12,
    position: "fixed",
    left: 0,
    top: 0,
    height: "100vh",
    zIndex: 20,
  },

  vbtn: (active) => ({
    width: 44,
    height: 44,
    borderRadius: 12,
    display: "grid",
    placeItems: "center",
    border: active ? "1px solid #bfd4ff" : "1px solid transparent",
    background: active ? "#eef4ff" : "transparent",
    cursor: "pointer",
    color: active ? "#274690" : "#4b5563",
    transition: "all .12s",
  }),
  vspacer: { flex: 1 },

  main: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
  },

  topbar: {
    height: 56,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "0 16px",
    borderBottom: "1px solid #e5e7eb",
    background: "#f9fcfccc",
    backdropFilter: "saturate(180%) blur(6px)",
    position: "sticky",
    top: 0,
    zIndex: 5,
  },
  title: { fontSize: 16, fontWeight: 700 },
  rightTools: { display: "flex", gap: 8, alignItems: "center" },
  ghostBtn: {
    padding: "8px 12px",
    borderRadius: 10,
    border: "1px solid #e5e7eb",
    background: "white",
    cursor: "pointer",
    fontSize: 13,
  },
  subtle: { color: "#6b7280", fontSize: 12.5 },

  content: {
    padding: 16,
    display: "grid",
    gridTemplateColumns: "minmax(0, 1fr)",
    gap: 16,
  },
  card: {
    background: "white",
    border: "1px solid #e5e7eb",
    borderRadius: 16,
    padding: 16,
    boxShadow: "0 1px 0 rgba(0,0,0,.02)",
  },

  // Logo image in public/
  logoImg: {
    width: 32,
    height: 32,
    objectFit: "contain",
    margin: "6px 0 10px",
    borderRadius: 8,
    
    background: "#fff",
  },

  // Chat floating button
  chatFab: {
    position: "fixed",
    right: 50,
    bottom: 50,
    width: 70,
    height: 70,
    borderRadius: 16, // rounded square
    display: "grid",
    placeItems: "center",
    background: "linear-gradient(to right, #3369ff, #6f33ff)",
    color: "white",
    border: "1px solid rgba(0,0,0,.05)",
    boxShadow: "0 10px 20px rgba(0,0,0,.15)",
    cursor: "pointer",
    zIndex: 30,
  },

  // Chat panel (slide-over)
  chatPanelWrap: (open) => ({
    position: "fixed",
    inset: 0,
    zIndex: 40,
    pointerEvents: open ? "auto" : "none",
  }),
  chatBackdrop: (open) => ({
    position: "absolute",
    inset: 0,
    background: "rgba(15,23,42,.28)",
    opacity: open ? 1 : 0,
    transition: "opacity .18s ease",
  }),
  chatPanel: (open) => ({
    position: "absolute",
    right: 0,
    top: 0,
    height: "100vh",
    width: "min(420px, 92vw)",
    background: "#fff",
    borderLeft: "1px solid #e5e7eb",
    boxShadow: "0 10px 30px rgba(0,0,0,.18)",
    transform: open ? "translateX(0)" : "translateX(105%)",
    transition: "transform .22s ease",
    display: "flex",
    flexDirection: "column",
  }),
  chatHeader: {
    height: 56,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "0 12px 0 14px",
    borderBottom: "1px solid #e5e7eb",
    background: "#ffffff",
  },
  chip: {
    display: "inline-flex",
    alignItems: "center",
    
    padding: "6px 10px",
    borderRadius: 999,
    
    
    fontSize: 14,
    color: "#0f172a",
  },
  chatBody: {
    flex: 1,
    overflowY: "auto",
    padding: 12,
    background: "#f4f5f7ff",
  },
  msgRow: { display: "flex", marginBottom: 8 },
  msgYou: {
    marginLeft: "auto",
    maxWidth: "85%",
    background: "#e8efff",
    border: "1px solid #bfd4ff",
    color: "#0f172a",
    padding: "10px 12px",
    borderRadius: 12,
  },
  msgBot: {
    marginRight: "auto",
    maxWidth: "85%",
    background: "white",
    border: "1px solid #e5e7eb",
    color: "#0f172a",
    padding: "10px 12px",
    borderRadius: 12,
  },
  chatInputBar: {
    display: "flex",
    gap: 8,
    padding: 10,
    borderTop: "1px solid #e5e7eb",
    background: "#fff",
  },
  chatInput: {
    flex: 1,
    height: 40,
    borderRadius: 10,
    border: "1px solid #e5e7eb",
    padding: "0 12px",
    outline: "none",
    fontSize: 14,
    background: "#fff",
  },
  chatSend: {
    width: 44,
    height: 40,
    borderRadius: 10,
    border: "1px solid #e5e7eb",
    background: "#4b71daff",
    color: "white",
    display: "grid",
    placeItems: "center",
    cursor: "pointer",
  },
};

// Icon-only items (titles show native tooltips)
const NAV_ITEMS = [
  { key: "insights", title: "Insights", icon: <BarChart3 size={20} /> },
  { key: "history", title: "Chat History", icon: <MessageSquare size={20} /> },
];

export default function App() {
  const [active, setActive] = useState("insights");
  const [file, setFile] = useState(null);
  const [uploaded, setUploaded] = useState(false);

  // Chat state
  const [chatOpen, setChatOpen] = useState(false);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hi! Ask me anything about your data or forecasts." },
  ]);
  const bodyRef = useRef(null);

  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [messages, chatOpen]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || sending) return;

    // Push user message
    setMessages((m) => [...m, { role: "user", content: text }]);
    setInput("");
    setSending(true);

    try {
      // Adjust endpoint to your backend if different
      const res = await fetch("http://localhost:8000/ai/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      if (!res.ok) {
        throw new Error(`Server ${res.status}`);
      }

      const data = await res.json(); // expect {reply: "..."}
      const reply = data?.reply ?? "Okay!";
      setMessages((m) => [...m, { role: "assistant", content: reply }]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content:
            "⚠️ I couldn't reach the AI service. Please check the backend at /ai/chat.",
        },
      ]);
    } finally {
      setSending(false);
    }
  };

  const handleFileUpload = async (e) => {
    const f = e.target.files?.[0];
    setFile(f || null);
    if (!f) return;

    const formData = new FormData();
    formData.append("file", f);
    try {
      const res = await fetch("http://localhost:8000/upload-csv/", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Upload failed");
      setUploaded(true);
    } catch (err) {
      console.error(err);
      setUploaded(false);
      alert("Upload failed. Is the backend running at :8000 ?");
    }
  };

  return (
    <div style={css.app}>
      {/* Fixed vertical icon navbar */}
      <aside style={css.vbar}>
        {/* Logo from public/ */}
        <img src="/logo192.png" alt="Logo" style={css.logoImg} />

        {NAV_ITEMS.map((n) => (
          <button
            key={n.key}
            style={css.vbtn(active === n.key)}
            onClick={() => setActive(n.key)}
            title={n.title}
            aria-label={n.title}
          >
            {n.icon}
          </button>
        ))}
        <div style={css.vspacer} />
        <button style={css.vbtn(false)} title="Profile" aria-label="Profile">
          <User size={20} />
        </button>
        <button style={css.vbtn(false)} title="Settings" aria-label="Settings">
          <Settings size={20} />
        </button>
      </aside>

      {/* Main content */}
      <main style={css.main}>
        <div style={css.topbar}>
          <div style={css.title}>
            {active === "insights" ? "Insights" : "Chat History"}
          </div>
          <div style={css.rightTools}>
            <label style={css.ghostBtn}>
              Upload CSV
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                style={{ display: "none" }}
              />
            </label>
            {file ? (
              <span style={css.subtle}>No file selected</span>
            ) : (
              <span style={css.subtle}>No file selected</span>
            )}
          </div>
        </div>

        <div style={css.content}>
          {active === "insights" && (
            <HistoricalWithKPI datasetId="default" uploaded={uploaded} />
          )}

          {active === "history" && (
            <div style={css.card}>
              <h3 style={{ margin: 0 }}>Chat History</h3>
              <p style={{ marginTop: 8, color: "#6b7280" }}>
                Build this section to list previous chats or saved analyses.
              </p>
            </div>
          )}
        </div>
      </main>

      {/* Floating Chat Button */}
      <button style={css.chatFab} onClick={() => setChatOpen(true)} title="AI Chat">
        <img src="/bot.png" alt="AI Bot Icon" style={{ filter: "brightness(0) invert(1)", width: 36, height: 36 }} />
      </button>

      {/* Slide-over Chat Panel */}
      <div style={css.chatPanelWrap(chatOpen)}>
        <div
          style={css.chatBackdrop(chatOpen)}
          onClick={() => setChatOpen(false)}
        />
        <div style={css.chatPanel(chatOpen)} role="dialog" aria-modal="true">
          <div style={css.chatHeader}>
            <div style={css.chip}>
              <b> AI Assistant </b>
            </div>
            <button
              onClick={() => setChatOpen(false)}
              style={{
                width: 32,
                height: 32,
                borderRadius: 8,
                border: "1px solid #e5e7eb",
                background: "#fff",
                display: "grid",
                placeItems: "center",
                cursor: "pointer",
              }}
              aria-label="Close chat"
              title="Close"
            >
              <X size={16} />
            </button>
          </div>

          <div style={css.chatBody} ref={bodyRef}>
            {messages.map((m, idx) => (
              <div key={idx} style={css.msgRow}>
                <div style={m.role === "user" ? css.msgYou : css.msgBot}>
                  {m.content}
                </div>
              </div>
            ))}
          </div>

          <div style={css.chatInputBar}>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask something…"
              style={css.chatInput}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSend();
              }}
            />
            <button
              style={{
                ...css.chatSend,
                opacity: sending ? 0.6 : 1,
                pointerEvents: sending ? "none" : "auto",
              }}
              onClick={handleSend}
              title="Send"
              aria-label="Send"
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

