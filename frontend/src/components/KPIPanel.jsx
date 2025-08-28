// // KPIPanel.jsx
// import React, { useRef, useState } from "react";

// /** ---------- Minimal Design Tokens ---------- */
// const c = {
//   bg: "#ffffffff",
//   panel: "#ffffff",
//   border: "#e6ebf2",
//   text: "#0f172a",
//   subtext: "#64748b",
//   brand: "#3866f2",
//   brandSoft: "#eef3ff",
//   successBg: "#e9f7ee",
//   successText: "#15803d",
// };

// const Card = ({ children, style }) => (
//   <div
//     style={{
//       background: c.panel,
//     //   border: `1px solid ${c.border}`,
//       borderRadius: 12,
//       fontcolor:"#282e36ff",
//       // flatter look: barely-there shadow
//       boxShadow: "0 2px 4px rgba(77, 89, 142, 0.14)",
//       padding: 16,
//       ...style,
//     }}
//   >
//     {children}
//   </div>
// );

// const Button = ({ children, onClick, disabled, variant = "primary" }) => {
//   const base = {
//     display: "inline-flex",
//     alignItems: "center",
//     justifyContent: "center",
//     height: 32,
//     minWidth: 140,
//     padding: "0 14px",
//     borderRadius: 10,
//     fontWeight: 600,
//     fontSize: 12,
//     cursor: disabled ? "not-allowed" : "pointer",
//     transition: "background .15s ease, border-color .15s ease",
//     opacity: disabled ? 0.7 : 1,
//   };
//   const styles =
//     variant === "primary"
//       ? {
//           border: `1px solid ${c.brand}`,
//           background: c.brandSoft,
//           color: c.brand,
//         }
//       : {
//           border: `1px solid ${c.border}`,
//           background: "#fff",
//           color: c.text,
//         };

//   return (
//     <button
//       onClick={disabled ? undefined : onClick}
//       disabled={disabled}
//       style={{ ...base, ...styles }}
//       onMouseEnter={(e) => {
//         if (disabled) return;
//         e.currentTarget.style.background =
//           variant === "primary" ? "#e6edff" : "#f8fafc";
//       }}
//       onMouseLeave={(e) => {
//         e.currentTarget.style.background =
//           variant === "primary" ? c.brandSoft : "#fff";
//       }}
//     >
//       {children}
//     </button>
//   );
// };

// /** ---------- Page ---------- */
// export default function KPIPanel() {
//   const [loading, setLoading] = useState(false);
//   const [aggs, setAggs] = useState([]);
//   const [error, setError] = useState("");
//   const [debug, setDebug] = useState([]);
//   const [rowsSetInfo, setRowsSetInfo] = useState(null);

//   // reliable file input trigger
//   const fileRef = useRef(null);
//   const openFilePicker = () => fileRef.current?.click();

//   const uploadCSV = async (file) => {
//     if (!file) return;
//     const fd = new FormData();
//     fd.append("file", file);
//     setError("");
//     setRowsSetInfo(null);
//     try {
//       const res = await fetch("http://localhost:8000/kpi/upload-csv", {
//         method: "POST",
//         body: fd,
//       });
//       const body = await res.json();
//       if (!body.ok) throw new Error(body.error || "Upload failed");
//       setRowsSetInfo({ rows: body.rows, cols: body.cols });
//     } catch (e) {
//       setError(e.message);
//     }
//   };

//   const compute = async () => {
//     setLoading(true);
//     setError("");
//     setAggs([]);
//     setDebug([]);
//     try {
//       const res = await fetch("http://localhost:8000/kpi/start", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({}),
//       });
//       const body = await res.json();
//       setAggs(body.aggregations || []);
//       setDebug(body.kpi_debug || []);
//       if (body.error) setError(body.error);
//     } catch (e) {
//       setError("Failed to compute KPIs");
//     } finally {
//       setLoading(false);
//     }
//   };

// //   const gradients = [
// //     "linear-gradient(180deg, #f6f9ff 0%, #ffffff 100%)",
// //     "linear-gradient(180deg, #f5fffd 0%, #ffffff 100%)",
// //     "linear-gradient(180deg, #fffaf3 0%, #ffffff 100%)",
// //     "linear-gradient(180deg, #faf5ff 0%, #ffffff 100%)",
// //   ];

//   // Replace your gradients array with this:
// const gradients = [
  
//    // orange-50,
//    "#eaf0ffff", // slate-50
//   "#eaf0ffff", // yellow-50
//   "#eaf0ffff", // green-50
//   "#eaf0ffff", // indigo-50
//   "#f0f4ffff", // pink-50
//   "#f0f4ffff", // teal-50
//   "#f0f4ffff", // neutral-50
//   "#f0f4ffff",
// ];


//   return (
//     <div style={{ background: c.bg, minHeight: "100%", padding: 18 }}>
//       {/* Hidden file input for robust upload */}
//       <input
//         ref={fileRef}
//         type="file"
//         accept=".csv,.xlsx"
//         style={{ display: "none" }}
//         onChange={(e) => uploadCSV(e.target.files?.[0] || null)}
//       />

//       {/* Header / Controls */}
//       <Card>
//         <div
//           style={{
//             display: "flex",
//             alignItems: "center",
//             gap: 10,
//             flexWrap: "wrap",
//           }}
//         >
//           <div style={{ fontSize: 18, fontWeight: 500, color: c.text }}>
//             Load dataset
//           </div>
          
//           <Button onClick={openFilePicker}>Upload CSV</Button>
//           <Button onClick={compute} disabled={loading} variant="secondary">
//             {loading ? "Computing KPIs…" : "Compute KPIs"}
//           </Button>

//           {error && (
//             <span style={{ color: "#b42318", fontWeight: 600 }}>{error}</span>
//           )}
//         </div>

//         {/* small summary */}
//         <div style={{ marginTop: 10, color: c.subtext, fontSize: 13 }}>
//           {rowsSetInfo ? (
//             <>
//               <b style={{ color: c.text }}>Loaded rows:</b> {rowsSetInfo.rows}
//               <div
//                 style={{
//                   marginTop: 6,
//                   display: "flex",
//                   gap: 6,
//                   flexWrap: "wrap",
//                 }}
//               >
//                 <b style={{ color: c.text }}>Columns:</b>
//                 {rowsSetInfo.cols.map((col, i) => (
//                   <span
//                     key={i}
//                     style={{
//                       background: "#f3f6fb",
//                       border: `1px solid ${c.border}`,
//                       color: "#334155",
//                       padding: "2px 8px",
//                       borderRadius: 999,
//                       fontSize:10,
//                     }}
//                   >
//                     {String(col)}
//                   </span>
//                 ))}
//               </div>
//             </>
//           ) : (
//             <>No dataset loaded. Upload a File to begin.</>
//           )}
//         </div>
//       </Card>

//       {/* KPI Tiles */}
//       <div style={{ marginTop: 14 }}>
//         <div
//           style={{
//             display: "grid",
//             gridTemplateColumns: "repeat(4, minmax(220px, 1fr))",
//             gap: 12,
//           }}
//         >
//           {(aggs.length ? aggs : Array.from({ length: 8 }).map(() => ({ name: "—", value: null }))).map(
//             (k, i) => (
//               <Card key={i} style={{ padding: 0 }}>
//                 <div
//                   style={{
//                     background: gradients[i % gradients.length],
//                     borderBottom: `1px solid ${c.border}`,
//                     padding: 12,
//                     borderRadius:12
//                   }}
//                 >
//                   <div
//                     style={{
//                       fontSize: 14,
//                       fontWeight: 500,
//                       color: "#1a2127ff",
//                       letterSpacing: 0.3,
//                     //   textTransform: "uppercase",
                      
//                     }}
//                   >
//                     {k.name}
//                   </div>
//                 </div>

//                 <div
//                   style={{
//                     padding: 14,
//                     display: "flex",
//                     alignItems: "baseline",
//                     gap: 10,
//                   }}
//                 >
//                   <div style={{ fontSize: 24, fontWeight: 500, color: c.text }}>
//                     {k.value !== null && k.value !== undefined
//                       ? Number(k.value).toLocaleString()
//                       : loading
//                       ? "…"
//                       : "—"}
//                   </div>
//                   {/* {k.value !== null && (
//                     <span
//                       style={{
//                         fontSize: 11,
//                         fontWeight: 400,
//                         color: c.successText,
//                         background: c.successBg,
//                         border: `1px solid #d2ead7`,
//                         padding: "2px 6px",
//                         borderRadius: 6,
//                       }}
//                     >
//                       Live
//                     </span>
//                   )} */}
//                 </div>
//               </Card>
//             )
//           )}
//         </div>
//       </div>

//       {/* Debug */}
//       {debug.length > 0 && (
//         <Card style={{ marginTop: 14, background: "#fbfdff" }}>
//           <div style={{ fontWeight: 600, color: c.text, marginBottom: 6 }}>
//             KPI Debug
//           </div>
//           <pre
//             style={{
//               margin: 0,
//               whiteSpace: "pre-wrap",
//               fontSize: 12,
//               color: c.subtext,
//             }}
//           >
//             {JSON.stringify(debug, null, 2)}
//           </pre>
//         </Card>
//       )}
//     </div>
//   );
// }

// KPIPanel.jsx
import React, { useRef, useState, useMemo } from "react";

/** ---------- Design Tokens ---------- */
const t = {
  // brand
  brand: "#4a69bd",
  brandSoft: "#87a4e4ff",
  brandTextOn: "#ffffff",

  // surface
  bg: "#f7f8fb",
  surface: "#ffffff",
  border: "#e6ebf2",

  // text
  text: "#0f172a",
  subtext: "#64748b",

  // subtle shadows
  cardShadow: "0 2px 6px rgba(18, 36, 99, 0.08)",
};

/** ---------- Small helpers ---------- */
const Card = ({ children, style }) => (
  <div
    style={{
      background: t.surface,
      borderRadius: 14,
      border: `1px solid ${t.border}`,
      boxShadow: t.cardShadow,
      ...style,
    }}
  >
    {children}
  </div>
);

const GhostButton = ({ children, onClick, disabled, title }) => (
  <button
    onClick={disabled ? undefined : onClick}
    disabled={disabled}
    title={title}
    style={{
      height: 34,
      padding: "0 14px",
      borderRadius: 10,
      border: `1px solid ${t.border}`,
      background: "#fff",
      color: t.text,
      fontWeight: 600,
      fontSize: 12,
      cursor: disabled ? "not-allowed" : "pointer",
    }}
    onMouseEnter={(e) => !disabled && (e.currentTarget.style.background = "#f8fafc")}
    onMouseLeave={(e) => !disabled && (e.currentTarget.style.background = "#fff")}
  >
    {children}
  </button>
);

const PrimaryButton = ({ children, onClick, disabled, title }) => (
  <button
    onClick={disabled ? undefined : onClick}
    disabled={disabled}
    title={title}
    style={{
      height: 34,
      padding: "0 14px",
      borderRadius: 10,
      border: `1px solid ${t.brand}`,
      background: "#eef3ff",
      color: t.brand,
      fontWeight: 700,
      fontSize: 12,
      cursor: disabled ? "not-allowed" : "pointer",
    }}
    onMouseEnter={(e) => !disabled && (e.currentTarget.style.background = "#e3ebff")}
    onMouseLeave={(e) => !disabled && (e.currentTarget.style.background = "#eef3ff")}
  >
    {children}
  </button>
);

/** ---------- Component ---------- */
export default function KPIPanel() {
  const [loading, setLoading] = useState(false);
  const [aggs, setAggs] = useState([]);
  const [error, setError] = useState("");
  const [debug, setDebug] = useState([]);

  const [rowsSetInfo, setRowsSetInfo] = useState(null);
  const [availableCols, setAvailableCols] = useState([]); // all columns from upload
  const [selectedCols, setSelectedCols] = useState([]);   // user-chosen columns

  const [showColsPopup, setShowColsPopup] = useState(false);
  const [search, setSearch] = useState("");

  const fileRef = useRef(null);
  const openFilePicker = () => fileRef.current?.click();

  /** ---------- Upload CSV ---------- */
  const uploadCSV = async (file) => {
  if (!file) return;

  const fd = new FormData();
  fd.append("file", file);

  setError("");
  setRowsSetInfo(null);
  setAggs([]);
  setDebug([]);
  setAvailableCols([]);
  setSelectedCols([]);

  try {
        const res = await fetch("http://localhost:8000/kpi/upload-csv", {
        method: "POST",
        body: fd,
        });

        const body = await res.json();

        if (!res.ok || !body.ok) throw new Error(body.error || "Upload failed");

        const cols = Array.isArray(body.cols) ? body.cols : [];
        setRowsSetInfo({ rows: body.rows, cols });
        setAvailableCols(cols);
        setSelectedCols(cols); // Default to selecting all columns

        console.log("✅ File uploaded and registered in backend:", body);
    } catch (e) {
        console.error("❌ Upload error:", e);
        setError(e.message);
    }
    };


  /** ---------- Compute KPIs using selected columns ---------- */
  const compute = async () => {
    setLoading(true);
    setError("");
    setAggs([]);
    setDebug([]);
    try {
       const res = await fetch("http://localhost:8000/kpi/start-selected", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ columns: selectedCols }),
        });
      const body = await res.json();
      setAggs(body.aggregations || []);
      setDebug(body.kpi_debug || []);
      if (body.error) setError(body.error);
    } catch (e) {
      setError("Failed to compute KPIs");
    } finally {
      setLoading(false);
    }
  };

  // KPI tile header backgrounds (soft)
  const tileBg = [
    "#eef3ff",
    "#f4f7ff",
    "#f6fbff",
    "#fdf7ff",
    "#f7fff9",
    "#fff7f0",
    "#f3f6fb",
  ];

  // Column pills truncation
  const MAX_PILLS = 8;
  const cols = rowsSetInfo?.cols || [];
  const pillPreview = selectedCols.slice(0, MAX_PILLS);
  const moreCount = Math.max(0, selectedCols.length - MAX_PILLS);

  // Filtered list for popup
  const filteredPopupCols = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return availableCols;
    return availableCols.filter((c) => String(c).toLowerCase().includes(q));
  }, [availableCols, search]);

  const toggleCol = (name) => {
    setSelectedCols((prev) =>
      prev.includes(name) ? prev.filter((c) => c !== name) : [...prev, name]
    );
  };

  const selectAll = () => setSelectedCols(availableCols);
  const clearAll = () => setSelectedCols([]);

  return (
    <div style={{  padding: 4, fontFamily: "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif" }}>
      {/* Hidden file input */}
      <input
        ref={fileRef}
        type="file"
        accept=".csv,.xlsx"
        style={{ display: "none" }}
        onChange={(e) => uploadCSV(e.target.files?.[0] || null)}
      />

      {/* ===== Top Row: Blue left(1/3) + White right(2/3) ===== */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 2fr", // 1/3 : 2/3
          gap: 12,
          alignItems: "stretch",
        }}
      >
        {/* Left: Blue container (square corners as requested) */}
        <div
          style={{
            borderRadius: 28,
            background: `linear-gradient(135deg, ${t.brand}, ${t.brandSoft})`,
            color: t.brandTextOn,
            boxShadow: t.cardShadow,
            border: `1px solid rgba(255,255,255,0.25)`,
            padding: 24,
            display: "flex",
            flexDirection: "column",
            minHeight: 220,
          }}
        >
          <div
            style={{
              fontSize: 18,
              fontWeight: 600,
              marginBottom: 10,
              fontFamily:
                "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif",
            }}
          >
            Load dataset
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <PrimaryButton onClick={openFilePicker} title="Upload a CSV/Excel file">
              Upload CSV
            </PrimaryButton>
            <GhostButton
              onClick={compute}
              disabled={loading || selectedCols.length === 0}
              title={
                selectedCols.length === 0
                  ? "Select at least one column"
                  : "Compute KPIs using selected columns"
              }
            >
              {loading ? "Computing KPIs…" : "Compute KPIs"}
            </GhostButton>
            {/* <GhostButton
              onClick={() => setShowColsPopup(true)}
              disabled={!availableCols.length}
              title="Add/Remove columns"
            >
              Manage Columns
            </GhostButton> */}
          </div>

          {error && (
            <div
              style={{
                marginTop: 10,
                background: "rgba(255,255,255,.16)",
                border: "1px solid rgba(255,255,255,.3)",
                color: "#fff",
                fontWeight: 600,
                borderRadius: 8,
                padding: "8px 10px",
              }}
            >
              {error}
            </div>
          )}

          {/* Rows / Columns summary */}
          <div
            style={{
              marginTop: 12,
              background: "rgba(255,255,255,.12)",
              border: "1px solid rgba(255,255,255,.28)",
              borderRadius: 12,
              padding: 12,
            }}
          >
            {rowsSetInfo ? (
              <>
                <div style={{ display: "flex", gap: 12, marginBottom: 8 }}>
                  <div>
                    <div style={{ opacity: 0.9, fontSize: 12 }}>Records</div>
                    <div style={{ fontWeight: 600, fontSize: 18 }}>
                      {rowsSetInfo.rows.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div style={{ opacity: 0.9, fontSize: 12 }}>Selected Columns</div>
                    <div style={{ fontWeight: 600, fontSize: 18 }}>
                      {selectedCols.length}
                    </div>
                  </div>
                </div>

                {/* Selected column pills preview + more */}
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: 6,
                    alignItems: "center",
                  }}
                >
                  {pillPreview.map((cname, i) => (
                    <span
                      key={i}
                      style={{
                        background: "rgba(255,255,255,.28)",
                        color: "#fff",
                        padding: "3px 8px",
                        borderRadius: 999,
                        fontSize: 11,
                        border: "1px solid rgba(255,255,255,.38)",
                      }}
                    >
                      {String(cname)}
                    </span>
                  ))}
                  {moreCount > 0 && (
                    <button
                      onClick={() => setShowColsPopup(true)}
                      style={{
                        background: "transparent",
                        border: "1px dashed rgba(255,255,255,.55)",
                        color: "#fff",
                        padding: "3px 8px",
                        borderRadius: 999,
                        fontSize: 11,
                        cursor: "pointer",
                      }}
                      title="Show & edit all selected columns"
                    >
                      … +{moreCount} more
                    </button>
                  )}
                </div>
              </>
            ) : (
              <div style={{ opacity: 0.9 }}>
                No dataset loaded. Upload a file to begin.
              </div>
            )}
          </div>
        </div>

        {/* Right: KPI tiles container */}
        <Card style={{ padding: 24, minHeight: 220, borderRadius: 28 }}>
          <div
            style={{
              fontSize: 16,
              fontWeight: 700,
              color: t.text,
              margin: "2px 2px 10px",
            }}
          >
            KPIs
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(3, minmax(160px, 1fr))",
              gap: 12,
            }}
          >
            {(aggs.length
              ? aggs
              : Array.from({ length: 6 }).map(() => ({ name: "—", value: null }))
            ).map((k, i) => (
              <Card key={i} style={{ padding: 0 }}>
                <div
                  style={{
                    // background: tileBg[i % tileBg.length],
                    // borderBottom: `1px solid ${t.border}`,
                    borderTopLeftRadius: 14,
                    borderTopRightRadius: 14,
                    padding: 10,
                  }}
                >
                  <div
                    style={{
                      fontSize: 13,
                      fontWeight: 600,
                      color: "#1a2127",
                      letterSpacing: 0.2,
                    }}
                  >
                    {k.name}
                  </div>
                </div>

                <div
                  style={{
                    padding: 14,
                    display: "flex",
                    alignItems: "baseline",
                    gap: 10,
                  }}
                >
                  <div style={{ fontSize: 24, fontWeight: 600, color: t.text }}>
                    {k.value !== null && k.value !== undefined
                      ? Number(k.value).toLocaleString()
                      : loading
                      ? "…"
                      : "—"}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </Card>
      </div>

      {/* ===== Debug block (optional) ===== */}
      {debug.length > 0 && (
        <Card style={{ marginTop: 14, background: "#fbfdff" }}>
          <div style={{ fontWeight: 700, color: t.text, marginBottom: 6 }}>
            KPI Debug
          </div>
          <pre
            style={{
              margin: 0,
              whiteSpace: "pre-wrap",
              fontSize: 12,
              color: t.subtext,
            }}
          >
            {JSON.stringify(debug, null, 2)}
          </pre>
        </Card>
      )}

      {/* ===== Columns Manager Popup ===== */}
      {showColsPopup && (
        <div
          onClick={() => setShowColsPopup(false)}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(15,23,42,.28)",
            zIndex: 50,
            display: "grid",
            placeItems: "center",
            padding: 12,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: "min(720px, 96vw)",
              maxHeight: "80vh",
              display: "flex",
              flexDirection: "column",
              background: "#fff",
              border: `1px solid ${t.border}`,
              borderRadius: 14,
              boxShadow: "0 20px 40px rgba(0,0,0,.18)",
            }}
          >
            {/* Header */}
            <div
              style={{
                padding: 12,
                borderBottom: `1px solid ${t.border}`,
                display: "flex",
                alignItems: "center",
                gap: 8,
                justifyContent: "space-between",
              }}
            >
              <div style={{ fontWeight: 800, fontSize: 16, color: t.text }}>
                Manage Columns ({selectedCols.length}/{availableCols.length})
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <GhostButton onClick={selectAll} title="Select all columns">
                  Select all
                </GhostButton>
                <GhostButton onClick={clearAll} title="Clear all selections">
                  Clear all
                </GhostButton>
              </div>
            </div>

            {/* Search */}
            <div style={{ padding: 12, borderBottom: `1px solid ${t.border}` }}>
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search columns…"
                style={{
                  width: "100%",
                  height: 36,
                  borderRadius: 10,
                  border: `1px solid ${t.border}`,
                  padding: "0 12px",
                  outline: "none",
                  fontSize: 14,
                }}
              />
            </div>

            {/* List */}
            <div
              style={{
                padding: 12,
                overflowY: "auto",
              }}
            >
              {filteredPopupCols.length === 0 ? (
                <div style={{ color: t.subtext, fontSize: 13 }}>
                  No columns match your search.
                </div>
              ) : (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(2, minmax(220px, 1fr))",
                    gap: 8,
                  }}
                >
                  {filteredPopupCols.map((name) => {
                    const checked = selectedCols.includes(name);
                    return (
                      <label
                        key={name}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 10,
                          padding: "8px 10px",
                          border: `1px solid ${checked ? t.brand : t.border}`,
                          borderRadius: 10,
                          cursor: "pointer",
                          background: checked ? "#eef3ff" : "#fff",
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => toggleCol(name)}
                          style={{ width: 16, height: 16 }}
                        />
                        <span style={{ fontSize: 13, color: t.text }}>
                          {String(name)}
                        </span>
                      </label>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Footer */}
            <div
              style={{
                padding: 12,
                borderTop: `1px solid ${t.border}`,
                display: "flex",
                justifyContent: "flex-end",
                gap: 8,
              }}
            >
              <GhostButton onClick={() => setShowColsPopup(false)}>Cancel</GhostButton>
              <PrimaryButton
                onClick={() => setShowColsPopup(false)}
                title="Use these columns"
              >
                Apply
              </PrimaryButton>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
