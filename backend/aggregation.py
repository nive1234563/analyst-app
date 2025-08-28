# aggregation.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
import os, json, importlib, re

router = APIRouter(prefix="/compare", tags=["compare"])

class StartReq(BaseModel):
    dataset_id: str = "default"  # kept for compatibility with existing frontend

# ======================== AI helpers ========================

def _has_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

def _ai_json(prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
    """
    Ask AI to output ONLY valid JSON.
    If parsing fails or AI key missing, return {}.
    """
    if not _has_openai():
        return {}
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are a data analysis planner. Output ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        txt = (resp.choices[0].message.content or "").strip()
        # Strip ```json fences if present
        if txt.startswith("```"):
            txt = txt.strip("`")
            nl = txt.find("\n")
            if nl != -1:
                txt = txt[nl + 1 :]
        return json.loads(txt)
    except Exception as e:
        print("AI JSON parse failed:", e)
        return {}

# ====================== Data access =========================

LAST_GOOD_DF: pd.DataFrame | None = None  # soft cache to avoid hard failures

def _load_uploaded_df() -> Tuple[pd.DataFrame | None, str | None]:
    """
    Late-import main and read global `uploaded_df`.
    Never raises; returns (df or None, warning_message or None).
    """
    try:
        main = importlib.import_module("main")
    except Exception as e:
        return None, f"Couldn't import main: {e}"

    if not hasattr(main, "uploaded_df"):
        return None, "uploaded_df not found in main. Upload a CSV first."

    raw = getattr(main, "uploaded_df")

    # Already a DataFrame
    if isinstance(raw, pd.DataFrame):
        if raw is None or raw.empty:
            return None, "uploaded_df is empty."
        return raw, None

    # Try typical shapes (list of dicts / dict of lists / {'rows': [...]})
    try:
        if isinstance(raw, list):
            df = pd.DataFrame(raw)
        elif isinstance(raw, dict):
            if "rows" in raw and isinstance(raw["rows"], list):
                df = pd.DataFrame(raw["rows"])
            else:
                df = pd.DataFrame(raw)
        else:
            return None, f"Unsupported uploaded_df type: {type(raw)}"
    except Exception as e:
        return None, f"uploaded_df conversion failed: {e}"

    if df is None or df.empty:
        return None, "uploaded_df converted but is empty."
    return df, None

# ====================== Chart helpers =======================

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)

def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or col not in df.columns:
        return pd.Series([], dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan).dropna()

def _cat_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or col not in df.columns:
        return pd.Series([], dtype="object")
    return df[col].astype(str).fillna("NA")

def _safe_chart_type(t: str, x_is_num: bool, y_is_num: bool, x_is_dt: bool) -> str:
    t = (t or "").lower()
    if t == "line" and (x_is_num or x_is_dt):
        return "line"
    if t == "scatter" and x_is_num and y_is_num:
        return "scatter"
    if t == "bar":
        return "bar"
    if x_is_num and y_is_num:
        return "scatter"
    if (not x_is_num) and y_is_num:
        return "bar"
    if x_is_dt and y_is_num:
        return "line"
    return "bar"

def _build_chart(df: pd.DataFrame, x_col: str, y_col: str, chart_type: str, max_points: int = 400) -> Dict[str, Any]:
    if df is None or x_col not in df.columns or y_col not in df.columns:
        return {"type": "bar", "x": [], "y": [], "xLabel": x_col, "yLabel": y_col}

    x = df[x_col]; y = df[y_col]
    x_is_num = _is_numeric(x); y_is_num = _is_numeric(y); x_is_dt = _is_datetime(x)
    t = _safe_chart_type(chart_type, x_is_num, y_is_num, x_is_dt)

    if t == "bar":
        x_cat = _cat_series(df, x_col)
        y_num = _num_series(df, y_col)
        temp = pd.DataFrame({x_col: x_cat, y_col: y_num}).dropna()
        if temp.empty:
            return {"type": "bar", "x": [], "y": [], "xLabel": x_col, "yLabel": y_col}
        agg = temp.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(20)
        return {"type": "bar", "x": agg.index.tolist(), "y": agg.values.tolist(), "xLabel": x_col, "yLabel": y_col}

    if t == "line":
        x_parsed = x if (x_is_dt or x_is_num) else pd.to_datetime(x, errors="coerce")
        y_num = _num_series(df, y_col)
        temp = pd.DataFrame({x_col: x_parsed, y_col: y_num}).dropna().sort_values(x_col)
        if temp.empty:
            return {"type": "line", "x": [], "y": [], "xLabel": x_col, "yLabel": y_col}
        if len(temp) > max_points:
            step = max(1, len(temp)//max_points)
            temp = temp.iloc[::step]
        # For simplicity we render x as positional indexes; keep label in xLabel
        return {"type": "line", "x": list(range(len(temp))), "y": temp[y_col].tolist(), "xLabel": x_col, "yLabel": y_col}

    # scatter
    x_num = _num_series(df, x_col)
    y_num = _num_series(df, y_col)
    n = min(len(x_num), len(y_num))
    temp = pd.DataFrame({x_col: x_num.iloc[:n], y_col: y_num.iloc[:n]}).dropna()
    if temp.empty:
        return {"type": "scatter", "x": [], "y": [], "xLabel": x_col, "yLabel": y_col}
    if len(temp) > max_points:
        temp = temp.sample(max_points, random_state=42)
    return {"type": "scatter", "x": temp[x_col].tolist(), "y": temp[y_col].tolist(), "xLabel": x_col, "yLabel": y_col}

def _variation_stats(s: pd.Series) -> Dict[str, Any]:
    s_num = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s_num) == 0:
        return {"count": 0}
    q1, q3 = np.percentile(s_num, [25, 75])
    return {
        "count": int(len(s_num)), "mean": float(np.mean(s_num)),
        "std": float(np.std(s_num, ddof=1)) if len(s_num) > 1 else 0.0,
        "min": float(np.min(s_num)), "max": float(np.max(s_num)),
        "median": float(np.median(s_num)), "q1": float(q1), "q3": float(q3),
        "unique": int(len(pd.unique(s_num))),
    }

# ======================= KPI helpers ========================

_ALLOWED_FUNCS = {
    "sum": "sum", "avg": "mean", "average": "mean", "mean": "mean",
    "min": "min", "max": "max", "median": "median",
    "std": "std", "stdev": "std", "stddev": "std",
    "count": "count",
    "distinct_count": "nunique", "nunique": "nunique", "unique": "nunique",
}

_CURRENCY = re.compile(r"[₹$€£,%\s]")

def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or col not in df.columns:
        return pd.Series([], dtype="float64")
    # commas, whitespace, currency, % signs
    s = df[col].astype(str).str.replace(",", "", regex=False)
    s = s.apply(lambda x: _CURRENCY.sub("", x))
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return s

def _compute_single_kpi(df: pd.DataFrame, col: str, func_key: str) -> Union[float, int, None]:
    func = _ALLOWED_FUNCS.get((func_key or "").lower().strip())
    if not func:
        return None
    if func in {"sum","mean","min","max","median","std"}:
        series = _to_numeric_series(df, col)
        if series.notna().sum() == 0:
            return None
        return getattr(series, func)()
    if func == "count":
        return 0 if df is None else int(len(df))
    if func == "nunique":
        if df is None or col not in df.columns:
            return None
        return int(df[col].nunique(dropna=True))
    return None

def _compute_kpi_with_reason(df: pd.DataFrame | None, k: Dict[str, Any]):
    name = str(k.get("name", "KPI")).strip()[:60] or "KPI"
    col  = str(k.get("column", "")).strip()
    agg  = str(k.get("aggregation", "")).strip().lower()
    if agg not in _ALLOWED_FUNCS:
        return None, f"unknown agg '{agg}'"
    val = _compute_single_kpi(df, col, agg)
    if val is None:
        miss = "missing col" if (df is not None and col and col not in (df.columns if hasattr(df, "columns") else [])) else "no numeric values"
        return None, f"cannot compute {agg} on '{col}' ({miss})"
    if isinstance(val, (float, np.floating)):
        out = float(val)
    elif isinstance(val, (int, np.integer)):
        out = int(val)
    else:
        try:
            out = float(val)
        except:
            return None, "non-castable value"
    return {"name": name, "value": out}, None

def _propose_kpis_without_ai(df: pd.DataFrame | None) -> List[dict]:
    """
    Deterministic KPI defs using name heuristics.
    Works even if df=None (we'll still emit count); returns up to 12.
    """
    cols = [] if df is None else df.columns.tolist()
    name_lc = {c: c.lower() for c in cols}

    def has(substrs):
        for c in cols:
            lc = name_lc[c]
            if any(sub in lc for sub in substrs):
                return c
        return None

    # Heuristic targets
    col_sales = has(["sales", "amount", "revenue", "total"])
    col_qty   = has(["quantity", "qty", "units", "count"])
    col_price = has(["price", "unit_price", "rate"])
    col_disc  = has(["discount", "disc", "offer"])
    col_cust  = has(["customer", "cust_id", "client", "user"])
    col_prod  = has(["product", "sku", "item"])
    col_loc   = has(["store", "location", "branch", "city", "region"])

    defs: List[dict] = []

    # Revenue-ish
    if col_sales:
        defs += [
            {"name": f"Total {col_sales}", "column": col_sales, "aggregation": "sum"},
            {"name": f"Avg {col_sales}",   "column": col_sales, "aggregation": "mean"},
            {"name": f"Max {col_sales}",   "column": col_sales, "aggregation": "max"},
        ]
    # Quantity-ish
    if col_qty:
        defs += [
            {"name": f"Total {col_qty}", "column": col_qty, "aggregation": "sum"},
            {"name": f"Avg {col_qty}",   "column": col_qty, "aggregation": "mean"},
        ]
    # Price-ish
    if col_price:
        defs += [
            {"name": f"Avg {col_price}", "column": col_price, "aggregation": "mean"},
            {"name": f"Max {col_price}", "column": col_price, "aggregation": "max"},
        ]
    # Discount
    if col_disc:
        defs.append({"name": f"Avg {col_disc}", "column": col_disc, "aggregation": "mean"})

    # Counts
    defs.append({"name": "Row Count", "column": "", "aggregation": "count"})
    if col_cust and len(defs) < 12:
        defs.append({"name": f"Unique {col_cust}", "column": col_cust, "aggregation": "distinct_count"})
    if col_prod and len(defs) < 12:
        defs.append({"name": f"Unique {col_prod}", "column": col_prod, "aggregation": "distinct_count"})
    if col_loc and len(defs) < 12:
        defs.append({"name": f"Unique {col_loc}", "column": col_loc, "aggregation": "distinct_count"})

    # Top up with averages of other numeric cols
    if df is not None:
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        for c in numeric_cols:
            if len(defs) >= 12:
                break
            if not any(d["column"] == c and d["aggregation"] == "mean" for d in defs):
                defs.append({"name": f"Avg {c}", "column": c, "aggregation": "mean"})

    # De-dup
    seen = set()
    dedup = []
    for d in defs:
        key = (d["name"], d["column"], d["aggregation"])
        if key not in seen:
            seen.add(key)
            dedup.append(d)
    return dedup[:12]

# ========================== Routes ==========================

@router.post("/start")
def compare_start(req: StartReq):
    """
    Returns: 200 always.
    {
      "pairs": [...],
      "aggregations": [...8 items...],
      "kpi_debug": [...],
      "error": "optional message if df missing/invalid"
    }
    """
    global LAST_GOOD_DF
    error_msg = None

    df, warn = _load_uploaded_df()
    if warn:
        error_msg = warn
        print("WARNING:", warn)

    # Soft-fallback to last good df if present
    if df is None and LAST_GOOD_DF is not None:
        df = LAST_GOOD_DF
        print("Using LAST_GOOD_DF fallback for /compare/start")

    # If still None, work in "empty" mode
    if df is None:
        pairs = []
        sample_rows = []
        schema = {}
    else:
        sample_rows = df.head(3).to_dict(orient="records")
        schema = {c: str(df[c].dtype) for c in df.columns.tolist()}

    # ----- Ask AI for PAIRS -----
    prompt_pairs = f"""
Given this table schema and 3 sample rows, propose up to 6 useful column comparisons.
Return STRICT JSON with top-level "pairs": [{{id,title,x_col,y_col,chart_type,chart_name,reason}}].
Chart types: bar | line | scatter.

SCHEMA:
{json.dumps(schema, indent=2)}

SAMPLE:
{json.dumps(sample_rows, indent=2)}
"""
    ai_pairs = _ai_json(prompt_pairs) if df is not None else {}
    print("AI response for pairs:", ai_pairs)
    pairs = ai_pairs.get("pairs") if isinstance(ai_pairs, dict) else None

    # Fallback pairs if needed
    if not pairs:
        pairs = []
        if df is not None:
            numeric = [c for c in df.columns if _is_numeric(df[c])]
            cats = [c for c in df.columns if not _is_numeric(df[c])]
            pid = 1
            # numeric vs numeric
            for i in range(min(2, max(0, len(numeric) - 1))):
                pairs.append({
                    "id": pid, "title": f"{numeric[i]} vs {numeric[i+1]}",
                    "x_col": numeric[i], "y_col": numeric[i+1],
                    "chart_type": "scatter", "chart_name": f"{numeric[i]} vs {numeric[i+1]}",
                    "reason": "Numeric relationship check."
                })
                pid += 1
            # cat vs numeric
            if cats and numeric:
                pairs.append({
                    "id": pid, "title": f"{cats[0]} vs {numeric[0]}",
                    "x_col": cats[0], "y_col": numeric[0],
                    "chart_type": "bar", "chart_name": f"{cats[0]} vs {numeric[0]}",
                    "reason": "Compare numeric across categories."
                })

    # Store pairs into df for /insight (only if we have a df)
    if df is not None:
        setattr(df, "_compare_pairs", {int(p["id"]): p for p in pairs})
        LAST_GOOD_DF = df  # update soft cache

    # ----- Ask AI for KPI DEFS -----
    kpi_defs: List[Dict[str, Any]] = []
    ai_kpis = _ai_json(_kpi_prompt(schema, sample_rows)) if df is not None else {}
    print("AI response for KPIs:", ai_kpis)

    if isinstance(ai_kpis, dict) and isinstance(ai_kpis.get("kpis"), list):
        for k in ai_kpis["kpis"]:
            try:
                name = str(k.get("name", "")).strip()[:60] or "KPI"
                col  = str(k.get("column", "")).strip()
                agg  = str(k.get("aggregation", "")).strip().lower()
                if agg in _ALLOWED_FUNCS:
                    kpi_defs.append({"name": name, "column": col, "aggregation": agg})
            except Exception:
                continue

    # If AI gave nothing, propose deterministically
    if not kpi_defs:
        kpi_defs = _propose_kpis_without_ai(df)

    # Compute KPI values with reasons; guarantee 8
    kpi_debug: List[Dict[str, Any]] = []
    aggregations: List[Dict[str, Any]] = []
    for k in kpi_defs[:12]:
        out, reason = _compute_kpi_with_reason(df, k)
        if out is not None:
            aggregations.append(out)
        else:
            kpi_debug.append({"requested": k, "skip_reason": reason})

    # Top up to 8 using deterministic defs
    if len(aggregations) < 8:
        for k in _propose_kpis_without_ai(df):
            if len(aggregations) >= 8:
                break
            out, reason = _compute_kpi_with_reason(df, k)
            if out is not None and all(a["name"] != out["name"] for a in aggregations):
                aggregations.append(out)
            elif reason:
                kpi_debug.append({"requested": k, "skip_reason": reason})

    # If still short (e.g., df None), pad
    while len(aggregations) < 8:
        aggregations.append({"name": f"KPI {len(aggregations)+1}", "value": 0})

    return {
        "pairs": pairs,
        "aggregations": aggregations[:8],
        "kpi_debug": kpi_debug,
        "error": error_msg,
    }

def _kpi_prompt(schema: Dict[str,str], sample_rows: List[Dict[str,Any]]) -> str:
    return f"""
You are given a table schema and 3 sample rows. Propose up to 12 KPI cards.

Return STRICT JSON with key "kpis": array of objects with:
  - "name": short display name
  - "column": exact column name to aggregate (or "" for row count)
  - "aggregation": one of ["sum","avg","mean","min","max","median","std","count","distinct_count"]

SCHEMA:
{json.dumps(schema, indent=2)}

SAMPLE ROWS:
{json.dumps(sample_rows, indent=2)}
"""

@router.get("/insight/{pair_id}")
def compare_insight(pair_id: int):
    """
    Chart + AI text insight for a given pair.
    If df is missing, returns empty chart and a default message (200).
    """
    global LAST_GOOD_DF

    df, warn = _load_uploaded_df()
    if df is None:
        df = LAST_GOOD_DF  # soft fallback

    pair_meta = {}
    if df is not None:
        pair_meta = getattr(df, "_compare_pairs", {}).get(int(pair_id), {})

    if not pair_meta:
        # graceful response without raising
        return {
            "chart": {"type": "bar", "x": [], "y": [], "xLabel": "", "yLabel": ""},
            "insight": "Run Start Analysis first to generate comparison pairs."
        }

    x_col, y_col = pair_meta["x_col"], pair_meta["y_col"]
    chart_type = pair_meta["chart_type"]
    chart_name = pair_meta["chart_name"]

    x_stats = _variation_stats(df[x_col]) if (df is not None and x_col in df.columns) else {"count": 0}
    y_stats = _variation_stats(df[y_col]) if (df is not None and y_col in df.columns) else {"count": 0}

    prompt = f"""
Columns: x={x_col}, y={y_col}
Chart type: {chart_type}
Chart name: {chart_name}
X_STATS: {json.dumps(x_stats)}
Y_STATS: {json.dumps(y_stats)}
Write a concise business insight (4–6 sentences). Plain text only.
"""

    insight_text = ""
    if _has_openai():
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.4,
                messages=[
                    {"role": "system", "content": "You are a crisp data analyst. Output plain text only."},
                    {"role": "user", "content": prompt},
                ],
            )
            insight_text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print("AI insight failed:", e)

    if not insight_text:
        insight_text = f"{chart_name} ({chart_type}). Inspect trend, variability, and any clusters/outliers."

    chart = _build_chart(df, x_col, y_col, chart_type)
    return {"chart": chart, "insight": insight_text}
