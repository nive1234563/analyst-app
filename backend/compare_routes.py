# analysis/compare_routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import os, json

from state import UPLOADED_DF  # do NOT materialize df at import time

router = APIRouter(prefix="/compare", tags=["compare"])

class StartReq(BaseModel):
    dataset_id: str = "default"

# -------- OpenAI helpers (optional) --------
def _has_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

def _ai_json(prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
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
        if txt.startswith("```"):
            # strip code fences and language tag
            txt = txt.strip("`")
            nl = txt.find("\n")
            if nl != -1:
                txt = txt[nl+1:]
        return json.loads(txt)
    except Exception as e:
        print("AI JSON parse failed:", e)
        return {}

# -------- Data + chart helpers --------
def _get_df() -> pd.DataFrame:
    """Unwrap the latest uploaded DataFrame each request."""
    if not isinstance(UPLOADED_DF, dict) or "default" not in UPLOADED_DF:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    df = UPLOADED_DF["default"]
    if df is None or not hasattr(df, "head"):
        raise HTTPException(status_code=400, detail="Uploaded data is invalid.")
    
    return df



def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)

def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan).dropna()

def _cat_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(str).fillna("NA")

def _safe_chart_type(t: str, x_is_num: bool, y_is_num: bool) -> str:
    t = (t or "").lower()
    if t in {"bar", "line", "scatter"}:
        if t == "line" and not (x_is_num or _is_datetime(x_is_num)):
            return "bar"
        if t == "scatter" and not (x_is_num and y_is_num):
            return "bar"
        return t
    # fallback guardrails
    if x_is_num and y_is_num:
        return "scatter"
    if (not x_is_num) and y_is_num:
        return "bar"
    return "bar"

def _build_chart(df: pd.DataFrame, x_col: str, y_col: str, chart_type: str, max_points: int = 400) -> Dict[str, Any]:
    x = df[x_col]; y = df[y_col]
    x_is_num = _is_numeric(x)
    y_is_num = _is_numeric(y)
    t = _safe_chart_type(chart_type, x_is_num, y_is_num)

    if t == "bar":
        x_cat = _cat_series(df, x_col)
        y_num = _num_series(df, y_col)
        temp = pd.DataFrame({x_col: x_cat, y_col: y_num}).dropna()
        agg = temp.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(20)
        return {"type": "bar", "x": agg.index.tolist(), "y": agg.values.tolist(), "xLabel": x_col, "yLabel": y_col}

    if t == "line":
        x_parsed = pd.to_datetime(x, errors="coerce") if (not x_is_num and not _is_datetime(x)) else x
        y_num = _num_series(df, y_col)
        temp = pd.DataFrame({x_col: x_parsed, y_col: y_num}).dropna().sort_values(x_col)
        if len(temp) > max_points:
            temp = temp.iloc[:: max(1, len(temp)//max_points)]
        return {"type": "line", "x": list(range(len(temp))), "y": temp[y_col].tolist(), "xLabel": x_col, "yLabel": y_col}

    # scatter
    x_num = _num_series(df, x_col)
    y_num = _num_series(df, y_col)
    n = min(len(x_num), len(y_num))
    temp = pd.DataFrame({x_col: x_num.iloc[:n], y_col: y_num.iloc[:n]}).dropna()
    if len(temp) > max_points:
        temp = temp.sample(max_points, random_state=42)
    return {"type": "scatter", "x": temp[x_col].tolist(), "y": temp[y_col].tolist(), "xLabel": x_col, "yLabel": y_col}

def _variation_stats(s: pd.Series) -> Dict[str, Any]:
    s_num = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s_num) == 0:
        return {"count": 0}
    q1, q3 = np.percentile(s_num, [25, 75])
    return {
        "count": int(len(s_num)),
        "mean": float(np.mean(s_num)),
        "std": float(np.std(s_num, ddof=1)) if len(s_num) > 1 else 0.0,
        "min": float(np.min(s_num)),
        "max": float(np.max(s_num)),
        "median": float(np.median(s_num)),
        "q1": float(q1), "q3": float(q3),
        "unique": int(len(pd.unique(s_num))),
    }

# ---------------- ROUTES ----------------

@router.post("/start")
def compare_start(req: StartReq):
    df = _get_df()  # <-- fetch live df now
    sample_rows = df.head(3).to_dict(orient="records")
    schema = {c: str(df[c].dtype) for c in df.columns.tolist()}

    prompt = f"""
Given this table schema and 3-row sample, propose up to 6 meaningful column comparisons
for exploratory analysis. For each, select a chart type from ONLY this set: ["bar","line","scatter"].
Return STRICT JSON with a top-level "pairs" array of objects:
  - id (integer starting from 1)
  - title (string, human-friendly)
  - x_col (string)
  - y_col (string)
  - chart_type (bar/line/scatter)
  - chart_name (short string, e.g., "Quantity vs Sales")
  - reason (1–2 lines why this pair is useful)

SCHEMA:
{json.dumps(schema, indent=2)}

SAMPLE (first 3 rows):
{json.dumps(sample_rows, indent=2)}
"""
    ai = _ai_json(prompt)
    pairs = ai.get("pairs") if isinstance(ai, dict) else None

    # Fallback if AI not available/parsable
    if not pairs:
        numeric_cols = [c for c in df.columns if _is_numeric(df[c])]
        cat_cols = [c for c in df.columns if not _is_numeric(df[c])]
        pairs = []
        pid = 1
        for i in range(min(2, max(0, len(numeric_cols)-1))):
            pairs.append({
                "id": pid,
                "title": f"{numeric_cols[i]} vs {numeric_cols[i+1]}",
                "x_col": numeric_cols[i],
                "y_col": numeric_cols[i+1],
                "chart_type": "scatter",
                "chart_name": f"{numeric_cols[i]} vs {numeric_cols[i+1]}",
                "reason": "Numeric vs numeric relationship."
            })
            pid += 1
        if cat_cols and numeric_cols:
            pairs.append({
                "id": pid,
                "title": f"{cat_cols[0]} vs {numeric_cols[0]}",
                "x_col": cat_cols[0],
                "y_col": numeric_cols[0],
                "chart_type": "bar",
                "chart_name": f"{cat_cols[0]} vs {numeric_cols[0]}",
                "reason": "Compare numeric values across categories."
            })

    # Remember chosen pairs on the same DataFrame object for the insight route
    setattr(df, "_compare_pairs", {int(p["id"]): p for p in pairs})
    return {"pairs": pairs}

@router.get("/insight/{pair_id}")
def compare_insight(pair_id: int):
    df = _get_df()  # <-- fetch live df now

    pair_meta = getattr(df, "_compare_pairs", {}).get(int(pair_id))
    if not pair_meta:
        raise HTTPException(status_code=404, detail="Pair not found. Run /compare/start first.")

    x_col, y_col = pair_meta["x_col"], pair_meta["y_col"]
    chart_type = pair_meta["chart_type"]
    chart_name = pair_meta["chart_name"]

    if x_col not in df.columns or y_col not in df.columns:
        raise HTTPException(status_code=400, detail="Columns missing.")

    x_stats = _variation_stats(df[x_col])
    y_stats = _variation_stats(df[y_col])

    prompt = f"""
Columns: x={x_col}, y={y_col}
Chart type: {chart_type}
Chart name: {chart_name}

Variation stats:
X_STATS: {json.dumps(x_stats)}
Y_STATS: {json.dumps(y_stats)}

Write a concise business insight (4–6 sentences).
Respect the chart type exactly. Output plain text only.
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
        insight_text = f"{chart_name} ({chart_type}). Inspect trend, variability, and any clusters/outliers. Consider segment cuts for validation."

    chart = _build_chart(df, x_col, y_col, chart_type)
    return {"chart": chart, "insight": insight_text}
