
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd

from state import UPLOADED_DF

try:
    from openai import OpenAI
    import os
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

router = APIRouter(prefix="/ai", tags=["AI"])

class DatasetInsightReq(BaseModel):
    dataset_id: str = "default"
    max_rows: int = 5

def _detect_date_cols(df: pd.DataFrame) -> List[str]:
    out = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(c)
            continue
        if s.dtype == object:
            sample = s.dropna().astype(str).head(200)
            ok = 0
            for v in sample:
                try:
                    pd.to_datetime(v, dayfirst=True)
                    ok += 1
                except Exception:
                    pass
            if len(sample) and ok / len(sample) >= 0.6:
                out.append(c)
    return out

def _detect_numeric_cols(df: pd.DataFrame) -> List[str]:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in df.columns:
        if c in num: 
            continue
        s = df[c].dropna().astype(str).head(200)
        if len(s) and (s.str.match(r"^-?\\d+(\\.\\d+)?$").mean() > 0.7):
            num.append(c)
    return list(dict.fromkeys(num))

def _detect_categorical_cols(df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
    return [c for c in df.columns if c not in numeric_cols]

def _feasibility(df: pd.DataFrame, date_cols: List[str], numeric_cols: List[str]) -> Dict[str, Dict[str, Any]]:
    forecast_ok = bool(date_cols) and bool(numeric_cols)
    historical_ok = (bool(date_cols) and bool(numeric_cols)) or (len(numeric_cols) > 0 and any(df[c].nunique() > 1 for c in df.columns if c not in numeric_cols))
    eda_ok = df.shape[1] >= 2
    return {
        "forecast": {"ok": forecast_ok, "reason": "Needs a date/time column and at least one numeric measure."},
        "historical": {"ok": historical_ok, "reason": "Needs either a date/time column or categorical dimensions with numeric measures."},
        "eda": {"ok": eda_ok, "reason": "General EDA is possible with any tabular dataset (2+ columns)."},
    }

def _to_preview_rows(df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
    sample = df.head(n).copy()
    for c in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[c]):
            sample[c] = sample[c].astype(str)
    return sample.astype(object).where(pd.notna(sample), None).to_dict(orient="records")

def _model_prompt(sample_rows: List[Dict[str, Any]], cols_meta: List[Dict[str, Any]], feas: Dict[str, Any]) -> str:
    return f"""
You are a senior data analyst. Below is a dataset sample and metadata.

GOAL:
1. First, give a concise 50-word summary of what this dataset is likely about. Use examples if needed.
2. Then, assess feasibility of Forecast, EDA, and Historical analysis:
   - For each: Is it possible? Why or why not?
   - If forecast is possible: recommend a date/time column and numeric target column.
   - Also, suggest if any column needs to be cleaned, converted, or imputed for forecasting.
3. Recommend useful KPIs and charts.
4. Suggest the kind of decisions or business insights this dataset can support.
5. Be clear, compact, and structured.

Return STRICT JSON in this format only:

{{
  "summary": "short description in ~50 words",
  "feasibility": {{
    "forecast": {{ "ok": true, "reason": "..." }},
    "eda": {{ "ok": true, "reason": "..." }},
    "historical": {{ "ok": true, "reason": "..." }}
  }},
  "forecast_guidance": {{
    "time_col": "string or null",
    "target_col": "string or null",
    "modifications": ["suggestion 1", "suggestion 2"]
  }},
  "suggested_kpis": ["..."],
  "suggested_charts": ["..."],
  "insight_potential": ["..."]
}}

Sample rows:
{json.dumps(sample_rows, ensure_ascii=False)}

Columns:
{json.dumps(cols_meta, ensure_ascii=False)}

Heuristic feasibility:
{json.dumps(feas, ensure_ascii=False)}
""".strip()

@router.post("/dataset_insights")
def dataset_insights(req: DatasetInsightReq):
    if req.dataset_id not in UPLOADED_DF:
        raise HTTPException(404, "Dataset not found. Upload a CSV first.")

    df = UPLOADED_DF[req.dataset_id].copy()
    if df is None or df.empty:
        raise HTTPException(400, "Dataset is empty.")

    for c in df.columns:
        if df[c].dtype == object:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
                if parsed.notna().mean() > 0.7:
                    df[c] = parsed
            except Exception:
                pass

    date_cols = _detect_date_cols(df)
    numeric_cols = _detect_numeric_cols(df)
    categorical_cols = _detect_categorical_cols(df, numeric_cols)
    feas = _feasibility(df, date_cols, numeric_cols)

    sample_rows = _to_preview_rows(df, req.max_rows)
    cols_meta = []
    for c in df.columns:
        cols_meta.append({
            "name": c,
            "dtype": str(df[c].dtype),
            "unique": int(df[c].nunique(dropna=True)),
            "example": sample_rows[0].get(c, None) if sample_rows else None
        })

    ai_json: Optional[Dict[str, Any]] = None
    ai_text: str = ""

    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = _model_prompt(sample_rows, cols_meta, feas)
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a precise, no-nonsense data analyst."},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
            )
            ai_text = resp.choices[0].message.content or ""
            try:
                ai_json = json.loads(ai_text)
            except Exception:
                ai_json = None
        except Exception as e:
            ai_text = f"⚠️ Model call failed: {e}"

    final_feas = feas.copy()
    if ai_json and isinstance(ai_json.get("feasibility"), dict):
        for k in ("forecast", "eda", "historical"):
            if isinstance(ai_json["feasibility"].get(k), dict) and "ok" in ai_json["feasibility"][k]:
                final_feas[k]["ok"] = bool(ai_json["feasibility"][k]["ok"])
                if "reason" in ai_json["feasibility"][k]:
                    final_feas[k]["reason"] = ai_json["feasibility"][k]["reason"]

    return {
        "dataset_id": req.dataset_id,
        "sample_rows": sample_rows,
        "columns": cols_meta,
        "date_cols": date_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feasibility": final_feas,
        "ai_json": ai_json,
        "ai_text": (None if ai_json else ai_text),
    }
