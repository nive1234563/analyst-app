# backend/analysis/compare_pairs.py

import os, json, math, re, uuid
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Request

# Optional OpenAI
try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

router = APIRouter(prefix="/compare", tags=["compare"])

_SAFE_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


# ------------------ Utility Functions ------------------ #

def _norm_name(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip())

def _build_pair_id() -> str:
    return uuid.uuid4().hex[:12]

def _preview_schema(df: pd.DataFrame, max_cols: int = 40) -> Dict[str, Any]:
    cols = []
    for c in list(df.columns)[:max_cols]:
        s = df[c]
        kind = "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical"
        samples = s.dropna().astype(str).head(5).tolist()
        stat = {}
        if kind == "numeric":
            stat = {
                "mean": float(np.nanmean(pd.to_numeric(s, errors="coerce"))),
                "std": float(np.nanstd(pd.to_numeric(s, errors="coerce"))),
                "min": float(np.nanmin(pd.to_numeric(s, errors="coerce"))),
                "max": float(np.nanmax(pd.to_numeric(s, errors="coerce"))),
            }
        cols.append({"name": c, "kind": kind, "samples": samples, "stats": stat})
    return {"n_rows": int(len(df)), "n_cols": int(len(df.columns)), "columns": cols}


# ------------------ AI Pair Suggestions ------------------ #

def _suggest_pairs(df: pd.DataFrame) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not HAS_OPENAI or not api_key:
        return _fallback_pairs(df)

    schema = _preview_schema(df)
    client = OpenAI(api_key=api_key)
    prompt = f"""
You are a data analyst. Given this dataset schema:
{json.dumps(schema, ensure_ascii=False)}
Return a JSON with key 'pairs': list of {{left, right, chart, title, reason}}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "Suggest meaningful pairs of columns for comparison."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=900,
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
        for p in data.get("pairs", []):
            p["id"] = _build_pair_id()
        return data
    except Exception:
        return _fallback_pairs(df)

def _fallback_pairs(df: pd.DataFrame) -> Dict[str, Any]:
    cols = list(df.columns)
    if len(cols) < 2:
        return {"pairs": []}
    return {
        "pairs": [
            {
                "id": _build_pair_id(),
                "left": cols[0],
                "right": cols[1],
                "chart": "scatter",
                "title": f"{cols[0]} vs {cols[1]}",
                "reason": "Fallback default pair"
            }
        ]
    }


# ------------------ Chart + Insight Generators ------------------ #

def _generate_chart(df: pd.DataFrame, pair: Dict[str, str]) -> Dict[str, Any]:
    x = pd.to_numeric(df[pair["left"]], errors="coerce").dropna()
    y = pd.to_numeric(df[pair["right"]], errors="coerce").dropna()
    n = min(len(x), len(y), 200)
    return {
        "type": pair["chart"],
        "x": x.head(n).round(2).tolist(),
        "y": y.head(n).round(2).tolist(),
        "xLabel": pair["left"],
        "yLabel": pair["right"]
    }

def _generate_insight(pair: Dict[str, str], df: pd.DataFrame) -> str:
    left, right = pair["left"], pair["right"]
    if left not in df.columns or right not in df.columns:
        return f"No data available for {left} vs {right}."

    s = df[[left, right]].dropna().copy()
    s[left] = pd.to_numeric(s[left], errors="coerce")
    s[right] = pd.to_numeric(s[right], errors="coerce")
    s = s.dropna()

    if len(s) < 3:
        return f"Not enough valid data for {left} vs {right} to generate insights."

    stats = {
        "x_mean": float(s[left].mean()),
        "x_std": float(s[left].std()),
        "y_mean": float(s[right].mean()),
        "y_std": float(s[right].std()),
        "count": len(s)
    }

    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = f"""
Write a clear, 50-word insight based on a scatter plot of:
X: {left}
Y: {right}
Stats: {json.dumps(stats)}
Describe trends, variation, or correlation. Use real numeric language.
"""
            resp = client.chat.completions.create(
                model="gpt-4.1-nano",
                temperature=0.3,
                max_tokens=80,
                messages=[
                    {"role": "system", "content": "You generate analytical insights for scatter plots."},
                    {"role": "user", "content": prompt}
                ]
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    return (
        f"Mean {left}: {stats['x_mean']:.1f}, Std: {stats['x_std']:.1f} | "
        f"{right}: {stats['y_mean']:.1f}, Std: {stats['y_std']:.1f}. "
        f"Explore trends or clusters."
    )


# ------------------ API Endpoints ------------------ #

@router.post("/start")
def start_analysis(payload: Dict[str, Any], request: Request):
    dataset_id = payload.get("dataset_id", "default")
    df = request.app.state.cache.get(dataset_id, {}).get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")

    result = _suggest_pairs(df)
    pairs = result.get("pairs", [])
    request.app.state.cache[dataset_id]["pairs"] = pairs

    return {
        "ok": True,
        "dataset_id": dataset_id,
        "pairs": pairs,
        "columns": list(df.columns),
    }

@router.get("/insight/{pair_id}")
def pair_insight(pair_id: str, request: Request):
    dataset_id = "default"
    df = request.app.state.cache.get(dataset_id, {}).get("df")
    pairs = request.app.state.cache.get(dataset_id, {}).get("pairs", [])
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")

    pair = next((p for p in pairs if p["id"] == pair_id), None)
    if not pair:
        raise HTTPException(status_code=404, detail="Pair not found")

    chart = _generate_chart(df, pair)
    insight = _generate_insight(pair, df)
    return {"pair": pair, "chart": chart, "insight": insight}
