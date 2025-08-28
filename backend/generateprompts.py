import os
import io
import json
import uuid
import textwrap
from datetime import datetime
from typing import Dict, Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from fastapi import Request
from fastapi.responses import JSONResponse


try:
    from openai import OpenAI
    HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
except ImportError:
    HAS_OPENAI = False


# Shared global cache - imported from main
try:
    from state import UPLOADED_DF
except ImportError:
    UPLOADED_DF = {}


router = APIRouter()

# Folder for storing .txt prompts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_ROOT = os.path.join(BASE_DIR, "static")
PROMPTS_DIR = os.path.join(STATIC_ROOT, "ai_prompts")
os.makedirs(PROMPTS_DIR, exist_ok=True)


def _sample_for_llm(df: pd.DataFrame, n: int = 5) -> Dict[str, Any]:
    sample = df.head(min(n, len(df))).convert_dtypes()
    for c in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[c]):
            sample[c] = sample[c].dt.strftime("%Y-%m-%d %H:%M:%S")
    schema = {
        "columns": [
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "example": None if sample[col].isna().all() else sample[col].dropna().iloc[0].item() if hasattr(sample[col].dropna().iloc[0], "item") else sample[col].dropna().iloc[0]
            }
            for col in df.columns
        ]
    }
    return {
        "shape": list(df.shape),
        "schema": schema,
        "sample_rows": json.loads(sample.to_json(orient="records", date_format="iso"))
    }


def _heuristic_prompts(df: pd.DataFrame) -> Dict[str, str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    y = numeric_cols[0] if numeric_cols else "your_numeric_metric"
    ds = datetime_cols[0] if datetime_cols else "your_date_column"

    forecast = textwrap.dedent(f"""\
        You are a senior data analyst. Perform a time-series forecast.
        - Data target (y): {y}
        - Timestamp column (ds): {ds}
        - If ds is not truly a time column, suggest a correct one or explain why forecasting is not suitable.
        - Output: key plots list, confidence intervals, top insights, and business actions.
        - Return steps to reproduce with Prophet or SARIMAX and how to choose hyperparameters.
    """)

    eda = textwrap.dedent(f"""\
        You are a senior data analyst. Do a concise yet deep EDA.
        - Suggest feature engineering for columns: numeric={numeric_cols[:8]}, datetime={datetime_cols[:4]}, text={text_cols[:4]}.
        - Provide a prioritized checklist of 10 actions to improve data quality and modeling.
    """)

    historical = textwrap.dedent(f"""\
        You are a senior analyst. Produce a historical performance review for the primary metric {y}.
        - Cohort or segment cuts (choose 2-3 best dimensions automatically).
        - Include an executive summary with 5 bullets and 5 recommended next actions.
    """)

    domain = "General tabular business data (heuristic)"
    return {"domain": domain, "forecast": forecast, "eda": eda, "historical": historical}


def _call_llm_classify(sample_payload: Dict[str, Any]) -> Dict[str, str]:
    if not HAS_OPENAI:
        raise RuntimeError("OpenAI not configured")

    client = OpenAI()
    system = (
        "You are a meticulous data analyst. "
        "Given a small tabular-data sample & schema, identify what the dataset is about, which business and what business action does it focus on "
        "in 3–8 words (domain), and craft three precise prompts to drive further analysis.\n"
        "Return ONLY strict JSON with keys: domain, forecast_prompt, eda_prompt, historical_prompt."
    )
    user = f"""Here is the dataset context as JSON:
{json.dumps(sample_payload, ensure_ascii=False, indent=2)}
Respond with JSON only."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    txt = resp.choices[0].message.content.strip()
    print("\n🧠 AI Output (raw):\n", txt, "\n")

    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:].strip()
    data = json.loads(txt)
    return {
        "domain": data.get("domain", "").strip() or "Unspecified dataset domain",
        "forecast": data.get("forecast_prompt", "").strip(),
        "eda": data.get("eda_prompt", "").strip(),
        "historical": data.get("historical_prompt", "").strip(),
    }


def _write_txt(name_prefix: str, content: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    filename = f"{name_prefix}_{ts}_{uid}.txt"
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    return f"/static/ai_prompts/{filename}"


@router.post("/nat/analyze")
def ai_nat_analyze(payload: Dict[str, Any], request: Request):

    dataset_id = (payload or {}).get("dataset_id", "default")
    df: pd.DataFrame = UPLOADED_DF.get(dataset_id)
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Upload a CSV first (dataset not found).")

    sample = _sample_for_llm(df)

    try:
        llm_out = _call_llm_classify(sample)
        if not (llm_out["forecast"] and llm_out["eda"] and llm_out["historical"]):
            raise ValueError("LLM returned incomplete prompts")
        domain = llm_out["domain"]
        prompts = {
            "forecast": llm_out["forecast"],
            "eda": llm_out["eda"],
            "historical": llm_out["historical"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI prompt generation failed: {e}")
        hp = _heuristic_prompts(df)
        domain = hp["domain"]
        prompts = {"forecast": hp["forecast"], "eda": hp["eda"], "historical": hp["historical"]}

    f_url = _write_txt("forecast_prompt", prompts["forecast"])
    e_url = _write_txt("eda_prompt", prompts["eda"])
    h_url = _write_txt("historical_prompt", prompts["historical"])

    if not hasattr(request.app.state, "cache"):
        request.app.state.cache = {}
    if dataset_id not in request.app.state.cache:
        request.app.state.cache[dataset_id] = {}

    request.app.state.cache[dataset_id]["nat_ai"] = {
        "about": domain,
        "files": {"forecast": f_url, "eda": e_url, "historical": h_url},
        "generated_at": datetime.utcnow().isoformat(),
    }

    return JSONResponse(
        {
            "about": domain,
            "files": {"forecast": f_url, "eda": e_url, "historical": h_url},
        }
    )
