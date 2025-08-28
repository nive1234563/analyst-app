# # kpi.py
# from fastapi import APIRouter, UploadFile, File
# from pydantic import BaseModel
# from typing import Dict, Any, List, Tuple, Union, Optional
# import numpy as np
# import pandas as pd
# import io, re, os, json

# router = APIRouter(prefix="/kpi", tags=["kpi"])

# # ========================= In-memory dataset =========================
# KPI_DF: Optional[pd.DataFrame] = None

# def _set_df(df: pd.DataFrame) -> None:
#     global KPI_DF
#     KPI_DF = df

# def _get_df() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
#     global KPI_DF
#     if KPI_DF is None or KPI_DF.empty:
#         return None, "No dataset loaded. Upload a CSV in this tab or POST /kpi/set first."
#     return KPI_DF, None

# # ============================= Models ===============================
# class StartReq(BaseModel):
#     dataset_id: Optional[str] = None  # parity; not used

# class SetRowsReq(BaseModel):
#     # Either 'rows' (list of dicts) OR 'data' (dict of lists/scalars)
#     rows: Optional[List[Dict[str, Any]]] = None
#     data: Optional[Dict[str, Any]] = None

# # =========================== Cleaning ===============================
# def clean_df(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Domain-agnostic cleaning:
#       1) drop columns that are entirely NaN/null
#       2) drop numeric columns that are entirely zeros (after coercion)
#     """
#     if df is None or df.empty:
#         return df
#     df = df.copy()

#     # 1) drop all-null columns
#     df = df.dropna(axis=1, how="all")

#     # 2) drop numeric all-zero columns
#     zero_cols = []
#     for c in df.columns:
#         s = pd.to_numeric(df[c], errors="coerce")
#         if s.notna().sum() > 0 and (s.fillna(0) == 0).all():
#             zero_cols.append(c)
#     if zero_cols:
#         df = df.drop(columns=zero_cols)

#     return df

# # ========================= Load helpers =============================
# def _smart_df_from_payload(
#     rows: Optional[List[Dict[str, Any]]],
#     data: Optional[Dict[str, Any]],
# ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
#     """
#     Accepts:
#       - rows: list[dict]               -> DataFrame(rows)
#       - data: dict of lists            -> DataFrame(data) (auto-broadcast scalars)
#       - data: dict of scalars          -> DataFrame([data])  (single row)
#     Returns (df, error).
#     """
#     if rows is not None:
#         if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
#             return None, "rows must be a list of objects"
#         if len(rows) == 0:
#             return None, "rows is empty"
#         return pd.DataFrame(rows), None

#     if data is not None:
#         if not isinstance(data, dict):
#             return None, "data must be an object"
#         # dict with any lists → make all values list-like of same length
#         if any(isinstance(v, list) for v in data.values()):
#             lens = {len(v) for v in data.values() if isinstance(v, list)}
#             if len(lens) > 1:
#                 return None, "data lists have different lengths"
#             n = next(iter(lens)) if lens else 1
#             fixed = {k: (v if isinstance(v, list) else [v] * n) for k, v in data.items()}
#             return pd.DataFrame(fixed), None
#         # dict of scalars → single row
#         return pd.DataFrame([data]), None

#     return None, "Provide either rows (list of objects) or data (object)."

# # ============================ AI helpers ============================
# def _has_openai() -> bool:
#     return bool(os.getenv("OPENAI_API_KEY"))

# def _ai_json(prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
#     """
#     Ask AI to output ONLY valid JSON. If parsing fails or key missing, return {}.
#     """
#     if not _has_openai():
#         return {}
#     try:
#         from openai import OpenAI
#         client = OpenAI()
#         resp = client.chat.completions.create(
#             model="gpt-4o-mini",
#             temperature=temperature,
#             messages=[
#                 {"role": "system", "content": "You are a meticulous data analysis planner. Output ONLY strict JSON."},
#                 {"role": "user", "content": prompt},
#             ],
#         )
#         txt = (resp.choices[0].message.content or "").strip()
#         if txt.startswith("```"):  # strip ```json fences if present
#             txt = txt.strip("`")
#             nl = txt.find("\n")
#             if nl != -1:
#                 txt = txt[nl + 1 :]
#         return json.loads(txt)
#     except Exception as e:
#         print("AI JSON parse failed:", e)
#         return {}

# def _kpi_prompt(df: pd.DataFrame) -> str:
#     sample = df.head(3).to_dict(orient="records")
#     schema = {c: str(df[c].dtype) for c in df.columns}
#     return f"""
# You are given a tabular dataset. Suggest up to 12 KPI cards that are broadly useful (domain-agnostic).
# Each KPI must include:
#   - "name": short display name (<= 60 chars)
#   - "column": exact column name to aggregate ("" allowed only for "count")
#   - "aggregation": one of ["sum","avg","mean","min","max","median","std","count","distinct_count"]

# Rules:
# - Use numeric aggregations (sum/avg/min/max/median/std) only on numeric-like columns.
# - "count" means row count and can have "column": "".
# - "distinct_count" can be used on any column to count unique values.
# - Prefer primary quantities (totals/averages) and simple uniqueness metrics when appropriate.

# Return STRICT JSON: {{ "kpis": [{{"name":"...","column":"...","aggregation":"..."}}, ...] }}

# COLUMNS (with inferred dtypes):
# {json.dumps(schema, indent=2)}

# FIRST 3 ROWS:
# {json.dumps(sample, indent=2)}
# """.strip()

# # ===================== KPI computation helpers =====================
# _ALLOWED_FUNCS = {
#     "sum": "sum", "avg": "mean", "average": "mean", "mean": "mean",
#     "min": "min", "max": "max", "median": "median",
#     "std": "std", "stdev": "std", "stddev": "std",
#     "count": "count",
#     "distinct_count": "nunique", "nunique": "nunique", "unique": "nunique",
# }

# def _is_numeric(s: pd.Series) -> bool:
#     return pd.api.types.is_numeric_dtype(s)

# # strip common symbols so "₹1,23,456", "$1,234.50", "12.5%", "  7 " all coerce
# _CURRENCY = re.compile(r"[₹$€£,%\s]")

# def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
#     if df is None or col not in df.columns:
#         return pd.Series([], dtype="float64")
#     s = df[col].astype(str).str.replace(",", "", regex=False)
#     s = s.apply(lambda x: _CURRENCY.sub("", x))
#     return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

# def _compute_single_kpi(df: pd.DataFrame, col: str, func_key: str) -> Union[float, int, None]:
#     func = _ALLOWED_FUNCS.get((func_key or "").lower().strip())
#     if not func:
#         return None
#     if func in {"sum", "mean", "min", "max", "median", "std"}:
#         series = _to_numeric_series(df, col)
#         if series.notna().sum() == 0:
#             return None
#         return getattr(series, func)()
#     if func == "count":
#         return 0 if df is None else int(len(df))
#     if func == "nunique":
#         if df is None or col not in df.columns:
#             return None
#         return int(df[col].astype(str).nunique(dropna=True))
#     return None

# def _compute_kpi_with_reason(df: Optional[pd.DataFrame], k: Dict[str, Any]):
#     name = str(k.get("name", "KPI")).strip()[:60] or "KPI"
#     col  = str(k.get("column", "")).strip()
#     agg  = str(k.get("aggregation", "")).strip().lower()
#     if agg not in _ALLOWED_FUNCS:
#         return None, f"unknown agg '{agg}'"
#     val = _compute_single_kpi(df, col, agg)
#     if val is None:
#         miss = "missing col" if (df is not None and col and col not in (df.columns if hasattr(df, "columns") else [])) else "no numeric values"
#         return None, f"cannot compute {agg} on '{col}' ({miss})"
#     if isinstance(val, (float, np.floating)): out = float(val)
#     elif isinstance(val, (int, np.integer)):  out = int(val)
#     else:
#         try: out = float(val)
#         except: return None, "non-castable value"
#     return {"name": name, "value": out}, None

# def _fallback_kpi_defs(df: Optional[pd.DataFrame]) -> List[dict]:
#     """
#     Domain-neutral fallback: numeric totals/averages + counts/uniques.
#     """
#     cols = [] if df is None else list(df.columns)
#     nums = [c for c in cols if df is not None and _is_numeric(df[c])]
#     cats = [c for c in cols if df is not None and not _is_numeric(df[c])]
#     defs: List[dict] = []

#     if nums:
#         first = nums[0]
#         defs += [
#             {"name": f"Total {first}", "column": first, "aggregation": "sum"},
#             {"name": f"Avg {first}",   "column": first, "aggregation": "mean"},
#             {"name": f"Max {first}",   "column": first, "aggregation": "max"},
#             {"name": f"Min {first}",   "column": first, "aggregation": "min"},
#         ]

#     defs.append({"name": "Row Count", "column": "", "aggregation": "count"})
#     if cats:
#         defs.append({"name": f"Distinct {cats[0]}", "column": cats[0], "aggregation": "distinct_count"})

#     for c in nums[1:]:
#         if len(defs) >= 12: break
#         defs.append({"name": f"Avg {c}", "column": c, "aggregation": "mean"})

#     # de-dup
#     seen=set(); out=[]
#     for d in defs:
#         key=(d["name"], d["column"], d["aggregation"])
#         if key not in seen:
#             seen.add(key); out.append(d)
#     return out[:12]

# # ============================== Routes ==============================
# @router.post("/upload-csv")
# async def kpi_upload_csv(file: UploadFile = File(...)):
#     """
#     Upload a CSV/Excel just for the KPI tab (independent of other routes).
#     """
#     content = await file.read()
#     df = None
#     try:
#         df = pd.read_csv(io.BytesIO(content))
#     except Exception:
#         try:
#             df = pd.read_excel(io.BytesIO(content))
#         except Exception as e:
#             return {"ok": False, "error": f"Failed to parse file: {e}"}

#     if df is None or df.empty:
#         return {"ok": False, "error": "File parsed but no rows found."}

#     df = clean_df(df)
#     _set_df(df)
#     return {"ok": True, "rows": int(len(df)), "cols": df.columns.tolist()}

# @router.post("/set")
# def kpi_set(req: SetRowsReq):
#     """
#     Load data from JSON:
#       - rows: list[dict]
#       - data: dict (lists or scalars)
#     """
#     df, err = _smart_df_from_payload(req.rows, req.data)
#     if err:
#         return {"ok": False, "error": err}
#     if df is None or df.empty:
#         return {"ok": False, "error": "Provided data is empty."}
#     df = clean_df(df)
#     _set_df(df)
#     return {"ok": True, "rows": int(len(df)), "cols": df.columns.tolist()}

# @router.post("/start")
# def kpi_start(req: StartReq):
#     """
#     Domain-agnostic KPI computation:
#       1) Clean df (drop all-null + all-zero numeric cols)
#       2) Ask AI for KPI defs from headers + first 3 rows
#       3) Validate & compute KPIs
#       4) Fallback + pad to exactly 8 KPI tiles
#     Always returns 200 (with placeholders if needed).
#     """
#     df, warn = _get_df()
#     error_msg = warn or None

#     if df is not None:
#         df = clean_df(df)

#     # --- AI-planned KPI definitions ---
#     kpi_defs: List[Dict[str, Any]] = []
#     if df is not None and _has_openai() and len(df.columns) > 0:
#         prompt = _kpi_prompt(df)
#         ai_kpis = _ai_json(prompt)
#         print("AI response for KPIs:", ai_kpis)  # debug
#         if isinstance(ai_kpis, dict) and isinstance(ai_kpis.get("kpis"), list):
#             for k in ai_kpis["kpis"]:
#                 try:
#                     name = str(k.get("name", "")).strip()[:60] or "KPI"
#                     col  = str(k.get("column", "")).strip()
#                     agg  = str(k.get("aggregation", "")).strip().lower()
#                     if agg in _ALLOWED_FUNCS:
#                         # permit empty column ONLY for count
#                         if agg == "count":
#                             kpi_defs.append({"name": name, "column": "", "aggregation": "count"})
#                         elif col in df.columns:
#                             kpi_defs.append({"name": name, "column": col, "aggregation": agg})
#                 except Exception:
#                     continue

#     # --- fallback if AI absent/invalid ---
#     if not kpi_defs:
#         kpi_defs = _fallback_kpi_defs(df)

#     # --- compute values; collect reasons for skipped items ---
#     aggregations: List[Dict[str, Any]] = []
#     kpi_debug: List[Dict[str, Any]] = []
#     for k in kpi_defs[:12]:
#         out, reason = _compute_kpi_with_reason(df, k)
#         if out is not None:
#             aggregations.append(out)
#         else:
#             kpi_debug.append({"requested": k, "skip_reason": reason})

#     # --- top up to 8 using fallback defs if AI ideas were not fully computable ---
#     if len(aggregations) < 8:
#         for k in _fallback_kpi_defs(df):
#             if len(aggregations) >= 8: break
#             out, reason = _compute_kpi_with_reason(df, k)
#             if out is not None and all(a["name"] != out["name"] for a in aggregations):
#                 aggregations.append(out)
#             elif reason:
#                 kpi_debug.append({"requested": k, "skip_reason": reason})

#     # --- pad with placeholders if still short (e.g., empty df) ---
#     while len(aggregations) < 8:
#         aggregations.append({"name": f"KPI {len(aggregations)+1}", "value": 0})

#     return {
#         "aggregations": aggregations[:8],
#         "kpi_debug": kpi_debug,
#         "error": error_msg,
#     }

# kpi.py

from state import UPLOADED_DF
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
import io, re, os, json

router = APIRouter(prefix="/kpi", tags=["kpi"])

# ========================= In-memory dataset =========================
KPI_DF: Optional[pd.DataFrame] = None

def _set_df(df: pd.DataFrame) -> None:
    global KPI_DF
    KPI_DF = df

def _get_df() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    global KPI_DF
    if KPI_DF is None or KPI_DF.empty:
        return None, "No dataset loaded. Upload a CSV in this tab or POST /kpi/set first."
    return KPI_DF, None

# ============================= Models ===============================
class StartReq(BaseModel):
    dataset_id: Optional[str] = None  # parity; not used

class StartSelectedReq(BaseModel):
    # compute KPIs using only these columns (intersection with real df)
    columns: List[str]
    dataset_id: Optional[str] = None  # parity; not used

class SetRowsReq(BaseModel):
    # Either 'rows' (list of dicts) OR 'data' (dict of lists/scalars)
    rows: Optional[List[Dict[str, Any]]] = None
    data: Optional[Dict[str, Any]] = None

# =========================== Cleaning ===============================
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-agnostic cleaning:
      1) drop columns that are entirely NaN/null
      2) drop numeric columns that are entirely zeros (after coercion)
    """
    if df is None or df.empty:
        return df
    df = df.copy()

    # 1) drop all-null columns
    df = df.dropna(axis=1, how="all")

    # 2) drop numeric all-zero columns
    zero_cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0 and (s.fillna(0) == 0).all():
            zero_cols.append(c)
    if zero_cols:
        df = df.drop(columns=zero_cols)

    return df

def subset_columns(df: Optional[pd.DataFrame], cols: List[str]) -> Optional[pd.DataFrame]:
    """
    Keep only the requested columns that truly exist in df.
    If nothing intersects, return an empty DataFrame with those names (so we
    can still compute 'count' KPIs and return helpful debug info).
    """
    if df is None:
        return None
    if not cols:
        return df.copy()
    valid = [c for c in cols if c in df.columns]
    if valid:
        return df[valid].copy()
    return pd.DataFrame(index=df.index)  # no intersection

# ========================= Load helpers =============================
def _smart_df_from_payload(
    rows: Optional[List[Dict[str, Any]]],
    data: Optional[Dict[str, Any]],
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Accepts:
      - rows: list[dict]               -> DataFrame(rows)
      - data: dict of lists            -> DataFrame(data) (auto-broadcast scalars)
      - data: dict of scalars          -> DataFrame([data])  (single row)
    Returns (df, error).
    """
    if rows is not None:
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            return None, "rows must be a list of objects"
        if len(rows) == 0:
            return None, "rows is empty"
        return pd.DataFrame(rows), None

    if data is not None:
        if not isinstance(data, dict):
            return None, "data must be an object"
        # dict with any lists → make all values list-like of same length
        if any(isinstance(v, list) for v in data.values()):
            lens = {len(v) for v in data.values() if isinstance(v, list)}
            if len(lens) > 1:
                return None, "data lists have different lengths"
            n = next(iter(lens)) if lens else 1
            fixed = {k: (v if isinstance(v, list) else [v] * n) for k, v in data.items()}
            return pd.DataFrame(fixed), None
        # dict of scalars → single row
        return pd.DataFrame([data]), None

    return None, "Provide either rows (list of objects) or data (object)."

# ============================ AI helpers ============================
def _has_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

def _ai_json(prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
    """
    Ask AI to output ONLY valid JSON. If parsing fails or key missing, return {}.
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
                {"role": "system", "content": "You are a meticulous data analysis planner. Output ONLY strict JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        txt = (resp.choices[0].message.content or "").strip()
        if txt.startswith("```"):  # strip ```json fences if present
            txt = txt.strip("`")
            nl = txt.find("\n")
            if nl != -1:
                txt = txt[nl + 1 :]
        return json.loads(txt)
    except Exception as e:
        print("AI JSON parse failed:", e)
        return {}

def _kpi_prompt(df: pd.DataFrame) -> str:
    sample = df.head(3).to_dict(orient="records")
    schema = {c: str(df[c].dtype) for c in df.columns}
    return f"""
You are given a tabular dataset. Suggest up to 8 KPI cards that are broadly useful (domain-agnostic) but only important ones.
Each KPI must include:
  - "name": short display name (<= 60 chars)
  - "column": exact column name to aggregate ("" allowed only for "count")
  - "aggregation": one of ["sum","avg","mean","min","max","median","std","count","distinct_count"]

Rules:
- Use numeric aggregations (sum/avg/min/max/median/std) only on numeric-like columns.
- "count" means row count and can have "column": "".
- "distinct_count" can be used on any column to count unique values.
- Prefer primary quantities (totals/averages) and simple uniqueness metrics when appropriate.

Return STRICT JSON: {{ "kpis": [{{"name":"...","column":"...","aggregation":"..."}}, ...] }}

COLUMNS (with inferred dtypes):
{json.dumps(schema, indent=2)}

FIRST 3 ROWS:
{json.dumps(sample, indent=2)}
""".strip()

# ===================== KPI computation helpers =====================
_ALLOWED_FUNCS = {
    "sum": "sum", "avg": "mean", "average": "mean", "mean": "mean",
    "min": "min", "max": "max", "median": "median",
    "std": "std", "stdev": "std", "stddev": "std",
    "count": "count",
    "distinct_count": "nunique", "nunique": "nunique", "unique": "nunique",
}

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

# strip common symbols so "₹1,23,456", "$1,234.50", "12.5%", "  7 " all coerce
_CURRENCY = re.compile(r"[₹$€£,%\s]")

def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or col not in df.columns:
        return pd.Series([], dtype="float64")
    s = df[col].astype(str).str.replace(",", "", regex=False)
    s = s.apply(lambda x: _CURRENCY.sub("", x))
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def _compute_single_kpi(df: pd.DataFrame, col: str, func_key: str) -> Union[float, int, None]:
    func = _ALLOWED_FUNCS.get((func_key or "").lower().strip())
    if not func:
        return None
    if func in {"sum", "mean", "min", "max", "median", "std"}:
        series = _to_numeric_series(df, col)
        if series.notna().sum() == 0:
            return None
        return getattr(series, func)()
    if func == "count":
        return 0 if df is None else int(len(df))
    if func == "nunique":
        if df is None or col not in df.columns:
            return None
        return int(df[col].astype(str).nunique(dropna=True))
    return None

def _compute_kpi_with_reason(df: Optional[pd.DataFrame], k: Dict[str, Any]):
    name = str(k.get("name", "KPI")).strip()[:60] or "KPI"
    col  = str(k.get("column", "")).strip()
    agg  = str(k.get("aggregation", "")).strip().lower()
    if agg not in _ALLOWED_FUNCS:
        return None, f"unknown agg '{agg}'"
    val = _compute_single_kpi(df, col, agg)
    if val is None:
        miss = "missing col" if (df is not None and col and col not in (df.columns if hasattr(df, "columns") else [])) else "no numeric values"
        return None, f"cannot compute {agg} on '{col}' ({miss})"
    if isinstance(val, (float, np.floating)): out = float(val)
    elif isinstance(val, (int, np.integer)):  out = int(val)
    else:
        try: out = float(val)
        except: return None, "non-castable value"
    return {"name": name, "value": out}, None

def _fallback_kpi_defs(df: Optional[pd.DataFrame]) -> List[dict]:
    """
    Domain-neutral fallback: numeric totals/averages + counts/uniques.
    """
    cols = [] if df is None else list(df.columns)
    nums = [c for c in cols if df is not None and _is_numeric(df[c])]
    cats = [c for c in cols if df is not None and not _is_numeric(df[c])]
    defs: List[dict] = []

    if nums:
        first = nums[0]
        defs += [
            {"name": f"Total {first}", "column": first, "aggregation": "sum"},
            {"name": f"Avg {first}",   "column": first, "aggregation": "mean"},
            {"name": f"Max {first}",   "column": first, "aggregation": "max"},
            {"name": f"Min {first}",   "column": first, "aggregation": "min"},
        ]

    defs.append({"name": "Row Count", "column": "", "aggregation": "count"})
    if cats:
        defs.append({"name": f"Distinct {cats[0]}", "column": cats[0], "aggregation": "distinct_count"})

    for c in nums[1:]:
        if len(defs) >= 12: break
        defs.append({"name": f"Avg {c}", "column": c, "aggregation": "mean"})

    # de-dup
    seen=set(); out=[]
    for d in defs:
        key=(d["name"], d["column"], d["aggregation"])
        if key not in seen:
            seen.add(key); out.append(d)
    return out[:12]

# ============================== Routes ==============================
from fastapi import UploadFile, File, Request
from state import UPLOADED_DF

# @router.post("/upload-csv")
# async def kpi_upload_csv(file: UploadFile = File(...), request: Request = None):
#     """
#     Upload a CSV/Excel just for the KPI tab (independent of other routes).
#     Also updates UPLOADED_DF and app.state.cache["default"] so forecasting can access it.
#     """
#     content = await file.read()
#     df = None
#     try:
#         df = pd.read_csv(io.BytesIO(content))
#     except Exception:
#         try:
#             df = pd.read_excel(io.BytesIO(content))
#         except Exception as e:
#             return {"ok": False, "error": f"Failed to parse file: {e}"}

#     if df is None or df.empty:
#         return {"ok": False, "error": "File parsed but no rows found."}

#     df = clean_df(df)
#     _set_df(df)

#     # ✅ Update global UPLOADED_DF used by model.py and ai.py
#     UPLOADED_DF["default"] = df

#     # ✅ Also set FastAPI app.state.cache so main.py or other modules can see it
#     if request is not None:
#         request.app.state.cache["default"] = {"df": df}

#     return {"ok": True, "rows": int(len(df)), "cols": df.columns.tolist()}


@router.post("/upload-csv")
async def kpi_upload_csv(file: UploadFile = File(...), request: Request = None):
    """
    Upload a CSV/Excel just for the KPI tab.
    Also updates UPLOADED_DF and app.state.cache so forecasting modules work.
    """
    content = await file.read()
    df = None

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        try:
            df = pd.read_excel(io.BytesIO(content))
        except Exception as e:
            return {"ok": False, "error": f"Failed to parse file: {e}"}

    if df is None or df.empty:
        return {"ok": False, "error": "File parsed but no rows found."}

    # KPI-specific preparation
    df = clean_df(df)
    _set_df(df)

    # ✅ Now also make it accessible to forecasting
    UPLOADED_DF["default"] = df
    if request is not None:
        request.app.state.cache["default"] = {"df": df}

    return {"ok": True, "rows": int(len(df)), "cols": df.columns.tolist()}

@router.post("/set")
def kpi_set(req: SetRowsReq):
    """
    Load data from JSON:
      - rows: list[dict]
      - data: dict (lists or scalars)
    """
    df, err = _smart_df_from_payload(req.rows, req.data)
    if err:
        return {"ok": False, "error": err}
    if df is None or df.empty:
        return {"ok": False, "error": "Provided data is empty."}
    df = clean_df(df)
    _set_df(df)

    try:
        
        UPLOADED_DF["default"] = df

    except Exception as e:
        print("Warning: could not sync to uploaded_df:", e)

    return {"ok": True, "rows": int(len(df)), "cols": df.columns.tolist()}

def _compute_kpis_pipeline(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Shared core used by both /start and /start-selected.
    """
    error_msg = None
    if df is None or df.empty:
        return {
            "aggregations": [{"name": f"KPI {i+1}", "value": 0} for i in range(8)],
            "kpi_debug": [],
            "error": "No dataset loaded or empty after filtering.",
        }

    # Clean again (idempotent)
    df = clean_df(df)

    # --- AI-planned KPI definitions ---
    kpi_defs: List[Dict[str, Any]] = []
    if _has_openai() and len(df.columns) > 0:
        prompt = _kpi_prompt(df)
        ai_kpis = _ai_json(prompt)
        print("AI response for KPIs:", ai_kpis)  # debug
        if isinstance(ai_kpis, dict) and isinstance(ai_kpis.get("kpis"), list):
            for k in ai_kpis["kpis"]:
                try:
                    name = str(k.get("name", "")).strip()[:60] or "KPI"
                    col  = str(k.get("column", "")).strip()
                    agg  = str(k.get("aggregation", "")).strip().lower()
                    if agg in _ALLOWED_FUNCS:
                        if agg == "count":
                            kpi_defs.append({"name": name, "column": "", "aggregation": "count"})
                        elif col in df.columns:
                            kpi_defs.append({"name": name, "column": col, "aggregation": agg})
                except Exception:
                    continue

    # --- fallback if AI absent/invalid ---
    if not kpi_defs:
        kpi_defs = _fallback_kpi_defs(df)

    # --- compute values; collect reasons for skipped items ---
    aggregations: List[Dict[str, Any]] = []
    kpi_debug: List[Dict[str, Any]] = []
    for k in kpi_defs[:12]:
        out, reason = _compute_kpi_with_reason(df, k)
        if out is not None:
            aggregations.append(out)
        else:
            kpi_debug.append({"requested": k, "skip_reason": reason})

    # --- top up to 8 using fallback defs if AI ideas were not fully computable ---
    if len(aggregations) < 8:
        for k in _fallback_kpi_defs(df):
            if len(aggregations) >= 8: break
            out, reason = _compute_kpi_with_reason(df, k)
            if out is not None and all(a["name"] != out["name"] for a in aggregations):
                aggregations.append(out)
            elif reason:
                kpi_debug.append({"requested": k, "skip_reason": reason})

    # --- pad with placeholders if still short ---
    if len(aggregations) < 4:
        for k in _fallback_kpi_defs(df):
            if len(aggregations) >= 4:
                break
            out, reason = _compute_kpi_with_reason(df, k)
            if out is not None and all(a["name"] != out["name"] for a in aggregations):
                aggregations.append(out)

    # return only the actual important KPIs (4–8 max)
    return {
        "aggregations": aggregations[:8],
        "kpi_debug": kpi_debug,
        "error": error_msg,
    }

@router.post("/start")
def kpi_start(req: StartReq):
    """
    Original KPI computation over the entire uploaded dataset.
    """
    df, warn = _get_df()
    if warn:
        return {
            "aggregations": [{"name": f"KPI {i+1}", "value": 0} for i in range(8)],
            "kpi_debug": [],
            "error": warn,
        }
    return _compute_kpis_pipeline(df)

@router.post("/start-selected")
def kpi_start_selected(req: StartSelectedReq):
    """
    KPI computation restricted to the user-selected columns.
    - If no columns provided, behaves like /start.
    - If columns don't intersect, returns row-count KPIs where possible and
      debug info for others.
    """
    df, warn = _get_df()
    if warn:
        return {
            "aggregations": [{"name": f"KPI {i+1}", "value": 0} for i in range(8)],
            "kpi_debug": [],
            "error": warn,
        }

    # subset to requested columns (intersection handled inside)
    filtered = subset_columns(df, req.columns or [])
    return _compute_kpis_pipeline(filtered)
