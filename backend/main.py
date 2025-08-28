# --- add near your other imports ---
from dataset_ai import router as dataset_ai_router
#from analysis.compare_pairs import router as compare_router

from generateprompts import router as prompts_router

from compare_routes import router as compare_router
from aggregation import router as aggregation_router

from kpi import router as kpi_router
import outliers

from forecasting import model



# main.py
import io
import os
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException

from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ydata_profiling import ProfileReport
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import math

# EDA libs
from ydata_profiling import ProfileReport
import sweetviz as sv

from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from econml.dr import DRLearner
from econml.metalearners import XLearner
from econml.dml import CausalForestDML

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from econml.dr import DRLearner
from econml.metalearners import XLearner
from econml.dml import CausalForestDML



from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMClassifier

from fastapi import APIRouter

from analysis.slices import analyze_slices

router = APIRouter(prefix="/analyze", tags=["analyze"])


from insight import generate_ai_insights
# from causalai import generate_causal_ai_insights
from typing import List  # (you already import this lower down—keep one)
from causal_ai import generate_causal_ai_insights

from forecasting.router import router as forecast_router

from state import UPLOADED_DF


app = FastAPI(title="Analytics API", version="1.1.0")
app.state.cache = {}


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast_router)

# --- after other app.include_router(...) lines ---
app.include_router(dataset_ai_router)

app.include_router(compare_router)
app.include_router(prompts_router, prefix="/ai")
app.include_router(aggregation_router)
app.include_router(kpi_router)
app.include_router(model.router)
#---app.include_router(outliers.router, tags=["outliers"])

# ---- Simple in-memory cache ----
CACHE: Dict[str, Dict[str, Any]] = {}

# in-memory model registry

CAUSAL_MODELS: Dict[str, Dict[str, Any]] = {}

ACTIVE_MODEL_ID = "drlearner_active"


# ---- Static assets for reports ----

ASSETS_DIR = os.path.abspath("./assets")
PROFILES_DIR = os.path.join(ASSETS_DIR, "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

STATIC_DIR = os.path.abspath("./static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")



# ---------- Helpers ----------

def _coerce_time_and_value(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["ds", "date", "Date", "timestamp", "Datetime"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _infer_numeric_columns(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in df.columns.difference(num_cols):
        sample = df[c].dropna().astype(str).head(200)
        if len(sample) > 0 and (sample.str.match(r"^-?\d+(\.\d+)?$").mean() > 0.8):
            df[c] = pd.to_numeric(df[c], errors="ignore")

@app.get("/causal/latest_model")
def causal_latest_model():
    if not CAUSAL_MODELS:
        raise HTTPException(404, "No trained models yet.")
    key = list(CAUSAL_MODELS.keys())[-1]
    m = CAUSAL_MODELS[key]
    return {
        "model_id": key,
        "estimator": type(m["est"]).__name__,
        "treatment_col": m["treatment_col"],
        "outcome_col": m["outcome_col"],
        "feature_cols": m["feature_cols"],
    }


def _ensure_binary_treatment(series: pd.Series, min_per_class: int = 2) -> pd.Series:
    """
    Coerce to 0/1 and guarantee both classes have at least `min_per_class` rows.
    If not, rethreshold by quantiles; as last resort, flip a few random rows.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().nunique() <= 2 and set(s.dropna().unique()).issubset({0, 1}):
        T = s.fillna(s.median()).astype(int)
    else:
        # first try median split
        T = (s > s.median()).astype(int)

    def ok(y):
        vals, cnts = np.unique(pd.Series(y).dropna().astype(int).values, return_counts=True)
        if len(vals) < 2:
            return False
        return int(cnts.min()) >= min_per_class

    if ok(T):
        return T.astype(int)

    # try 70th percentile
    thr = s.quantile(0.70)
    T = (s > thr).astype(int)
    if ok(T):
        return T.astype(int)

    # last resort: force a few 1s so model can train (documented fallback)
    T = (s > s.median()).astype(int)
    need = max(0, min_per_class - int((T == 1).sum()))
    if need > 0:
        idx0 = np.where(T.values == 0)[0]
        if len(idx0) >= need:
            flip = np.random.default_rng(42).choice(idx0, size=need, replace=False)
            T.iloc[flip] = 1
    need0 = max(0, min_per_class - int((T == 0).sum()))
    if need0 > 0:
        idx1 = np.where(T.values == 1)[0]
        if len(idx1) >= need0:
            flip = np.random.default_rng(43).choice(idx1, size=need0, replace=False)
            T.iloc[flip] = 0

    return T.astype(int)


def _safe_train_test_split(X, T, Y, test_size=0.2, random_state=42):
    """
    Use stratify only when both classes exist and each has >=2 samples.
    """
    vals, cnts = np.unique(pd.Series(T).astype(int).values, return_counts=True)
    use_stratify = len(vals) == 2 and cnts.min() >= 2
    return train_test_split(
        X, T, Y,
        test_size=test_size,
        random_state=random_state,
        stratify=T if use_stratify else None
    )

def _ensure_binary_treatment(series: pd.Series, min_per_class: int = 4) -> pd.Series:
    """
    Coerce to 0/1 and ensure each class has at least `min_per_class` samples.
    We try median/quantile thresholds; as a last resort we deterministically flip
    a few labels to meet the minimum.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().nunique() <= 2 and set(pd.Series(s.dropna().unique()).astype(int)).issubset({0, 1}):
        T = s.fillna(s.median()).astype(int)
    else:
        T = (s > s.median()).astype(int)

    def ok(y):
        vals, cnts = np.unique(pd.Series(y).dropna().astype(int).values, return_counts=True)
        return len(vals) == 2 and int(cnts.min()) >= min_per_class

    if ok(T):
        return T.astype(int)

    # try a higher threshold
    thr = s.quantile(0.70)
    T = (s > thr).astype(int)
    if ok(T):
        return T.astype(int)

    # last resort: flip a few labels (deterministic RNG) to hit the target
    T = (s > s.median()).astype(int)
    need1 = max(0, min_per_class - int((T == 1).sum()))
    if need1 > 0:
        idx0 = np.where(T.values == 0)[0]
        if len(idx0) >= need1:
            flip = np.random.default_rng(42).choice(idx0, size=need1, replace=False)
            T.iloc[flip] = 1
    need0 = max(0, min_per_class - int((T == 0).sum()))
    if need0 > 0:
        idx1 = np.where(T.values == 1)[0]
        if len(idx1) >= need0:
            flip = np.random.default_rng(43).choice(idx1, size=need0, replace=False)
            T.iloc[flip] = 0
    return T.astype(int)


def _safe_train_test_split(X, T, Y, test_size=0.2, random_state=42, max_tries=25):
    """
    Try multiple random seeds to get a TRAIN split that still has ≥2 samples of each class.
    Use stratify when possible; if not, degrade gracefully.
    """
    T = pd.Series(T).astype(int).values
    vals, cnts = np.unique(T, return_counts=True)
    can_stratify = len(vals) == 2 and cnts.min() >= 2

    seeds = [random_state + i for i in range(max_tries)]
    for s in seeds:
        Xtr, Xte, Ttr, Tte, Ytr, Yte = train_test_split(
            X, T, Y,
            test_size=test_size,
            random_state=s,
            stratify=T if can_stratify else None,
        )
        u, c = np.unique(Ttr, return_counts=True)
        if len(u) == 2 and c.min() >= 2:
            return Xtr, Xte, Ttr, Tte, Ytr, Yte

    # last fallback: smaller test_size to keep more minority in train
    Xtr, Xte, Ttr, Tte, Ytr, Yte = train_test_split(
        X, T, Y, test_size=0.1, random_state=random_state,
        stratify=T if can_stratify else None,
    )
    return Xtr, Xte, Ttr, Tte, Ytr, Yte


def _make_ohe():
    """Works on sklearn >=1.2 (sparse_output) and older (sparse)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def _build_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    """
    Use column **indices** so it works whether X is a DataFrame or a NumPy array.
    """
    num_idx, cat_idx = [], []
    for i, c in enumerate(feature_cols):
        if pd.api.types.is_numeric_dtype(df[c]):
            num_idx.append(i)
        else:
            cat_idx.append(i)

    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_ohe()),
    ])

    transformers = []
    if num_idx:
        transformers.append(("num", num_tf, num_idx))
    if cat_idx:
        transformers.append(("cat", cat_tf, cat_idx))

    # If everything is numeric or everything is categorical, we still handle it.
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )

#-----------Causal--------------

class CausalTrainReq(BaseModel):
    dataset_id: str = "default"
    treatment_col: str
    outcome_col: str
    feature_cols: List[str]
    segment_col: Optional[str] = None
    estimator: str = "DRLearner"  # DRLearner | XLearner | CausalForestDML
    test_size: float = 0.2
    n_splits: int = 3
    calibrate: bool = False  # placeholder for future

class CausalPredictReq(BaseModel):
    model_id: str
    rows: Optional[List[Dict[str, Any]]] = None
    return_counterfactuals: bool = True

class UpliftBySegmentReq(BaseModel):
    model_id: str
    segment_col: str
    k_quantiles: int = 10

class CounterfactualCardsReq(BaseModel):
    model_id: str
    k: int = 5
    policy: str = "top_k"  # top_k | threshold
    threshold: Optional[float] = None

class CausalInsightReq(BaseModel):
    model_id: Optional[str] = None
    segment_col: Optional[str] = None
    k_cards: int = 6
    include_deciles: bool = True
    custom_prompt: Optional[str] = None


# --- AUTO TRAIN (picks reasonable cols) ---
class AutoTrainReq(BaseModel):
    dataset_id: str = "default"
    outcome_col: Optional[str] = None
    treatment_col: Optional[str] = None
    k_features: int = 12  # max features to include

class SliceReq(BaseModel):
    date_col: Optional[str] = None
    target_col: Optional[str] = None
    max_dims: int = 12
    top_k: int = 10
    top_lines: int = 3


# --- endpoint -----------------------------------------------------------------
@app.post("/causal/auto_train")
def causal_auto_train(req: AutoTrainReq):
    """
    Auto-select outcome/treatment/features and train a DRLearner using the same pipeline
    as /causal/train. Returns the model_id and chosen columns.
    """
    if req.dataset_id not in CACHE or "df" not in CACHE[req.dataset_id]:
        raise HTTPException(404, "Dataset not found. Upload first.")
    df = CACHE[req.dataset_id]["df"].copy()

    # ----- choose outcome -----
    outcome = req.outcome_col
    if not outcome:
        if "SALES" in df.columns:
            outcome = "SALES"
        else:
            nums = df.select_dtypes(include=[np.number]).columns.tolist()
            if not nums:
                raise HTTPException(400, "No numeric columns available to use as outcome.")
            outcome = nums[-1]
    if outcome not in df.columns:
        raise HTTPException(400, f"Missing outcome column: {outcome}")

    # ----- choose treatment -----
    t_col = req.treatment_col
    if not t_col:
        # prefer obvious binary promo flags
        binary_like = [
            c for c in df.columns
            if any(k in c.upper() for k in ["PROMO", "DISCOUNT", "COUPON", "TREAT", "IS_", "FLAG"])
            and df[c].dropna().nunique() <= 2
        ]
        if binary_like:
            t_col = binary_like[0]
        else:
            # fall back: strongest numeric corr with outcome
            cand = [c for c in df.select_dtypes(include=[np.number]).columns if c != outcome]
            if not cand:
                raise HTTPException(400, "No candidate columns for treatment.")
            t_col = df[cand].corrwith(pd.to_numeric(df[outcome], errors="coerce")).abs().sort_values(ascending=False).index[0]

    # ensure binary treatment with enough samples per class
    df[t_col] = _ensure_binary_treatment(df[t_col], min_per_class=2)

    # ----- features -----
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in df.columns if c not in num]
    low_card = [c for c in cat if df[c].nunique(dropna=True) <= 30]

    pool = [c for c in (num + low_card) if c not in [t_col, outcome]]
    if not pool:
        raise HTTPException(400, "No eligible feature columns found.")

    # rank numerics by abs corr to outcome; keep others after that
    ranked = []
    if num:
        corr_idx = (
            df[[c for c in num if c in pool]]
            .corrwith(pd.to_numeric(df[outcome], errors="coerce"))
            .abs()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        ranked = corr_idx + [c for c in pool if c not in corr_idx]
    else:
        ranked = pool

    feature_cols = ranked[: max(3, min(req.k_features, len(pool)))]

    # ----- train using your existing trainer (/causal/train) -----
    payload = CausalTrainReq(
        dataset_id=req.dataset_id,
        treatment_col=t_col,
        outcome_col=outcome,
        feature_cols=feature_cols,
        estimator="DRLearner",
    )

    # IMPORTANT: ensure /causal/train uses _safe_train_test_split and (optionally)
    # coerces T via _ensure_binary_treatment to avoid stratify errors.
    res = causal_train(payload)

    return {
        "model_id": res["model_id"],
        "chosen": {
            "outcome": outcome,
            "treatment": t_col,
            "features": feature_cols,
        },
        "summary": res["summary"],
        "effects": res["effects"],
    }



class SummaryPointsReq(BaseModel):
    model_id: Optional[str] = None
    k: int = 6

@app.post("/causal/summary_points")
def causal_summary_points(req: SummaryPointsReq):
    # choose model
    model_id = req.model_id or (list(CAUSAL_MODELS.keys())[-1] if CAUSAL_MODELS else None)
    if not model_id or model_id not in CAUSAL_MODELS:
        raise HTTPException(404, "No trained model found. Train via /causal/auto_train or /causal/train.")

    m = CAUSAL_MODELS[model_id]
    est, X_test, feat_cols = m["est"], m["X_test"], m["feature_cols"]
    outcome_col, treatment_col = m["outcome_col"], m["treatment_col"]

    X_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=feat_cols)
    cate = est.effect(X_df).astype(float)

    rows = []
    for c in X_df.columns:
        try:
            x = pd.to_numeric(X_df[c], errors="coerce")
            r = np.corrcoef(x.fillna(x.median()), cate)[0, 1]
            if np.isfinite(r):
                rows.append((abs(r), r, c))
        except Exception:
            continue
    rows.sort(key=lambda t: t[0], reverse=True)
    top = rows[: max(1, int(req.k))]

    messages = []
    for _, r, c in top:
        dir_word = "higher" if r > 0 else "lower"
        impact = "larger" if r > 0 else "smaller"
        messages.append(f"{c}: {dir_word} values tend to show {impact} expected uplift from treatment (corr with uplift r={r:.2f}).")

    return {
        "model_id": model_id,
        "model": {"estimator": type(est).__name__, "treatment_col": treatment_col, "outcome_col": outcome_col},
        "points": messages
    }



# ----------EDA Models ----------
class EDARequest(BaseModel):
    dataset_id: Optional[str] = "default"
    title: Optional[str] = "EDA Report"

class EDARequest(BaseModel):
    dataset_id: Optional[str] = "default"
    title: Optional[str] = "EDA Report"



# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}


from fastapi import UploadFile, File
from state import handle_uploaded_file

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()

        if file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    # Optional: preprocessing for forecast/EDA
    _coerce_time_and_value(df)
    _infer_numeric_columns(df)

    dataset_id = "default"
    CACHE[dataset_id] = {"df": df, "uploaded_at": datetime.utcnow().isoformat()}
    UPLOADED_DF[dataset_id] = df
    app.state.cache[dataset_id] = {"df": df}

    print("Upload complete")
    print("DF shape:", df.shape)
    print("CACHE keys:", CACHE.keys())
    print("UPLOADED_DF keys:", UPLOADED_DF.keys())
    print("app.state.cache keys:", app.state.cache.keys())

    return {
        "dataset_id": dataset_id,
        "rows": int(df.shape[0]),
        "cols": df.columns.tolist(),
        "message": "Upload successful",
    }

# @app.post("/upload-csv/")
# async def upload_csv(file: UploadFile = File(...)):
#     try:
#         content = await file.read()
#         df = pd.read_csv(io.BytesIO(content))
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

#     _coerce_time_and_value(df)
#     _infer_numeric_columns(df)
#     dataset_id = "default"
#     CACHE[dataset_id] = {"df": df, "uploaded_at": datetime.utcnow().isoformat()}

#     # from state import UPLOADED_DF
#     UPLOADED_DF["default"] = df

#     # Store in app.state.cache
#     app.state.cache["default"] = {"df": df}

#     return {
#         "dataset_id": dataset_id,
#         "rows": int(df.shape[0]),
#         "cols": df.columns.tolist(),
#         "message": "Upload successful",
#     }


@app.post("/eda/profile/ydata")
def eda_ydata(req: EDARequest):
    dsid = req.dataset_id or "default"
    if dsid not in CACHE or "df" not in CACHE[dsid]:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload first.")
    df = CACHE[dsid]["df"]

    # Generate report
    title = req.title or "EDA Report"
    profile = ProfileReport(
        df,
        title=title,
        minimal=False,
        explorative=True,
        correlations={"auto": {"calculate": True}},
        samples=None,
    )

    fname = f"{dsid}_ydata_{int(datetime.utcnow().timestamp())}.html"
    fpath = os.path.join(PROFILES_DIR, fname)
    profile.to_file(fpath)

    return {
        "dataset_id": dsid,
        "engine": "ydata-profiling",
        "report_url": f"/assets/profiles/{fname}",
    }






# --- add EDA endpoint ---

@app.post("/eda/profile/ydata/kpis")
def eda_ydata_kpis(req: EDARequest):
    dsid = req.dataset_id or "default"
    if dsid not in CACHE or "df" not in CACHE[dsid]:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload first.")
    df = CACHE[dsid]["df"]

    

    profile = ProfileReport(
        df,
        title=req.title or "EDA KPIs",
        minimal=False,
        explorative=True,
        correlations={"auto": {"calculate": True}},
        samples=None,
    )
    desc = profile.get_description()

    print("desc.table:", desc.table)
    print("n:", getattr(desc.table, "n", "MISSING"))
    print("n_var:", getattr(desc.table, "n_var", "MISSING"))


    # ---- dataset-level ----
    table = getattr(desc, "table", {})
    n_rows = getattr(table, "n", None)
    n_cols = getattr(table, "n_var", None)
    missing_cells_pct = (getattr(table, "p_missing", 0) or 0) * 100
    duplicate_rows_pct = (getattr(table, "p_duplicate", 0) or 0) * 100
    alerts = getattr(desc, "alerts", []) or []

    # ---- per-column ----
    vars_dict: Dict[str, Dict[str, Any]] = getattr(desc, "variables", {}) or {}

    top_missing = []
    for col, v in vars_dict.items():
        pm = v.get("p_missing")
        if isinstance(pm, (int, float)):
            top_missing.append((col, pm * 100))
    top_missing.sort(key=lambda x: x[1], reverse=True)
    top_missing = [{"name": c, "missing_pct": round(p, 2)} for c, p in top_missing[:5]]

    top_distinct = []
    for col, v in vars_dict.items():
        pdist = v.get("p_distinct")
        if isinstance(pdist, (int, float)):
            top_distinct.append((col, pdist * 100))
    top_distinct.sort(key=lambda x: x[1], reverse=True)
    top_distinct = [{"name": c, "distinct_pct": round(p, 2)} for c, p in top_distinct[:5]]

    top_skew = []
    for col, v in vars_dict.items():
        skew = v.get("skewness")
        if isinstance(skew, (int, float)) and not math.isnan(skew):
            top_skew.append((col, skew))
    top_skew.sort(key=lambda x: abs(x[1]), reverse=True)
    top_skew = [{"name": c, "skewness": round(s, 3)} for c, s in top_skew[:5]]

    # Strongly correlated pairs (Pearson, abs(r) >= 0.7)
    strong_pairs: List[Dict[str, Any]] = []
    correlations = getattr(desc, "correlations", {})
    pearson = getattr(correlations, "pearson", {})

    for a, row in pearson.items():
        for b, r in (row or {}).items():
            if a < b and isinstance(r, (int, float)):
                strong_pairs.append({"a": a, "b": b, "r": round(r, 3)})
    strong_pairs.sort(key=lambda x: abs(x["r"]), reverse=True)

    strong_pairs = [p for p in strong_pairs if abs(p["r"]) >= 0.7][:10]

    # Alerts
    alerts_list = [{"column": getattr(a, "column_name", ""),
                    "type": getattr(a, "alert_type", ""),
                    "text": getattr(a, "text", "")} for a in alerts][:20]

    # Health score
    health = 100.0
    health -= min(40.0, missing_cells_pct * 0.4)
    health -= min(20.0, duplicate_rows_pct * 0.5)
    health -= min(30.0, len(alerts) * 2.0)
    health = max(0.0, round(health, 1))

    try:
        profiling_summary = {
            "top_missing": top_missing,
            "top_distinct": top_distinct,
            "top_skew": top_skew,
            "strong_correlations": strong_pairs,
        }

        insight_text = generate_ai_insights(profiling_summary)
    except Exception as e:
        insight_text = f"⚠️ Insight generation failed: {str(e)}"

    return {
        "insights": insight_text,
        "dataset_id": dsid,
        "summary": {
            "rows": n_rows,
            "cols": n_cols,
            "missing_cells_pct": round(missing_cells_pct, 2),
            "duplicate_rows_pct": round(duplicate_rows_pct, 2),
            "health_score": health
        },
        "top_missing": top_missing,
        "top_distinct": top_distinct,
        "top_skewed": top_skew,
        "strong_correlations": strong_pairs,
        "alerts": alerts_list
    }


#---------Causal End point------------
# ---- helper: build featurizer ----
def _make_featurizer(df: pd.DataFrame, cols: List[str]) -> ColumnTransformer:
    num = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in cols if c not in num]
    transformers = []
    if num:
        transformers.append(("num", "passthrough", num))
    if cat:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers.append(("cat", enc, cat))
    return ColumnTransformer(transformers)


# ---- TRAIN ----

outcome_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
propensity_model = LGBMClassifier(n_estimators=100, max_depth=5, random_state=42)
cv = 3


def _choose_estimator(name, outcome_model, propensity_model, featurizer, cv):
    key = (name or "DRLearner").strip().lower()
    if key in ("drlearner", "dr"):
        return "DRLearner", DRLearner(
            model_regression=outcome_model,
            model_propensity=propensity_model,
            featurizer=featurizer,
            cv=cv,
        )
    elif key in ("xlearner", "x"):
        return "XLearner", XLearner(models=outcome_model, featurizer=featurizer)
    elif key in ("causalforestdml", "causalforest", "forest", "cf"):
        return "CausalForestDML", CausalForestDML(
            n_estimators=400,
            min_samples_leaf=10,
            discrete_treatment=True,
            featurizer=featurizer,
            random_state=42,
        )
    else:
        # fallback default
        return "DRLearner", DRLearner(
            model_regression=outcome_model,
            model_propensity=propensity_model,
            featurizer=featurizer,
            cv=cv,
        )



@app.post("/causal/train")
def causal_train(req: CausalTrainReq):
    # --- fetch data
    if req.dataset_id not in CACHE or "df" not in CACHE[req.dataset_id]:
        raise HTTPException(404, "Dataset not found. Upload first.")
    df = CACHE[req.dataset_id]["df"].copy()

    # --- validate columns
    needed = [req.treatment_col, req.outcome_col] + list(req.feature_cols)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing column(s): {missing}")

    # --- build X, T, Y (robust)
    T_series = _ensure_binary_treatment(df[req.treatment_col], min_per_class=4)  # helper defined earlier
    T = T_series.values.astype(int)
    Y = pd.to_numeric(df[req.outcome_col], errors="coerce").astype(float).values
    X = df[req.feature_cols].copy()

    # --- split (guarantee both classes in TRAIN when possible)
    X_train, X_test, T_train, T_test, Y_train, Y_test = _safe_train_test_split(  # helper defined earlier
        X, T, Y, test_size=req.test_size, random_state=42
    )
    outcome_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    propensity_model = LGBMClassifier(n_estimators=100, max_depth=5, random_state=42)
    cv = 3


        # inside your /causal/train handler, before building the estimator:
    est_name, est = _choose_estimator(req.estimator, outcome_model, propensity_model, featurizer, cv)
    est.fit(Y_train, T_train, X=X_train)

    # save to fixed slot
    model_id = "drlearner_active"
    CAUSAL_MODELS[model_id] = dict(
        est=est,
        featurizer=featurizer,
        X_test=X_test,
        T_test=T_test,
        Y_test=Y_test,
        feature_cols=req.feature_cols,
        treatment_col=req.treatment_col,
        outcome_col=req.outcome_col
    )

    # --- featurizer for final CATE stage (your existing helper)
    featurizer = _make_featurizer(X, req.feature_cols)

    # --- preprocessing for nuisance models
    pre = _build_preprocessor(df, req.feature_cols)

    # --- base models (try LightGBM; fall back gracefully)
    try:
        from lightgbm import LGBMRegressor, LGBMClassifier
        outcome_base = LGBMRegressor(n_estimators=200, random_state=42)
        propensity_base = LGBMClassifier(n_estimators=200, random_state=42)
    except Exception:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        outcome_base = RandomForestRegressor(n_estimators=300, random_state=42)
        propensity_base = RandomForestClassifier(n_estimators=300, random_state=42)

    # --- nuisance model pipelines (so strings like 'Shipped' are encoded)
    outcome_model = Pipeline([("pre", pre), ("model", outcome_base)])
    propensity_model = Pipeline([("pre", pre), ("model", propensity_base)])

    # --- cross-fitting CV (econml requirement: each training fold must contain both classes)
    uniq, cnts_train = np.unique(T_train, return_counts=True)
    min_class_train = int(cnts_train.min()) if len(uniq) == 2 else 0
    cv_splits = 2
    if min_class_train >= 2:
        cv_splits = min(max(2, req.n_splits), min_class_train)

    if len(uniq) == 2 and min_class_train >= 2:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=2, shuffle=True, random_state=42)

    # --- choose estimator
    est_name = (req.estimator or "DRLearner").strip()
    if est_name == "DRLearner":
        est = DRLearner(
            model_regression=outcome_model,
            model_propensity=propensity_model,
            featurizer=featurizer,
            cv=cv,
        )
    elif est_name == "XLearner":
        est = XLearner(models=outcome_model, featurizer=featurizer)
    elif est_name == "CausalForestDML":
        est = CausalForestDML(
            n_estimators=400,
            min_samples_leaf=10,
            discrete_treatment=True,
            featurizer=featurizer,
            random_state=42,
        )
    else:
        raise HTTPException(400, f"Unsupported estimator: {est_name}")

    # --- fit
    est.fit(Y_train, T_train, X=X_train)

    # --- simple overlap diagnostic (independent from the DRLearner’s internals)
    from sklearn.linear_model import LogisticRegression
    Xtr_f = featurizer.fit_transform(X_train)
    Xte_f = featurizer.transform(X_test)
    prop_clf = LogisticRegression(max_iter=1000).fit(Xtr_f, T_train)
    ps_test = prop_clf.predict_proba(Xte_f)[:, 1]
    common_support_pct = float(((ps_test >= 0.05) & (ps_test <= 0.95)).mean() * 100)
    overlap_ok = common_support_pct >= 80.0  # heuristic

    # --- effects on test
    cate_hat = est.effect(X_test).astype(float)
    ate = float(np.nanmean(cate_hat))
    att = float(np.nanmean(cate_hat[T_test == 1])) if (T_test == 1).any() else float("nan")
    atc = float(np.nanmean(cate_hat[T_test == 0])) if (T_test == 0).any() else float("nan")

    # --- stash model
    model_id = f"causal_{req.dataset_id}_{est_name.lower()}_{len(CAUSAL_MODELS)+1:03d}"
    CAUSAL_MODELS[model_id] = dict(
        est=est,
        featurizer=featurizer,
        X_test=X_test,
        T_test=T_test,
        Y_test=Y_test,
        feature_cols=req.feature_cols,
        treatment_col=req.treatment_col,
        outcome_col=req.outcome_col,
    )

    return {
        "model_id": model_id,
        "summary": {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "treatment_rate": float(np.mean(T)),
            "common_support_pct": round(common_support_pct, 2),
            "overlap_ok": bool(overlap_ok),
        },
        "effects": {
            "ATE": {"point": None if np.isnan(ate) else round(ate, 4)},
            "ATT": {"point": None if np.isnan(att) else round(att, 4)},
            "ATC": {"point": None if np.isnan(atc) else round(atc, 4)},
        },
    }


# ---- PREDICT (CATE & counterfactuals) ----
@app.post("/causal/predict")
def causal_predict(req: CausalPredictReq):
    if req.model_id not in CAUSAL_MODELS:
        raise HTTPException(404, "model_id not found")
    m = CAUSAL_MODELS[req.model_id]
    est, featurizer = m["est"], m["featurizer"]

    if req.rows is None:
        X = m["X_test"]
    else:
        Xdf = pd.DataFrame(req.rows)[m["feature_cols"]]
        X = Xdf

    cate = est.effect(X).astype(float).tolist()
    out = {"n": len(cate), "cate": [round(v, 4) for v in cate]}

    if req.return_counterfactuals:
        # econml supports const treatment vectors for effect inference
        mu1 = est.const_marginal_effect(X)  # not y1; we approximate using models when available
        # To keep it simple for now, provide only cate; y1/y0 optional:
        out.update({"y1_hat": None, "y0_hat": None})

    # uplift ranking (descending)
    idx = np.argsort(cate)[::-1].tolist()
    out["uplift_rank"] = {"indices": idx[:50], "scores": [round(float(cate[i]), 4) for i in idx[:50]]}
    return out


# ---- UPLIFT BY SEGMENT + DECILES ----
@app.post("/causal/uplift_by_segment")
def causal_uplift_by_segment(req: UpliftBySegmentReq):
    if req.model_id not in CAUSAL_MODELS:
        raise HTTPException(404, "model_id not found")
    m = CAUSAL_MODELS[req.model_id]
    est, X_test = m["est"], m["X_test"]

    # Need original df to get segment
    df = CACHE["default"]["df"]
    seg = df.loc[X_test.index, req.segment_col] if isinstance(X_test, pd.DataFrame) else df[req.segment_col].iloc[:len(X_test)]
    cate = est.effect(X_test).astype(float)

    by_seg = []
    for s, grp in pd.DataFrame({"seg": seg, "cate": cate}).groupby("seg"):
        vals = grp["cate"].values
        by_seg.append({"segment": str(s), "n": int(len(vals)), "uplift_mean": round(float(vals.mean()), 4)})

    # deciles
    qs = pd.qcut(cate, q=req.k_quantiles, labels=False, duplicates="drop")
    dec = []
    for b in sorted(pd.Series(qs).dropna().unique()):
        vals = cate[qs == b]
        dec.append({"bin": int(b + 1), "n": int(len(vals)), "uplift_avg": round(float(vals.mean()), 4)})
    return {"by_segment": by_seg, "deciles": dec}

# ---- COUNTERFACTUAL CARDS (top-K by uplift) ----
@app.post("/causal/counterfactual_cards")
def causal_counterfactual_cards(req: CounterfactualCardsReq):
    if req.model_id not in CAUSAL_MODELS:
        raise HTTPException(404, "model_id not found")
    m = CAUSAL_MODELS[req.model_id]
    est, X_test = m["est"], m["X_test"]
    feat_cols = m["feature_cols"]

    X_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=feat_cols)
    cate = est.effect(X_test).astype(float)
    order = np.argsort(cate)[::-1]

    if req.policy == "threshold" and req.threshold is not None:
        keep = [i for i in order if cate[i] >= req.threshold]
    else:
        keep = order[:req.k]

    cards = []
    for i in keep:
        features = {c: (X_df.iloc[i][c].item() if hasattr(X_df.iloc[i][c], "item") else X_df.iloc[i][c]) for c in feat_cols}
        uplift = float(cate[i])
        cards.append({
            "id": int(i),
            "features": features,
            "y0_hat": None, "y1_hat": None,  # placeholders; can be added later
            "uplift": round(uplift, 4),
            "action": "RUN PROMO" if uplift > 0 else "DO NOT TREAT",
            "expected_gain": round(max(uplift, 0.0), 4)
        })

    treat_n = len([c for c in cards if c["uplift"] > 0])
    expected_gain = round(float(sum(max(cate[i], 0.0) for i in keep)), 4)
    return {"cards": cards, "policy_summary": {"treat_n": treat_n, "expected_total_gain": expected_gain}}


# ---- AUTO CAUSAL (top 3 correlated numerical predictors) ----
from sklearn.linear_model import LinearRegression

class AutoCausalRequest(BaseModel):
    dataset_id: str = "default"
    outcome: str = "y"

@app.post("/causal/auto")
def causal_auto(req: AutoCausalRequest):
    if req.dataset_id not in CACHE or "df" not in CACHE[req.dataset_id]:
        raise HTTPException(404, detail="Dataset not found. Upload first.")

    df = CACHE[req.dataset_id]["df"].copy()
    if req.outcome not in df.columns:
        raise HTTPException(400, detail=f"Missing outcome column: {req.outcome}")

    # Select numeric features
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if req.outcome in numeric_cols:
        numeric_cols.remove(req.outcome)

    if not numeric_cols:
        raise HTTPException(400, detail="No numeric columns available for causal analysis.")

    # Calculate correlation with outcome
    corrs = df[numeric_cols].corrwith(df[req.outcome]).abs().sort_values(ascending=False)
    top3 = corrs.index.tolist()

    results = []
    for treatment in top3:
        try:
            # Fit simple linear regression as causal proxy
            model = LinearRegression()
            model.fit(df[[treatment]], df[req.outcome])
            effect = model.coef_[0]
            corr = df[treatment].corr(df[req.outcome])
            results.append({
                "treatment": treatment,
                "effect": round(effect, 4),
                "correlation": round(corr, 4)
            })
        except Exception as e:
            results.append({
                "treatment": treatment,
                "error": str(e)
            })

    return {"results": results}

@app.post("/causal/insights")
def causal_insights(req: CausalInsightReq):
    # choose model
    model_id = ACTIVE_MODEL_ID
    if model_id not in CAUSAL_MODELS:
        raise HTTPException(400, "No trained model in memory. Call /causal/train or /causal/auto_train first.")
    m = CAUSAL_MODELS[model_id]

    m = CAUSAL_MODELS[model_id]
    est = m["est"]
    X_test = m["X_test"]
    T_test = m["T_test"]
    feat_cols = m["feature_cols"]

    # dataframe view of X
    X_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=feat_cols)

    # effects
    cate = est.effect(X_df).astype(float)
    ate = float(np.nanmean(cate))
    att = float(np.nanmean(cate[T_test == 1])) if (T_test == 1).any() else float("nan")
    atc = float(np.nanmean(cate[T_test == 0])) if (T_test == 0).any() else float("nan")

    # deciles (optional)
    deciles = []
    if req.include_deciles:
        try:
            qs = pd.qcut(cate, q=10, labels=False, duplicates="drop")
            for b in sorted(pd.Series(qs).dropna().unique()):
                vals = cate[qs == b]
                deciles.append({"bin": int(b + 1), "n": int(len(vals)), "uplift_avg": round(float(vals.mean()), 4)})
        except Exception:
            deciles = []

    # segment uplift (optional)
    segment_uplift = []
    if req.segment_col:
        df_all = CACHE["default"]["df"]
        if isinstance(X_df.index, pd.RangeIndex):
            seg_series = df_all[req.segment_col].iloc[: len(X_df)]
        else:
            seg_series = df_all.loc[X_df.index, req.segment_col]
        tmp = pd.DataFrame({"seg": seg_series, "cate": cate})
        for s, grp in tmp.groupby("seg"):
            vals = grp["cate"].values
            segment_uplift.append({"segment": str(s), "n": int(len(vals)), "uplift_mean": round(float(np.nanmean(vals)), 4)})

    # top-K "cards" by uplift
    order = np.argsort(cate)[::-1]
    keep = order[: max(1, int(req.k_cards))]
    cards = []
    for i in keep:
        row = X_df.iloc[i]
        features = {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in feat_cols}
        uplift = float(cate[i])
        cards.append({
            "id": int(i),
            "features": features,
            "uplift": round(uplift, 4),
            "action": "TREAT (RUN PROMO)" if uplift > 0 else "DO NOT TREAT",
            "expected_gain": round(max(uplift, 0.0), 4),
        })

    facts = {
        "model_id": model_id,
        "estimator": type(est).__name__,
        "summary": {"n_test": int(len(X_df)), "treatment_rate_test": float(np.mean(T_test))},
        "effects": {
            "ATE": round(ate, 4),
            "ATT": None if np.isnan(att) else round(att, 4),
            "ATC": None if np.isnan(atc) else round(atc, 4),
        },
        "deciles": deciles,
        "segment_uplift": segment_uplift,
        "top_counterfactual_cards": cards,
    }

    try:
        narrative = generate_causal_ai_insights(facts, custom_prompt=req.custom_prompt)
    except Exception as e:
        narrative = f"⚠️ AI insight generation failed: {str(e)}"

    return {"model_id": model_id, "facts": facts, "insights": narrative}


# ------------
@router.post("/slices")
def analyze_slices_api(body: SliceReq):
    # adapt this to however you store the uploaded dataframe
    df2: pd.DataFrame = UPLOADED_DF.get("default")
    if df2 is None or df2.empty:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    try:
        out = analyze_slices(
            df2,
            date_col=body.date_col,
            target_col=body.target_col,
            max_dims=body.max_dims,
            top_k=body.top_k,
            top_lines=body.top_lines
        )
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Slice analysis failed: {e}")

app.include_router(router)