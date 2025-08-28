from state import UPLOADED_DF

# model.py
from forecast_schema import get_schema
from transforms import apply_modifications, run_custom_transform_if_any

from fastapi import Request  # new

import re


# backend/forecasting/model.py
import io
import json
import math
import base64
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def load_default_dataset_from_disk(path="uploaded/default.csv"):
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"No uploaded file found at {path}")
    df = pd.read_csv(path)
    if "ds" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "ds"})
    if "y" not in df.columns:
        for col in ["sales", "value", "target","price"]:
            if col in df.columns:
                df = df.rename(columns={col: "y"})
                break
    _DATASETS["default"] = df



# Optional imports
try:
    from prophet import Prophet  # new package name
except Exception:
    try:
        from prophet import Prophet  # legacy
    except Exception:
        Prophet = None

try:
    import pmdarima as pm
except Exception:
    pm = None

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -------------------- In-memory stores --------------------
_DATASETS: Dict[str, pd.DataFrame] = {}
_MODELS: Dict[str, dict] = {}  # {dataset_id: {"model": obj, "kind": "prophet|arima|ml", "freq": "D", ...}}
_CACHE_PLOTS: Dict[Tuple[str, str], str] = {}  # (dataset_id, kind) -> base64 PNG
_LAST_FCSTS: Dict[str, pd.DataFrame] = {}      # dataset_id -> forecast df (ds, yhat)


# -------------------- Utilities --------------------
def _to_base64_png():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _infer_freq(idx: pd.DatetimeIndex) -> str:
    freq = pd.infer_freq(idx)
    if freq:
        return "D" if freq.startswith("D") else ("W" if freq.startswith("W") else ("M" if freq.startswith("M") else "D"))
    # fallback: median diff
    diffs = np.diff(idx.view("i8")) / 86_400_000_000_000  # days
    med = np.median(diffs) if len(diffs) else 1.0
    if med <= 1.5:
        return "D"
    if med <= 10:
        return "W"
    return "M"

def _resample_df(df, freq: str, target_col: str) -> pd.DataFrame:
    df = df.sort_values("ds")
    df = df.set_index("ds")
    rule = {"D": "D", "W": "W", "M": "MS"}[freq]
    # For sales/metrics, summation is common when going to coarser frequency; daily keeps values.
    agg = "sum" if rule in ["W", "MS"] else "sum"
    out = df[[target_col]].resample(rule).agg(agg).rename(columns={target_col: "y"})
    out["y"] = out["y"].astype(float).fillna(0.0)
    out.reset_index(inplace=True)
    out.rename(columns={"index": "ds"}, inplace=True)
    return out

def _smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1e-9
    return np.mean(200.0 * np.abs(y_pred - y_true) / denom)

def _rolling_splits(n: int, horizon: int, folds: int) -> List[Tuple[int, int]]:
    """
    Returns list of (train_end_index_exclusive, test_end_index_exclusive)
    Train always starts at 0, expands each fold, tests exactly 'horizon' points.
    """
    splits = []
    for i in range(folds, 0, -1):
        test_end = n - (i - 1) * horizon
        train_end = test_end - horizon
        if train_end <= max(10, horizon):  # ensure enough history
            continue
        splits.append((train_end, test_end))
    return splits

def _date_features(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    z["ds"] = pd.to_datetime(z["ds"])
    z["year"] = z["ds"].dt.year
    z["month"] = z["ds"].dt.month
    z["day"] = z["ds"].dt.day
    z["dow"] = z["ds"].dt.dayofweek
    z["week"] = z["ds"].dt.isocalendar().week.astype(int)
    # cyclic encodings
    z["sin_month"] = np.sin(2 * np.pi * z["month"] / 12)
    z["cos_month"] = np.cos(2 * np.pi * z["month"] / 12)
    z["sin_dow"]   = np.sin(2 * np.pi * z["dow"] / 7)
    z["cos_dow"]   = np.cos(2 * np.pi * z["dow"] / 7)
    return z

def _add_lags(df: pd.DataFrame, lags=(1,2,7,14,28), roll=(7,14,28)) -> pd.DataFrame:
    z = df.copy()
    for L in lags:
        z[f"lag_{L}"] = z["y"].shift(L)
    for r in roll:
        z[f"rmean_{r}"] = z["y"].shift(1).rolling(r).mean()
    return z

# -------------------- Candidate models --------------------
def _fit_prophet(train: pd.DataFrame):
    if Prophet is None:
        return None
    m = Prophet(
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.1
    )
    m.fit(train.rename(columns={"ds": "ds", "y": "y"}))
    return m

def _forecast_prophet(model, future_df: pd.DataFrame, horizon: int, freq: str) -> pd.DataFrame:
    future = pd.DataFrame({"ds": pd.date_range(start=future_df["ds"].iloc[-1] + pd.tseries.frequencies.to_offset(freq),
                                               periods=horizon, freq=freq)})
    fcst = model.predict(future)
    return fcst[["ds", "yhat"]]

def _fit_arima(train: pd.DataFrame):
    if pm is None:
        return None
    y = train["y"].values
    model = pm.auto_arima(y, seasonal=True, m=_guess_m(train["ds"]))
    return model

def _forecast_arima(model, last_ds: pd.Timestamp, horizon: int, freq: str) -> pd.DataFrame:
    yhat = model.predict(horizon)
    future_ds = pd.date_range(start=last_ds + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)
    return pd.DataFrame({"ds": future_ds, "yhat": yhat})

def _guess_m(ds: pd.Series) -> int:
    # seasonal period guess from inferred frequency
    freq = _infer_freq(pd.DatetimeIndex(ds))
    return 7 if freq == "D" else (52 if freq == "W" else 12)

def _fit_ml(train: pd.DataFrame):
    # Feature build
    z = _date_features(train)
    z = _add_lags(z)
    z = z.dropna().reset_index(drop=True)
    X = z.drop(columns=["y", "ds"])
    y = z["y"].values
    if len(z) < 50:
        # too short for ML
        return None, None
    if HAS_XGB:
        model = XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=42
        )
    else:
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    model.fit(X, y)
    return model, z.columns.tolist()

def _forecast_ml(model, cols: List[str], hist: pd.DataFrame, horizon: int, freq: str) -> pd.DataFrame:
    # recursive forecast
    df_hist = hist.copy()
    last_date = df_hist["ds"].max()
    preds = []
    cur = df_hist.copy()
    for step in range(1, horizon + 1):
        next_ds = last_date + pd.tseries.frequencies.to_offset(freq) * step
        tmp = pd.DataFrame({"ds": [next_ds]})
        tmp = pd.concat([cur[["ds", "y"]], tmp], ignore_index=True)
        feat = _date_features(tmp.tail(1).assign(y=cur["y"].iloc[-1]))
        merged = pd.concat([cur, feat], ignore_index=True).tail(len(cur) + 1)
        # build features on the full (y) series with new row appended
        features_base = _date_features(cur.copy())
        features = _add_lags(pd.concat([features_base, pd.DataFrame({"ds": [next_ds], "y": [cur["y"].iloc[-1]]})], ignore_index=True))
        row = _date_features(pd.DataFrame({"ds": [next_ds], "y": [np.nan]}))
        features = _add_lags(pd.concat([cur[["ds", "y"]], row], ignore_index=True)).tail(1)
        # align columns
        X_row = features.drop(columns=["y", "ds"])
        # fillna from history
        X_row = X_row.fillna(cur["y"].mean())
        yhat = model.predict(X_row)[0]
        preds.append((next_ds, float(yhat)))
        # append to cur to allow next-step lags
        cur = pd.concat([cur, pd.DataFrame({"ds": [next_ds], "y": [yhat]})], ignore_index=True)
    return pd.DataFrame({"ds": [p[0] for p in preds], "yhat": [p[1] for p in preds]})


# -------------------- Training / selection --------------------
def _backtest_candidate(train_df: pd.DataFrame, candidate: str, freq: str, horizon: int, folds: int) -> float:
    y = train_df["y"].values
    n = len(train_df)
    splits = _rolling_splits(n, horizon, folds)
    scores = []
    for (train_end, test_end) in splits:
        tr = train_df.iloc[:train_end].reset_index(drop=True)
        te = train_df.iloc[train_end:test_end].reset_index(drop=True)

        if candidate == "prophet" and Prophet is not None:
            m = _fit_prophet(tr)
            if m is None: 
                continue
            fc = _forecast_prophet(m, tr, len(te), freq)
            score = _smape(te["y"].values, fc["yhat"].values)
        elif candidate == "arima" and pm is not None:
            m = _fit_arima(tr)
            if m is None:
                continue
            fc = _forecast_arima(m, tr["ds"].iloc[-1], len(te), freq)
            score = _smape(te["y"].values, fc["yhat"].values)
        elif candidate == "ml":
            m, cols = _fit_ml(tr)
            if m is None:
                continue
            fc = _forecast_ml(m, cols, tr, len(te), freq)
            score = _smape(te["y"].values, fc["yhat"].values)
        else:
            continue

        if not np.isfinite(score):
            continue
        scores.append(score)

    return float(np.mean(scores)) if scores else np.inf

df1=[]

def train_best_model(request: Request, dataset_id: str, date_col="ds", target_col="y", horizon: int = 30, folds: int = 3):

    if dataset_id not in _DATASETS:
        cache = request.app.state.cache
        if dataset_id not in cache:
            raise ValueError(f"No dataset '{dataset_id}' found in app.state.cache.")
        df = cache[dataset_id]["df"].copy()

        original_cols = df.columns.tolist()
        lower_cols = [col.lower() for col in original_cols]

        date_col = None
        target_col = None

        # ——— Regex-based detection ———
        for col in lower_cols:
            if date_col is None and re.search(r"(date)", col):
                date_col = col
            if target_col is None and re.search(r"(sale[s]?$|amount|value|revenue|target|price)", col):
                target_col = col

        if not date_col or not target_col:
            raise ValueError(f"Could not detect both date and sales columns using regex.")

        # ——— Rename to canonical names ———
        df = df.rename(columns={
            original_cols[lower_cols.index(date_col)]: "ds",
            original_cols[lower_cols.index(target_col)]: "y"
        })

        _DATASETS[dataset_id] = df


    df = _DATASETS[dataset_id].copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce", dayfirst=True)

    df["ds"] = pd.to_datetime(df["ds"]).dt.floor("D")      # Standardize
    df1 = df.dropna(subset=["ds", "y"])        # Remove nulls
    df1 = df1.groupby("ds", as_index=False)["y"].sum()  # ✅ Aggregate
    df1 = df1.sort_values("ds")               # optional but good


    if len(df1) < max(40, horizon * 2):
        raise ValueError("Not enough data to train a robust forecaster.")

    freq = _infer_freq(pd.DatetimeIndex(df1["ds"]))
    df1 = _resample_df(df1, freq, "y")

    # Evaluate candidates
    candidates = []
    if Prophet is not None:
        candidates.append("prophet")
    # if pm is not None:
    #     candidates.append("arima")
    candidates.append("ml")  # ML is always an option (falls back to RF if XGB missing)

    scores = {}
    for cand in candidates:
        scores[cand] = _backtest_candidate(df1, cand, freq, horizon, folds)

    best = min(scores, key=lambda k: scores[k]) if scores else None
    if best is None or not np.isfinite(scores[best]):
        raise RuntimeError("No viable model could be trained on this dataset.")

    # Fit best on full data
    if best == "prophet":
        model = _fit_prophet(df1)
    elif best == "arima":
        model = _fit_arima(df1)
    else:
        model, cols = _fit_ml(df1)
        if model is None:
            # Fallback if ML failed: try ARIMA or Prophet
            if "arima" in candidates:
                model = _fit_arima(df1)
                best = "arima"
            elif "prophet" in candidates:
                model = _fit_prophet(df1)
                best = "prophet"
            else:
                raise RuntimeError("ML model could not train and no other candidate available.")
        else:
            # stash columns for ML
            _MODELS[dataset_id] = {"model": model, "kind": "ml", "freq": freq, "horizon": horizon, "scores": scores, "cols": cols, "history": df1}
            return {
                "dataset_id": dataset_id,
                "chosen": "ml" if HAS_XGB else "rf",
                "scores": scores,
                "freq": freq,
                "horizon": horizon
            }

    _MODELS[dataset_id] = {"model": model, "kind": best, "freq": freq, "horizon": horizon, "scores": scores, "history": df1}
    return {"dataset_id": dataset_id, "chosen": best, "scores": scores, "freq": freq, "horizon": horizon}


def predict(dataset_id: str, horizon: Optional[int] = None) -> pd.DataFrame:

    if "default" not in _DATASETS:
      load_default_dataset_from_disk()


    if dataset_id not in _MODELS:
        raise ValueError("Model not trained. Train first.")
    meta = _MODELS[dataset_id]
    model = meta["model"]
    kind = meta["kind"]
    freq = meta["freq"]
    hist = meta["history"].copy()
    H = int(horizon or meta["horizon"])
    last_ds = hist["ds"].iloc[-1]

    if kind == "prophet":
        fc = _forecast_prophet(model, hist, H, freq)
    elif kind == "arima":
        fc = _forecast_arima(model, last_ds, H, freq)
    else:
        fc = _forecast_ml(model, meta.get("cols", []), hist, H, freq)

    _LAST_FCSTS[dataset_id] = fc.copy()
    return fc


def plot(dataset_id: str, kind: str) -> str:
    key = (dataset_id, kind)
    if key in _CACHE_PLOTS:
        return _CACHE_PLOTS[key]

    if dataset_id not in _MODELS:
        raise ValueError("Train a model first.")

    meta = _MODELS[dataset_id]
    df1 = meta["history"].copy()

    if kind == "history":
        plt.figure(figsize=(9, 4.5))
        #plt.plot(df1["ds"], df1["y"])
        plt.box(False) 
        plt.plot(df1["ds"], df1["y"], label="History", color="navy")
        plt.title("History")
        plt.xlabel("Date"); plt.ylabel("y")
        img = _to_base64_png()
    elif kind == "forecast":
        if dataset_id not in _LAST_FCSTS:
            _ = predict(dataset_id)  # compute default
        fc = _LAST_FCSTS[dataset_id]
        plt.figure(figsize=(9, 4.5))
        plt.plot(df1["ds"], df1["y"], label="History")
        plt.plot(fc["ds"], fc["yhat"], label="Forecast")
        plt.axvline(df1["ds"].iloc[-1], linestyle="--")
        plt.box(False) 
        plt.legend()
        plt.title("Forecast")
        plt.xlabel("Date"); plt.ylabel("y")
        img = _to_base64_png()
    elif kind == "decomposition":
        plt.figure(figsize=(10, 6))
        s = STL(df1.set_index("ds")["y"], robust=True).fit()
        ax1 = plt.subplot(3,1,1); ax1.plot(s.trend); ax1.set_title("Trend");
        ax2 = plt.subplot(3,1,2); ax2.plot(s.seasonal); ax2.set_title("Seasonality")
        ax3 = plt.subplot(3,1,3); ax3.plot(s.resid); ax3.set_title("Residuals")
        img = _to_base64_png()
    else:
        raise ValueError("Unknown plot kind.")

    _CACHE_PLOTS[key] = img
    return img


def upload_dataset(dataset_id: str, df: pd.DataFrame, date_col="ds", target_col="y"):
    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Columns not found. Need '{date_col}' and '{target_col}'.")
    _DATASETS[dataset_id] = df.copy()
    # clear stale caches
    _MODELS.pop(dataset_id, None)
    _CACHE_PLOTS_keys = [k for k in _CACHE_PLOTS if k[0] == dataset_id]
    for k in _CACHE_PLOTS_keys:
        _CACHE_PLOTS.pop(k, None)
    _LAST_FCSTS.pop(dataset_id, None)
    return {"ok": True}


import io
import re
import json
import pandas as pd
from datetime import datetime
from typing import Optional
from fastapi import Request, APIRouter, UploadFile, File
from openai import OpenAI

router = APIRouter()

@router.post("/forecast/check")
async def forecast_check(request: Request, dataset_id: str = "default"):
    cache = request.app.state.cache
    if dataset_id not in cache:
        return {"ok": False, "error": f"No dataset '{dataset_id}' in memory."}

    df = cache[dataset_id]["df"]
    if df.empty:
        return {"ok": False, "error": "Dataset is empty."}

    sample_rows = df.head(3).to_dict(orient="records")
    columns = df.columns.tolist()

    prompt = f"""
You are a forecasting analyst.

You are given a sample of a dataset. Your job is to check if it's suitable for time-series forecasting.
If yes:
- Return the exact column names that represent the datetime (`ds`) and the target variable (`y`) — use only the actual column names.
- If any transformation is needed before forecasting (like aggregating or filling), mention that too.

If no:
- Just say "Forecasting not possible" and explain why.

Columns: {columns}
Sample (first 3 rows): {json.dumps(sample_rows, indent=2)}
"""

    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful forecasting assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        text = response.choices[0].message.content.strip()
    except Exception as e:
        return {"ok": False, "error": f"OpenAI request failed: {e}"}

    # Attempt to extract ds and y column names using simple regex
    ds_col = None
    y_col = None

    ds_match = re.search(r"(?i)[`']?ds[`']?\s*[:=]\s*[`']?([\w\s\-]+)[`']?", text)
    y_match = re.search(r"(?i)[`']?y[`']?\s*[:=]\s*[`']?([\w\s\-]+)[`']?", text)

    if ds_match:
        ds_col = ds_match.group(1).strip()
    if y_match:
        y_col = y_match.group(1).strip()

    return {
        "ok": True,
        "ai_response": text,
        "ds": ds_col,
        "y": y_col
    }
