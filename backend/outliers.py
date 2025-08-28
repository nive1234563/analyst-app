# # backend/outliers.py
# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# import pandas as pd
# import numpy as np

# from state import UPLOADED_DF


# router = APIRouter()

# # ------------ Models ------------
# class SchemaIn(BaseModel):
#     dataset_id: str = "default"

# class DetectIn(BaseModel):
#     dataset_id: str = "default"
#     target_col: str
#     method: str = "iqr"   # iqr | zscore | isoforest
#     group_by: str | None = None

# class TimeSeriesIn(BaseModel):
#     dataset_id: str = "default"
#     date_col: str
#     target_col: str
#     window: int = 7
#     z_threshold: float = 3.0

# # ------------ Helpers ------------
# def _get_df(dataset_id: str) -> pd.DataFrame:
#     if dataset_id not in CACHE or "df" not in CACHE[dataset_id]:
#         raise HTTPException(status_code=400, detail="Dataset not found. Upload CSV first.")
#     df = CACHE[dataset_id]["df"]
#     if not isinstance(df, pd.DataFrame):
#         raise HTTPException(status_code=500, detail="Cached dataset is not a DataFrame.")
#     return df

# def _to_datetime_safe(s: pd.Series) -> pd.Series:
#     return pd.to_datetime(s, errors="coerce", dayfirst=True)

# # ------------ Endpoints ------------
# @router.post("/dataset/schema")
# def dataset_schema(inp: SchemaIn):
#     df = _get_df(inp.dataset_id)
#     cols = df.columns.tolist()
#     # Infer numeric and date-like columns
#     numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
#     # Try to detect date columns by dtype or parse success rate
#     date_candidates = []
#     for c in cols:
#         if pd.api.types.is_datetime64_any_dtype(df[c]):
#             date_candidates.append(c)
#         else:
#             # heuristic: sample 50 rows to avoid heavy parsing
#             sample = df[c].dropna().astype(str).head(50)
#             if not len(sample):
#                 continue
#             parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
#             if parsed.notna().mean() > 0.7:
#                 date_candidates.append(c)

#     return {"columns": cols, "numeric": numeric, "dates": list(dict.fromkeys(date_candidates))}

# @router.post("/outliers/detect")
# def outliers_detect(inp: DetectIn):
#     df = _get_df(inp.dataset_id).copy()
#     col = inp.target_col
#     if col not in df.columns:
#         raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

#     if not pd.api.types.is_numeric_dtype(df[col]):
#         # try coercion
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     valid = df[col].dropna()
#     if valid.empty:
#         return {"count": 0, "rows": [], "meta": {"note": "No numeric values"}}

#     rows = []
#     meta = {}

#     def detect_series(s: pd.Series) -> pd.Index:
#         if inp.method.lower() == "iqr":
#             q1, q3 = np.nanpercentile(s, 25), np.nanpercentile(s, 75)
#             iqr = q3 - q1
#             lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
#             meta["threshold"] = {"low": float(lo), "high": float(hi)}
#             return s[(s < lo) | (s > hi)].index
#         elif inp.method.lower() == "zscore":
#             m, sd = np.nanmean(s), np.nanstd(s)
#             sd = sd if sd > 0 else 1e-9
#             z = (s - m) / sd
#             meta["threshold"] = {"|z|": 3.0}
#             return s[np.abs(z) > 3.0].index
#         elif inp.method.lower() == "isoforest":
#             try:
#                 from sklearn.ensemble import IsolationForest
#             except Exception as e:
#                 raise HTTPException(status_code=500, detail=f"sklearn not available: {e}")
#             X = s.to_frame(name=col).values
#             clf = IsolationForest(contamination="auto", random_state=42)
#             pred = clf.fit_predict(X)  # -1 = outlier
#             idx = s.index[pred == -1]
#             meta["threshold"] = {"model": "IsolationForest"}
#             return idx
#         else:
#             raise HTTPException(status_code=400, detail=f"Unknown method '{inp.method}'")

#     if inp.group_by and inp.group_by in df.columns:
#         out_idx = []
#         for g, gdf in df.groupby(inp.group_by):
#             s = pd.to_numeric(gdf[col], errors="coerce")
#             idx = detect_series(s)
#             out_idx.extend(idx)
#         out_df = df.loc[out_idx, [inp.group_by, col]].copy()
#     else:
#         s = pd.to_numeric(df[col], errors="coerce")
#         out_idx = detect_series(s)
#         out_df = df.loc[out_idx, [col]].copy()

#     out_df = out_df.dropna()
#     rows = out_df.reset_index().rename(columns={"index": "_row"}).to_dict(orient="records")
#     return {"count": len(rows), "rows": rows, "meta": meta}

# @router.post("/outliers/timeseries")
# def outliers_timeseries(inp: TimeSeriesIn):
#     df = _get_df(inp.dataset_id).copy()
#     if inp.date_col not in df.columns or inp.target_col not in df.columns:
#         raise HTTPException(status_code=400, detail="date_col or target_col not found")

#     # coerce
#     df[inp.date_col] = _to_datetime_safe(df[inp.date_col])
#     df[inp.target_col] = pd.to_numeric(df[inp.target_col], errors="coerce")
#     sdf = df[[inp.date_col, inp.target_col]].dropna().sort_values(inp.date_col)

#     if sdf.empty:
#         return {"points": [], "meta": {"note": "no valid points"}}

#     s = sdf[inp.target_col]
#     # rolling mean/std z-score
#     w = max(3, int(inp.window))
#     roll_mean = s.rolling(w, min_periods=1).mean()
#     roll_std = s.rolling(w, min_periods=1).std().replace(0, np.nan).fillna(1e-9)
#     z = (s - roll_mean) / roll_std
#     outliers = np.abs(z) > float(inp.z_threshold)

#     points = []
#     for ds, y, is_o in zip(sdf[inp.date_col], s, outliers):
#         points.append({"ds": str(ds), "y": float(y), "is_outlier": bool(is_o)})

#     return {
#         "points": points,
#         "meta": {"window": w, "z_threshold": float(inp.z_threshold)}
#     }
