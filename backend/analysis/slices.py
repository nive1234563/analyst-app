# backend/analysis/slices.py
import os, re, json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

# Optional OpenAI
try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# ---------- column detection & parsing ----------

DATE_ALIASES = {
    'ds','date','orderdate','transactiondate','invoice_date','invoice date',
    'datetime','timestamp','order_datetime','createdat','eventtime'
}
TARGET_ALIASES = {
    'y','sales','sale','amount','revenue','price','gmv','net_sales','gross_sales','total'
}
EXCLUDE_NAME_REGEX = re.compile(
    r"(?:^|_)(id|uuid|code|email|phone|mobile|lat|lon|long|zipcode|pincode|address|name)(?:_|$)",
    re.I
)

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def detect_date_target(df: pd.DataFrame,
                       date_col: Optional[str] = None,
                       target_col: Optional[str] = None) -> Tuple[str, str]:
    nm = {c: _norm(c) for c in df.columns}

    if date_col is None:
        # require "date" as token or known aliases
        for c in df.columns:
            tokens = re.split(r'[_\W]+', c.lower())
            if "date" in tokens or nm[c] in DATE_ALIASES:
                date_col = c
                break
    if target_col is None:
        for c in df.columns:
            n = nm[c]
            if n in TARGET_ALIASES or re.search(r"(sale[s]?$|amount|value|revenue|price|gmv|total)", n):
                target_col = c
                break

    if not date_col:
        raise ValueError("No date-like column found. Set date_col explicitly.")
    if not target_col:
        raise ValueError("No sales/target column found. Set target_col explicitly.")
    return date_col, target_col

def parse_dates(series: pd.Series) -> pd.Series:
    # robust: try mixed/dayfirst, then numeric epochs, then YYYYMMDD
    try:
        dt = pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")
    except TypeError:
        dt = pd.to_datetime(series, dayfirst=True, errors="coerce")
    if dt.isna().mean() > 0.5:
        s_nonnull = series.dropna()
        if pd.api.types.is_numeric_dtype(s_nonnull):
            x = s_nonnull.astype("int64")
            if x.between(10**12, 10**14).mean() > 0.6:  # ms epoch
                dt = pd.to_datetime(series, unit="ms", errors="coerce")
            elif x.between(10**9, 10**11).mean() > 0.6:  # s epoch
                dt = pd.to_datetime(series, unit="s", errors="coerce")
            elif s_nonnull.astype(str).str.len().eq(8).mean() > 0.6:
                dt = pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")
    return dt

def auto_freq(d0: pd.Timestamp, d1: pd.Timestamp) -> str:
    if not isinstance(d0, pd.Timestamp) or not isinstance(d1, pd.Timestamp):
        return "M"
    span = (d1 - d0).days
    if span <= 120:
        return "D"
    elif span <= 900:
        return "W"
    else:
        return "M"

# ---------- core per-dimension analysis ----------

def choose_dimensions(df: pd.DataFrame,
                      date_col: str,
                      target_col: str,
                      max_dims: int = 12) -> List[str]:
    dims: List[str] = []
    for c in df.columns:
        if c == date_col or c == target_col:
            continue
        if EXCLUDE_NAME_REGEX.search(c):
            continue
        nun = df[c].nunique(dropna=True)
        # small/medium cardinality and mostly categorical/text
        if nun >= 2 and nun <= 100 and (df[c].dtype.name in ("object", "category") or nun < len(df) / 2):
            dims.append(c)
    # deterministic order: country, city first if present
    priority = ["country", "city", "state", "region", "category", "segment", "product", "brand", "channel"]
    dims = sorted(dims, key=lambda x: (priority.index(x.lower()) if x.lower() in priority else 999, x.lower()))
    return dims[:max_dims]

def bar_and_line_for_dim(df: pd.DataFrame,
                         dim: str,
                         date_col: str,
                         target_col: str,
                         freq: Optional[str] = None,
                         top_k: int = 10,
                         top_lines: int = 3) -> Dict[str, Any]:
    out = df.copy()

    # BAR: totals by dim
    bar = (out.groupby(dim, dropna=False)[target_col]
             .agg(total="sum", avg="mean", n="count")
             .sort_values("total", ascending=False)
             .head(top_k)
             .reset_index())
    bar["total"] = bar["total"].round(2)
    bar_json = bar[[dim, "total"]].rename(columns={dim: "label"}).to_dict("records")

    # LINE: resample by time x dim, keep top_lines series
    out = out.set_index(date_col)
    if freq is None:
        freq = auto_freq(out.index.min(), out.index.max())
    ts = out.groupby(dim)[target_col].resample(freq).sum().reset_index()
    pv = ts.pivot_table(index=date_col, columns=dim, values=target_col, aggfunc="sum").fillna(0.0)
    totals = pv.sum().sort_values(ascending=False)
    keep = totals.head(min(top_lines, len(totals))).index.tolist()
    pv = pv[keep]
    pv = pv.round(2)

    line = {
        "labels": [pd.to_datetime(i).strftime("%Y-%m-%d") for i in pv.index],
        "series": [{"name": c, "data": pv[c].astype(float).tolist()} for c in pv.columns]
    }

    # PEAK/TROUGH for insights
    peaks = []
    for c in keep:
        s = pv[c]
        if len(s) == 0:
            continue
        idxmax = s.idxmax()
        idxmin = s.idxmin()
        peaks.append({
            "value": c,
            "peak_period": pd.to_datetime(idxmax).strftime("%Y-%m-%d") if pd.notna(idxmax) else None,
            "peak_total": float(s.max()) if np.isfinite(s.max()) else None,
            "trough_period": pd.to_datetime(idxmin).strftime("%Y-%m-%d") if pd.notna(idxmin) else None,
            "trough_total": float(s.min()) if np.isfinite(s.min()) else None
        })

    best_overall = None
    worst_overall = None
    overall_ts = pv.sum(axis=1)
    if len(overall_ts):
        i_max = overall_ts.idxmax()
        i_min = overall_ts.idxmin()
        best_overall = {
            "period": pd.to_datetime(i_max).strftime("%Y-%m-%d"),
            "total": float(overall_ts.max())
        }
        worst_overall = {
            "period": pd.to_datetime(i_min).strftime("%Y-%m-%d"),
            "total": float(overall_ts.min())
        }

    return {
        "bar": bar_json,
        "line": line,
        "peaks": peaks,
        "best_overall": best_overall or {},
        "worst_overall": worst_overall or {},
        "freq": freq
    }

# ---------- AI insights (single call for all dims) ----------

def ai_insights_for_dims(dim_contexts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not HAS_OPENAI or not api_key or not dim_contexts:
        # no AI - return empty mapping
        return {ctx["dimension"]: [] for ctx in dim_contexts}

    client = OpenAI(api_key=api_key)

    ctx_json = json.dumps(dim_contexts, ensure_ascii=False)
    prompt = f"""
You are a data analyst. You will receive an array of dimension contexts (JSON) for sales.
For EACH object, give concise, executive insights  , total words - 100 ,
about that single dimension: deep business insights with statistics . also give some suggestions on what can be done to improve the business. and what might be going wrong . Give strategic suggestions.
You can refer to other bjects and give related insight but only if it is highly important and significant. 

Return ONLY a JSON object mapping dimension name -> array of strings.

DIMENSIONS_CONTEXT:
{ctx_json}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You write clear, business-ready insights with evidence."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=900,
        )
        raw = resp.choices[0].message.content
        return json.loads(raw)
    except Exception as e:
        print("AI insights failed:", e)
        return {ctx["dimension"]: [] for ctx in dim_contexts}

# ---------- public API: analyze all dims ----------

def analyze_slices(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    target_col: Optional[str] = None,
    max_dims: int = 12,
    top_k: int = 10,
    top_lines: int = 3,
) -> Dict[str, Any]:

    if df is None or df.empty:
        raise ValueError("Empty dataframe.")

    # detect columns & clean
    date_col, target_col = detect_date_target(df, date_col, target_col)
    data = df.copy()
    data[date_col] = parse_dates(data[date_col])
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=[date_col, target_col])
    data = data.sort_values(date_col)

    dims = choose_dimensions(data, date_col, target_col, max_dims=max_dims)

    # build per-dimension outputs + compact AI context
    dimensions_payload: List[Dict[str, Any]] = []
    ai_contexts: List[Dict[str, Any]] = []

    for dim in dims:
        part = bar_and_line_for_dim(
            df=data,
            dim=dim,
            date_col=date_col,
            target_col=target_col,
            freq=None,
            top_k=top_k,
            top_lines=top_lines
        )
        dimensions_payload.append({
            "name": dim,
            "bar": part["bar"],
            "line": part["line"],
            "freq": part["freq"],
            "best_overall": part["best_overall"],
            "worst_overall": part["worst_overall"]
        })

        ai_contexts.append({
            "dimension": dim,
            "freq": part["freq"],
            "top_values": part["bar"],           # label + total
            "best_overall": part["best_overall"],
            "worst_overall": part["worst_overall"],
            "peaks": part["peaks"]              # per top series
        })

    # one AI call for all dims (maps dim -> insights[])
    insights_map = ai_insights_for_dims(ai_contexts)

    # attach insights (array of strings)
    for d in dimensions_payload:
        d["insights"] = insights_map.get(d["name"], [])

    return {
        "date_col": date_col,
        "target_col": target_col,
        "dimensions": dimensions_payload
    }
