# transforms.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable
import pandas as pd
import numpy as np
import re

COMMON_HINTS = {
    # key -> handler function signature: handler(df, args_dict) -> df
    "parse_time_column": "Parse time column to datetime (dayfirst-safe).",
    "coerce_target_numeric": "Coerce target column to numeric (strip commas).",
    "fillna_target_zero": "Fill NA in target with 0.",
    "groupby_date_sum": "Group by date and sum target.",
    "sort_by_date": "Sort by the date column ascending.",
    "drop_na_ds_y": "Drop rows where date or target is NA."
}

def _coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "", regex=False)
    # also handle stray spaces and currency symbols
    s = s.str.replace(r"[^\d\.\-\+eE]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def apply_modifications(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    modifications: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Applies a small set of common operations indicated by the AI 'modifications' list.
    Items can be free-form; we match by keywords to keep it robust.
    """
    mods = [m.lower() for m in (modifications or [])]

    # Always try to parse time column if looks like it needs parsing
    if "parse" in " ".join(mods) or True:
        if time_col in df.columns:
            try:
                parsed = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
                if parsed.notna().mean() > 0.3:
                    df[time_col] = parsed.dt.floor("D")
            except Exception:
                pass

    # Target numeric coerce if asked or if dtype is object
    if ("numeric" in " ".join(mods)) or (df[target_col].dtype == object):
        df[target_col] = _coerce_numeric(df[target_col])

    # Fill NA target -> 0 if hint present
    if any("fillna" in m or "impute" in m or "zero" in m for m in mods):
        df[target_col] = df[target_col].fillna(0.0)

    # Drop NA ds/y if asked
    if any("drop na" in m or "drop null" in m for m in mods):
        df = df.dropna(subset=[time_col, target_col])

    # Group by date sum if suggested
    if any("group" in m and "date" in m for m in mods) or any("aggregate" in m for m in mods):
        if time_col in df.columns and target_col in df.columns:
            df = (
                df.groupby(time_col, as_index=False)[target_col]
                .sum()
            )

    # Sort by date if asked (or by default after groupby)
    if any("sort" in m for m in mods) or True:
        if time_col in df.columns:
            df = df.sort_values(time_col)

    return df

def run_custom_transform_if_any(
    df: pd.DataFrame,
    transform_py: Optional[str]
) -> pd.DataFrame:
    """
    Runs an optional custom transform snippet if provided.
    Snippet must define a function:
        def transform(df: pd.DataFrame) -> pd.DataFrame:
            ...
    """
    if not transform_py or not transform_py.strip():
        return df

    # Very lightweight sandbox: provide only pandas, numpy as names, no builtins write access.
    # Note: If running untrusted code, consider stronger sandboxing.
    safe_globals: Dict[str, Any] = {
        "__builtins__": {"len": len, "range": range, "min": min, "max": max, "sum": sum},
        "pd": pd,
        "np": np,
    }
    safe_locals: Dict[str, Any] = {}

    try:
        exec(transform_py, safe_globals, safe_locals)
    except Exception as e:
        raise RuntimeError(f"Failed compiling custom transform: {e}")

    fn = safe_locals.get("transform") or safe_globals.get("transform")
    if not callable(fn):
        raise RuntimeError("Custom transform code must define a function: transform(df) -> df")

    try:
        out = fn(df.copy())
    except Exception as e:
        raise RuntimeError(f"Custom transform failed: {e}")

    if not isinstance(out, pd.DataFrame):
        raise RuntimeError("Custom transform must return a pandas DataFrame.")
    return out
