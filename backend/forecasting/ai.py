# backend/forecasting/ai.py
from json import loads, JSONDecodeError

import os
import numpy as np
import pandas as pd
from typing import Dict, Any

# Optional OpenAI (v1) import
try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

def _summary_stats(history: pd.DataFrame, forecast: pd.DataFrame) -> Dict[str, Any]:
    df = history.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")
    y = df["y"].values
    n = len(df)

    trend = float(np.polyfit(np.arange(n), y, 1)[0])
    growth = "rising" if trend > 0 else ("falling" if trend < 0 else "flat")
    cv = float(np.std(y) / (np.mean(y) + 1e-9))

    df["month"] = df["ds"].dt.month
    df["dow"] = df["ds"].dt.dayofweek
    m_means = df.groupby("month")["y"].mean().sort_values(ascending=False)
    d_means = df.groupby("dow")["y"].mean().sort_values(ascending=False)

    ac = pd.Series(y - y.mean())
    ac_vals = np.correlate(ac, ac, mode="full")
    ac_vals = ac_vals[ac_vals.size // 2:]
    ac_vals = ac_vals / (ac_vals[0] + 1e-9)

    if len(ac_vals) > 30:
        peak_lag = int(np.argmax(ac_vals[1:30]) + 1)
        seasonality_lag = peak_lag
    else:
        seasonality_lag = None

    total_next = float(forecast["yhat"].sum()) if forecast is not None and "yhat" in forecast else None

    return {
        "growth": growth,
        "trend_slope": trend,
        "volatility_cv": cv,
        "top_months": m_means.head(3).index.tolist(),
        "top_weekdays": d_means.head(3).index.tolist(),
        "seasonality_lag": seasonality_lag,
        "next_total": total_next
    }

def _heuristic_text(stats: Dict[str, Any]) -> str:
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    tops_m = ", ".join(month_names.get(m, str(m)) for m in stats["top_months"])
    tops_d = ", ".join(dow_names[d] for d in stats["top_weekdays"]) if stats["top_weekdays"] else ""
    parts = []
    parts.append(f"Baseline trend looks {stats['growth']} (slope={stats['trend_slope']:.4f}).")
    parts.append(f"Volatility (CV) ≈ {stats['volatility_cv']:.2f}.")
    if stats["top_months"]:
        parts.append(f"Peak months: {tops_m}.")
    if stats["top_weekdays"]:
        parts.append(f"Strong weekdays: {tops_d}.")
    if stats["seasonality_lag"]:
        parts.append(f"Autocorrelation suggests a repeating pattern roughly every {stats['seasonality_lag']} steps.")
    if stats["next_total"] is not None:
        parts.append(f"Projected total over the forecast horizon ≈ {stats['next_total']:.2f}.")
    parts.append("Consider capacity planning and inventory alignment around the peak periods.")
    return "\n".join(parts)

def generate_ai_insights(history: pd.DataFrame, forecast: pd.DataFrame) -> Dict[str, str]:
    stats = _summary_stats(history, forecast)
    heuristic = _heuristic_text(stats)

    

    # Exit early if OpenAI not available or no key
    api_key = os.getenv("OPENAI_API_KEY")
    if not HAS_OPENAI or api_key in (None, "", "None"):
        return {"heuristics": heuristic, "ai": ""}

    client = OpenAI(api_key=api_key)

    prompt = """
You are a data analyst assistant generating an executive summary from time series forecast data.

You are given:
1. Historical daily data in a dataframe with columns: ds (date), y (value)
2. Forecasted data in another dataframe with columns: ds (date), yhat (prediction), yhat_lower, yhat_upper

Generate a comprehensive insight summary on several of the following topics:
- Historical trends and seasonality
- Volatility or variability in the data
- Peak months, weak months
- Strong and weak days of the week
- Autocorrelation or cyclic behavior
- Forecast direction (rising/falling/stable)
- Projected totals and expected growth
- Observations from uncertainty bands
- Any anomalies or changepoints visible
- Actionable business suggestions (inventory, staffing, etc.)

The tone should be executive-friendly and insight-driven. Do not bold any words. Keep each point under 2 lines. Separate each point by a newline. Limit the response to 300 words.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        ai_text = response.choices[0].message.content
        ai_points = [p.strip() for p in ai_text.split("\n") if p.strip()]

    except Exception as e:
        print("OpenAI call failed:", e)
        ai_text = ""

    return {"heuristics": heuristic, "ai": ai_points}



# --- ADD THIS NEAR THE BOTTOM OF ai.py ---

def assess_forecastability(sample_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Decide if forecasting is feasible and which columns should be ds (date) and y (numeric target).
    Returns: {"possible": bool, "ds": Optional[str], "y": Optional[str], "reason": str}
    """
    # --- Heuristic fallback we can always rely on ---
    def _heuristic(df: pd.DataFrame) -> Dict[str, Any]:
        cols = df.columns.tolist()
        lower = [c.lower() for c in cols]

        # date-like by dtype or by name hint
        ds_guess = None
        for c in cols:
            s = df[c]
            # try dtype or parseability
            if np.issubdtype(s.dtype, np.datetime64):
                ds_guess = c
                break
            try:
                pd.to_datetime(s, errors="raise", dayfirst=True)
                ds_guess = c
                break
            except Exception:
                pass
        if ds_guess is None:
            for c in cols:
                lc = c.lower()
                if any(k in lc for k in ["date", "ds", "day", "month", "year"]):
                    ds_guess = c
                    break

        # numeric target that isn't an id/key
        bad_words = {"id", "code", "key"}
        num_candidates = []
        for c in cols:
            if c == ds_guess:
                continue
            lc = c.lower()
            if any(w in lc for w in bad_words):
                continue
            # numeric-ish check
            try:
                pd.to_numeric(df[c], errors="coerce")
                num_candidates.append(c)
            except Exception:
                pass

        y_guess = None
        # prefer businessy names
        for pref in ["sales", "revenue", "amount", "value", "target", "qty", "price"]:
            for c in num_candidates:
                if pref in c.lower():
                    y_guess = c
                    break
            if y_guess:
                break
        if y_guess is None and num_candidates:
            y_guess = num_candidates[0]

        ok = ds_guess is not None and y_guess is not None
        reason = "Detected date and numeric target." if ok else "Could not find a date column and a numeric target."
        return {"possible": bool(ok), "ds": ds_guess, "y": y_guess, "reason": reason}

    # If no OpenAI, do heuristic only
    api_key = os.getenv("OPENAI_API_KEY")
    if not HAS_OPENAI or api_key in (None, "", "None"):
        return _heuristic(sample_df)

    # With OpenAI: ask for strict JSON
    client = OpenAI(api_key=api_key)

    as_csv = sample_df.to_csv(index=False)
    cols_str = ", ".join(sample_df.columns.tolist())

    user_msg = f"""
You are given the first 3 rows of a tabular dataset (CSV snippet below).
Task: Decide if time-series forecasting is possible on this dataset.
Rules:
- Choose exactly one date/time column for `ds`.
- Choose exactly one numeric measure suitable as a target for `y`.
- If unsuitable, set "possible": false and leave "ds": null, "y": null with a brief reason.

Return STRICT JSON with keys: possible (bool), ds (string or null), y (string or null), reason (string).

Columns: {cols_str}

CSV (3 rows):
{as_csv}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a careful data analyst. Respond with strict JSON only."},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.2,
            max_tokens=250
        )
        raw = resp.choices[0].message.content.strip()
        try:
            data = loads(raw)
            # sanity check
            if not isinstance(data, dict):
                raise ValueError("Not a dict")
            for k in ["possible", "ds", "y", "reason"]:
                data.setdefault(k, None)
            # fall back if AI said no but heuristics can still help
            if not data.get("possible"):
                return _heuristic(sample_df)
            return {
                "possible": bool(data.get("possible")),
                "ds": data.get("ds"),
                "y": data.get("y"),
                "reason": data.get("reason") or "AI check successful."
            }
        except (JSONDecodeError, Exception):
            # AI parsing failed → heuristics
            return _heuristic(sample_df)
    except Exception as e:
        print("assess_forecastability OpenAI error:", e)
        return _heuristic(sample_df)
