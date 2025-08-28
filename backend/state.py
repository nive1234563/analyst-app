

# state.py
import io
import pandas as pd
from datetime import datetime

UPLOADED_DF = {}

def handle_uploaded_file(content: bytes, filename: str, app_state: dict) -> dict:
    try:
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return {"ok": False, "error": f"Failed to parse file: {e}"}

    if df is None or df.empty:
        return {"ok": False, "error": "File parsed but no rows found."}

    dataset_id = "default"
    UPLOADED_DF[dataset_id] = df
    app_state[dataset_id] = {"df": df, "uploaded_at": datetime.utcnow().isoformat()}

    return {
        "ok": True,
        "dataset_id": dataset_id,
        "rows": int(len(df)),
        "cols": df.columns.tolist(),
    }
