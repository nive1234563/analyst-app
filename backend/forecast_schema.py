# forecast_schema.py
from __future__ import annotations
from typing import Dict, Optional, Any

# In-memory registry of forecast schema & transforms chosen for each dataset
# {
#   dataset_id: {
#       "time_col": str,          # required
#       "target_col": str,        # required
#       "modifications": list,    # optional list[str]
#       "transform_py": str|None  # optional python code defining transform(df)->df
#   }
# }
_FORECAST_SCHEMA: Dict[str, Dict[str, Any]] = {}

def set_schema(dataset_id: str,
               time_col: str,
               target_col: str,
               modifications: Optional[list] = None,
               transform_py: Optional[str] = None) -> Dict[str, Any]:
    _FORECAST_SCHEMA[dataset_id] = {
        "time_col": time_col,
        "target_col": target_col,
        "modifications": modifications or [],
        "transform_py": transform_py or None,
    }
    return {"ok": True, "saved": _FORECAST_SCHEMA[dataset_id]}

def get_schema(dataset_id: str) -> Optional[Dict[str, Any]]:
    return _FORECAST_SCHEMA.get(dataset_id)

def clear_schema(dataset_id: str) -> None:
    _FORECAST_SCHEMA.pop(dataset_id, None)
