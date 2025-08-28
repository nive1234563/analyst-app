# backend/forecasting/router.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi import Body
from pydantic import BaseModel
import pandas as pd

from .model import upload_dataset, train_best_model, predict, plot, _MODELS
from .ai import generate_ai_insights

router = APIRouter(prefix="/forecast", tags=["Forecast"])

class TrainReq(BaseModel):
    dataset_id: str = "default"
    date_col: str = "ds"
    target_col: str = "y"
    horizon: int = 30
    folds: int = 3

@router.post("/upload")
async def upload_csv(dataset_id: str = Form("default"), file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(pd.io.common.BytesIO(content))

    if "ds" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "ds"})

    if "y" not in df.columns:
        for alt in ["sales", "value", "target"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "y"})
                break

    res = upload_dataset(dataset_id, df, date_col="ds", target_col="y")
    return {"ok": True, "rows": len(df), "columns": list(df.columns), **res}


@router.post("/train")
def train(req: TrainReq):
    try:
        meta = train_best_model(
            dataset_id=req.dataset_id,
            date_col=req.date_col,
            target_col=req.target_col,
            horizon=req.horizon,
            folds=req.folds
        )
        return meta
    except Exception as e:
        raise HTTPException(400, f"Train failed: {e}")

@router.get("/predict")
def do_predict(dataset_id: str, horizon: int | None = None):
    try:
        fc = predict(dataset_id, horizon)
        return {"forecast": fc.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(400, f"Predict failed: {e}")

@router.get("/plot/{kind}")
def get_plot(kind: str, dataset_id: str):
    try:
        img64 = plot(dataset_id, kind)
        return {"image": img64}
    except Exception as e:
        raise HTTPException(400, f"Plot failed: {e}")

@router.post("/insights")
def insights(dataset_id: str = Body(..., embed=True)):
    try:
        if dataset_id not in _MODELS:
            raise HTTPException(400, "Train a model first.")
        hist = _MODELS[dataset_id]["history"]
        # get fresh forecast if missing
        from .model import _LAST_FCSTS, predict as _predict
        if dataset_id not in _LAST_FCSTS:
            _ = _predict(dataset_id)
        fc = _LAST_FCSTS[dataset_id]
        out = generate_ai_insights(hist, fc)
        return out
    except Exception as e:
        raise HTTPException(400, f"Insights failed: {e}")
