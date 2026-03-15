"""
Example FastAPI backend for serving ranked delay predictions
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .schemas import PredictRequest, PredictResponse
from .service import ModelService


model_service: ModelService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_service
    model_service = ModelService()
    yield
    model_service = None


app = FastAPI(
    title="PHL Delay Prediction API",
    description="Serve delay-risk predictions for batches of inbound flights.",
    version="1.0.0",
    lifespan=lifespan,
)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def get_model_info() -> dict:
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model service not initialized.")
    return {"model": model_service.model_info()}

@app.get("/example-datasets")
def list_example_datasets() -> dict:
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model service not initialized.")

    out = {}
    for name in ["small", "medium", "large"]:
        df = model_service.load_demo_dataset(name)
        out[name] = {"rows": len(df), "columns": list(df.columns)}

    return out

@app.get("/predict/example")
def predict_example(
    dataset: str = Query("small", pattern="^(small|medium|large)$"),
    top_k: int = Query(10, ge=1, le=250),
) -> dict:
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model service not initialized.")

    try:
        predictions = model_service.predict_demo(
            dataset_name=dataset,
            top_k=top_k,
            threshold=0.5,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Example prediction failed: {exc}") from exc

    return {
        "dataset": dataset,
        "n_records_returned": len(predictions),
        "top_k": top_k,
        "predictions": predictions,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model service not initialized.")

    try:
        predictions = model_service.predict(
            records=request.records,
            top_k=request.top_k,
            threshold=0.5,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return PredictResponse(
        n_records_scored=len(request.records),
        top_k_returned=request.top_k,
        predictions=predictions,
    )