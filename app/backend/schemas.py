from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        ...,
        description="List of model input records."
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Optional number of top-risk rows to return."
    )


class PredictionRow(BaseModel):
    flight_id: int | None = None
    probability_delayed: float
    predicted_label: int


class PredictResponse(BaseModel):
    n_records_scored: int
    top_k_returned: int | None = None
    predictions: list[dict[str, Any]]