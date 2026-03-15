from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_xgb_pipeline.joblib"
DEFAULT_DEMO_DIR = PROJECT_ROOT / "data" / "external"/ "inference"


class ModelService:
    def __init__(self, model_path: str | Path | None = None) -> None:
        env_model_path = os.getenv("MODEL_PATH")
        final_model_path = Path(model_path or env_model_path or DEFAULT_MODEL_PATH)

        if not final_model_path.exists():
            raise FileNotFoundError(f"Model file not found at {final_model_path}")

        self.demo_dir = DEFAULT_DEMO_DIR
        self.model_path = final_model_path
        self.model = joblib.load(final_model_path)

    def model_info(self) -> str:
        return repr(self.model)
    
    def load_demo_dataset(self, dataset_name: str) -> pd.DataFrame:
        valid = {"small", "medium", "large"}
        if dataset_name not in valid:
            raise ValueError(f"dataset_name must be one of {sorted(valid)}")

        path = self.demo_dir / f"demo_{dataset_name}_inference.csv"
        if not path.exists():
            raise FileNotFoundError(f"Demo dataset not found at {path}")

        df = pd.read_csv(path).reset_index(names="flight_id")
        return df

    def predict(
        self,
        records: list[dict],
        top_k: int | None = None,
        threshold: float = 0.5,
    ) -> list[dict]:
        if not records:
            return []

        df = pd.DataFrame(records).copy()

        if "flight_id" not in df.columns:
            df = df.reset_index(names="flight_id")

        probs = self.model.predict_proba(df)[:, 1]
        preds = (probs >= threshold).astype(int)

        df["probability_delayed"] = probs
        df["predicted_label"] = preds

        df = df.sort_values("probability_delayed", ascending=False)

        if top_k is not None:
            df = df.head(top_k)

        return df.to_dict(orient="records")
    
    def predict_demo(
        self,
        dataset_name: str,
        top_k: int | None = None,
        threshold: float = 0.5,
    ) -> list[dict]:
        df = self.load_demo_dataset(dataset_name)

        probs = self.model.predict_proba(df)[:, 1]
        preds = (probs >= threshold).astype(int)

        df["probability_delayed"] = probs
        df["predicted_label"] = preds

        df = df.sort_values("probability_delayed", ascending=False)

        if top_k is not None:
            df = df.head(top_k)

        return df.to_dict(orient="records")