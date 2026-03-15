"""Train the final XGBoost pipeline, select a threshold, and save artifacts."""

from __future__ import annotations

import sys
import json
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from flight_delay_prediction.data.load_data import load_flights, load_destination_weather, load_origin_weather
from flight_delay_prediction.data.preprocess import clean_flight_data, clean_destination_weather_data, clean_origin_weather_data, join_flights_and_weather
from flight_delay_prediction.features.build_features import build_model_dataframe, train_test_split_model_data, get_modeling_metadata
from flight_delay_prediction.models.train import tune_xgb_model
from flight_delay_prediction.models.thresholding import (
    build_threshold_table,
    select_best_threshold_by_f1,
)
from flight_delay_prediction.utils.metrics import evaluate_model


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_xgb_pipeline.joblib"
THRESHOLD_TABLE_PATH = ARTIFACTS_DIR / "threshold_table.csv"
METRICS_PATH = ARTIFACTS_DIR / "metrics_summary.json"
TRAINING_SUMMARY_PATH = ARTIFACTS_DIR / "training_summary.json"


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading modeling dataset...")
    model_df = pd.read_parquet(PROJECT_ROOT / "data/processed/cleaned_pre_encoding_data.parquet")
    model_df_metadata = get_modeling_metadata()
    
    X = model_df.drop(columns=[model_df_metadata["target"]])
    y = model_df[model_df_metadata["target"]]

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split_model_data(model_df)

    print("Tuning XGBoost with cross-validation...")
    xgb_search = tune_xgb_model(
        X_train=X_train,
        y_train=y_train,
        oh_encoding_cols=model_df_metadata["oh_encoding_cols"],
        target_encoding_cols=model_df_metadata["target_encoding_cols"],
        n_iter=10,
        scoring="average_precision",
        cv=5,
        n_jobs=-1,
        verbose=2,
    )

    best_pipeline = xgb_search.best_estimator_

    print("\nBest CV average precision:")
    print(f"{xgb_search.best_score_:.4f}")

    print("\nBest hyperparameters:")
    print(xgb_search.best_params_)

    print("\nEvaluating tuned model on train/test split...")
    metrics = evaluate_model(
        pipeline=best_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    metrics_series = pd.Series(metrics).round(4)
    print("\nModel metrics:")
    print(metrics_series)

    print("\nBuilding threshold table from out-of-fold train predictions...")
    threshold_df, _ = build_threshold_table(
        pipeline=best_pipeline,
        X_train=X_train,
        y_train=y_train,
    )

    best_threshold_row = select_best_threshold_by_f1(threshold_df)
    best_threshold = float(best_threshold_row["threshold"])

    print("\nBest threshold by F1:")
    print(best_threshold_row)

    print("\nSaving artifacts...")
    joblib.dump(best_pipeline, MODEL_PATH)
    threshold_df.to_csv(THRESHOLD_TABLE_PATH, index=False)

    metrics_payload = {k: float(v) for k, v in metrics.items()}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    training_summary = {
        "best_cv_average_precision": float(xgb_search.best_score_),
        "best_params": xgb_search.best_params_,
        "selected_threshold_metric": "f1",
        "selected_threshold": best_threshold,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_features": int(X.shape[1]),
    }
    with open(TRAINING_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    print("\nSaved:")
    print(f"- model: {MODEL_PATH}")
    print(f"- threshold table: {THRESHOLD_TABLE_PATH}")
    print(f"- metrics: {METRICS_PATH}")
    print(f"- training summary: {TRAINING_SUMMARY_PATH}")

    print("\nTraining pipeline complete.")


if __name__ == "__main__":
    main()