from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

def evaluate_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series) -> dict:

    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    y_train_prob = pipeline.predict_proba(X_train)[:, 1]
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {}

    # Train metrics
    metrics["train_accuracy"] = accuracy_score(y_train, y_train_pred)
    metrics["train_recall"] = recall_score(y_train, y_train_pred)
    metrics["train_precision"] = precision_score(y_train, y_train_pred)
    metrics["train_f1"] = f1_score(y_train, y_train_pred)
    metrics["train_roc_auc"] = roc_auc_score(y_train, y_train_prob)
    metrics["train_avg_precision"] = average_precision_score(y_train, y_train_prob)

    # Test metrics
    metrics["test_accuracy"] = accuracy_score(y_test, y_test_pred)
    metrics["test_recall"] = recall_score(y_test, y_test_pred)
    metrics["test_precision"] = precision_score(y_test, y_test_pred)
    metrics["test_f1"] = f1_score(y_test, y_test_pred)
    metrics["test_roc_auc"] = roc_auc_score(y_test, y_test_prob)
    metrics["test_avg_precision"] = average_precision_score(y_test, y_test_prob)

    return metrics

def precision_at_k(model, X, y, k):
    probs = model.predict_proba(X)[:, 1]

    y_array = np.asarray(y)

    df = pd.DataFrame({
        "prob": probs,
        "actual": y_array
    })

    top_k = df.sort_values("prob", ascending=False).head(k)
    return top_k