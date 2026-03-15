import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from flight_delay_prediction.config import SEED


def build_threshold_table(
    pipeline,
    X_train,
    y_train,
    thresholds=None,
    cv_splits=5,
):
    if thresholds is None:
        thresholds = np.arange(0.05, 1.00, 0.05)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)

    oof_probs = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    rows = []
    for t in thresholds:
        y_pred = (oof_probs >= t).astype(int)

        rows.append({
            "threshold": t,
            "precision": precision_score(y_train, y_pred, zero_division=0),
            "recall": recall_score(y_train, y_pred, zero_division=0),
            "f1": f1_score(y_train, y_pred, zero_division=0),
            "n_alerts": int(y_pred.sum()),
        })

    return pd.DataFrame(rows), oof_probs



def select_best_threshold_by_f1(threshold_df):
    return threshold_df.sort_values("f1", ascending=False).iloc[0]



def select_threshold_by_precision_floor(threshold_df, min_precision=0.40):
    valid = threshold_df[threshold_df["precision"] >= min_precision]
    if valid.empty:
        raise ValueError("No thresholds satisfy the precision floor.")
    return valid.sort_values("recall", ascending=False).iloc[0]