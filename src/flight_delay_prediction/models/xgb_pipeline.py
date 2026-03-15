from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from flight_delay_prediction.config import SEED
from flight_delay_prediction.features import build_preprocessor


def build_xgb_pipeline(
    oh_encoding_cols,
    target_encoding_cols,
    y_train,
):
    preprocessor = build_preprocessor(
        oh_encoding_cols=oh_encoding_cols,
        target_encoding_cols=target_encoding_cols,
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("xgb", XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=SEED,
        )),
    ])

    return pipeline


def get_xgb_param_space():
    return {
        "xgb__n_estimators": [100, 200, 300, 500],
        "xgb__max_depth": [3, 4, 5, 6],
        "xgb__learning_rate": [0.03, 0.05, 0.1],
        "xgb__min_child_weight": [1, 3, 5],
        "xgb__subsample": [0.7, 0.85, 1.0],
        "xgb__colsample_bytree": [0.7, 0.85, 1.0],
    }