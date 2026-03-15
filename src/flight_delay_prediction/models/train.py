from sklearn.model_selection import RandomizedSearchCV

from flight_delay_prediction.config import SEED
from flight_delay_prediction.models.xgb_pipeline import (
    build_xgb_pipeline,
    get_xgb_param_space,
)


def tune_xgb_model(
    X_train,
    y_train,
    oh_encoding_cols,
    target_encoding_cols,
    n_iter=10,
    scoring="average_precision",
    cv=5,
    n_jobs=-1,
    verbose=2,
):
    pipeline = build_xgb_pipeline(
        oh_encoding_cols=oh_encoding_cols,
        target_encoding_cols=target_encoding_cols,
        y_train=y_train,
    )

    param_space = get_xgb_param_space()

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=SEED,
    )

    search.fit(X_train, y_train)
    return search