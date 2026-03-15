from flight_delay_prediction.utils.metrics import evaluate_model
import pandas as pd

def compare_models(
    models: dict,
    X_train,
    y_train,
    X_test,
    y_test
) -> list[dict]:
    """
    Train and evaluate multiple models and return a comparison table.

    Parameters
    ----------
    models : dict
        Dictionary of {model_name: pipeline}
    """

    results = []

    for name, pipeline in models.items():

        print(f"Training {name}...")

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            pipeline,
            X_train,
            y_train,
            X_test,
            y_test
        )

        metrics["model"] = name

        results.append(metrics)

    return results