# Extracted modules

This folder contains a direct modularization of the cleaning and feature-engineering code from the class notebook in `../../class/CIS_5450_Final_Project.ipynb` and the matching modeling metadata used in the notebook. The extraction preserves and refines the notebook logic rather than heavily redesigning the pipeline.

## Included modules

- `src/flight_delay_prediction/config.py`
- `src/flight_delay_prediction/utils/helpers.py`
- `src/flight_delay_prediction/utils/metrics.py`
- `src/flight_delay_prediction/utils/compare_models.py`
- `src/flight_delay_prediction/data/load_data.py`
- `src/flight_delay_prediction/data/preprocess.py`
- `src/flight_delay_prediction/data/download_bts.py`
- `src/flight_delay_prediction/data/fetch_weather.py`
- `src/flight_delay_prediction/features/build_features.py`
- `src/flight_delay_prediction/models/thresholding.py`
- `src/flight_delay_prediction/models/train.py`
- `src/flight_delay_prediction/models/xgb_pipeline.py`

## Notes
- The extraction tries to keep the same joins, drops, engineered columns, seasonal logic, time-of-day bucketing, cyclic transforms, and train/test split setup from the notebook.
