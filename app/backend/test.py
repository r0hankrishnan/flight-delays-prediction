from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_xgb_pipeline.joblib"

print("=" * 50)
print(PROJECT_ROOT)
print(DEFAULT_MODEL_PATH)