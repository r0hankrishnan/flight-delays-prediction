from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
import streamlit as st


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="PHL Delay Triage Demo",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------
# Paths
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data" / "external" / "inference"

MODEL_PATH = ARTIFACTS_DIR / "best_xgb_pipeline.joblib"
DATASET_PATHS = {
    "small": DATA_DIR / "demo_small_inference.csv",
    "medium": DATA_DIR / "demo_medium_inference.csv",
    "large": DATA_DIR / "demo_large_inference.csv",
}
TOP_K_OPTIONS = [5, 10, 50, 100, 250]


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .hero-card {
            background: white;
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 8px 24px rgba(19, 48, 86, 0.08);
            border: 1px solid rgba(19, 48, 86, 0.08);
            margin-bottom: 1rem;
        }
        .section-card {
            background: white;
            border-radius: 18px;
            padding: 1rem 1.2rem;
            box-shadow: 0 8px 24px rgba(19, 48, 86, 0.06);
            border: 1px solid rgba(19, 48, 86, 0.08);
            margin-top: 1rem;
        }
        .small-note {
            color: #4f6177;
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_demo_datasets() -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for name, path in DATASET_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Demo dataset not found at {path}")
        df = pd.read_csv(path)
        df = df.reset_index(names="flight_id")
        datasets[name] = df
    return datasets


# -----------------------------
# Helpers
# -----------------------------
def get_preview_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "flight_id",
        "FlightDate",
        "Origin",
        "Dest",
        "airline_name",
        "CRSElapsedTime",
        "AirTime",
        "Distance",
    ]
    return [col for col in preferred if col in df.columns]


@st.cache_data(show_spinner=False)
def run_inference(dataset_name: str, top_k: int) -> pd.DataFrame:
    model = load_model()
    datasets = load_demo_datasets()
    inference_df = datasets[dataset_name].copy()

    delay_probs = model.predict_proba(inference_df)[:, 1]
    inference_df["prob_delayed"] = delay_probs

    sort_cols = ["prob_delayed"]
    ranked_df = inference_df.sort_values(sort_cols, ascending=False).head(top_k).copy()

    display_cols = ["flight_id"]
    for col in ["FlightDate", "Origin", "Dest", "airline_name"]:
        if col in ranked_df.columns:
            display_cols.append(col)
    display_cols.append("prob_delayed")

    rename_map = {
        "flight_id": "Flight ID",
        "FlightDate": "Flight Date",
        "Origin": "Origin Airport",
        "Dest": "Destination Airport",
        "airline_name": "Airline",
        "prob_delayed": "Probability of Delayed Arrival",
    }

    ranked_df = ranked_df[display_cols].rename(columns=rename_map)
    return ranked_df


def style_probability_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    prob_col = "Probability of Delayed Arrival"

    def color_prob(val: float) -> str:
        if val < 0.40:
            return "background-color: #d1fae5; color: #065f46; font-weight: 700;"
        if val < 0.60:
            return "background-color: #fef3c7; color: #92400e; font-weight: 700;"
        return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"

    styled = (
        df.style
        .format({prob_col: "{:.3f}"})
        .applymap(color_prob, subset=[prob_col])
        .set_properties(**{
            "text-align": "left",
            "border-color": "#d7e0ea",
        })
        .set_table_styles([
            {"selector": "th", "props": [("background-color", "#eef4fb"), ("color", "#183153"), ("font-weight", "600")]},
            {"selector": "td", "props": [("padding", "0.5rem 0.65rem")]},
            {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%")]},
        ])
    )
    return styled


# -----------------------------
# App
# -----------------------------
model = load_model()
datasets = load_demo_datasets()

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin-bottom: 0.25rem; color: #183153;">PHL Delay Triage Demo</h1>
        <p class="small-note" style="margin-bottom: 0;">
            Load a cached demo batch, inspect the inference inputs, and rank flights by predicted delay risk.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b = st.columns([3, 1])
with col_a:
    preview_dataset_name = st.segmented_control(
        "Preview inference dataset",
        options=list(DATASET_PATHS.keys()),
        default="small",
        key="preview_dataset_name",
    )
    if preview_dataset_name is None:
        preview_dataset_name = "small"
with col_b:
    with st.popover("Model info"):
        st.code(repr(model), language="python")

preview_df = datasets[preview_dataset_name]
preview_cols = get_preview_columns(preview_df)

#st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Inference data preview")
st.caption(f"Showing the cached '{preview_dataset_name}' demo dataset: {len(preview_df):,} rows")
st.dataframe(
    preview_df[preview_cols] if preview_cols else preview_df,
    use_container_width=True,
    hide_index=True,
)
#st.markdown('</div>', unsafe_allow_html=True)

#st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Run predictions")
st.caption("Select a cached batch and a top-k triage size, then rank the flights by delay probability.")

with st.form("prediction_form"):
    selected_dataset_name = st.selectbox(
        "Inference dataset",
        options=list(DATASET_PATHS.keys()),
        index=list(DATASET_PATHS.keys()).index(preview_dataset_name),
        help="Choose which cached synthetic inference dataset to score.",
    )

    top_k = st.selectbox(
        "Top-k flights to show",
        options=TOP_K_OPTIONS,
        index=1,
        help="Return the flights with the highest predicted probability of delayed arrival.",
    )

    run_predictions = st.form_submit_button("Run predictions", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if run_predictions:
    with st.spinner("Scoring flights and building triage table..."):
        ranked_df = run_inference(selected_dataset_name, top_k)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Airport operations triage table")
    st.caption(
        f"Top {top_k} flights from the '{selected_dataset_name}' dataset, ranked by predicted delay probability."
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Dataset size", f"{len(datasets[selected_dataset_name]):,}")
    metric_col2.metric("Displayed flights", f"{len(ranked_df):,}")
    metric_col3.metric(
        "Highest risk",
        f"{ranked_df['Probability of Delayed Arrival'].max():.3f}",
    )

    st.dataframe(
        style_probability_table(ranked_df),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Choose a dataset and top-k size, then click 'Run predictions' to generate the ranked triage table.")
