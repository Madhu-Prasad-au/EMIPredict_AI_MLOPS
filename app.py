"""EMIPredict AI — home / overview page.

Run with:
    python entrypoint.py          # preloads artifacts, then starts Streamlit
    streamlit run app.py          # development alternative
"""

from __future__ import annotations

import streamlit as st

from emi_app import config
from emi_app.data import DataAssetError, load_eda_sample, load_reports, load_summary
from emi_app.ui import (
    footer,
    format_currency,
    format_number,
    get_artifacts_or_stop,
    get_data_or_stop,
    hero,
    setup_page,
    sidebar,
    status_pill,
)

setup_page("Overview", icon="💰")
artifacts = get_artifacts_or_stop()
sidebar(artifacts)

summary = get_data_or_stop(load_summary, "EDA summary")
reports = get_data_or_stop(load_reports, "Model performance reports")

hero(
    "EMIPredict AI",
    "EMI eligibility classification and maximum-affordable-EMI prediction, "
    "served from the trained artifacts.",
)

st.markdown(
    status_pill(
        True,
        f"Artifacts loaded · {artifacts.load_seconds:.1f}s startup",
    ),
    unsafe_allow_html=True,
)
st.write("")

total_records = int(summary.get("records", 0))
class_counts = summary.get("class_counts", {})
scenario_counts = summary.get("scenario_counts", {})
classification_metrics = reports.classification_final_metrics
regression_metrics = reports.regression_final_metrics

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Training records", f"{total_records:,}")
with c2:
    st.metric("Input features", len(artifacts.feature_names))
with c3:
    st.metric("Eligibility classes", len(artifacts.classification_classes))
with c4:
    st.metric("EMI scenarios", len(scenario_counts) or 5)

st.divider()

st.subheader("What the application does")
left, right = st.columns([1.15, 1])
with left:
    st.markdown(
        """
        **1. Classification** answers *"Is this applicant eligible for the requested EMI?"*
        using the saved XGBoost pipeline. The three classes are
        **Eligible**, **High_Risk**, and **Not_Eligible**.

        **2. Regression** estimates the **maximum affordable monthly EMI** using
        the saved Random Forest pipeline.

        **3. Decision support** combines both outputs with the requested EMI
        (loan amount ÷ tenure) and highlights whether the request is within the
        estimated affordability limit.
        """
    )
    st.markdown(
        f"""
        <div class="emi-muted">
        Best classification model: <b>{artifacts.classification_model_name}</b>
        (held-out F1 {classification_metrics.get('F1 Score', float('nan')):.4f}) ·
        Best regression model: <b>{artifacts.regression_model_name}</b>
        (held-out R² {regression_metrics.get('R2 Score', float('nan')):.4f})
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.info(
        "All artifacts are loaded once at process startup. Streamlit reruns, "
        "new sessions, and page navigation reuse the in-memory pipelines."
    )
    st.warning(
        "Model limitation: the training data included an `affordability_ratio` "
        "feature derived from the regression target. The app reconstructs it "
        "iteratively and flags approximate results. See Model Performance."
    )

st.divider()
st.subheader("Open a workspace")

nav1, nav2, nav3 = st.columns(3)
with nav1:
    with st.container(border=True):
        st.markdown("#### 🔮 EMI assessment")
        st.caption("Enter applicant details and get both predictions.")
        st.page_link(
            "pages/1_Prediction.py",
            label="Run an assessment",
            width="stretch",
        )
with nav2:
    with st.container(border=True):
        st.markdown("#### 📊 EDA analytics")
        st.caption("Filter the 60k stratified sample and inspect distributions.")
        st.page_link(
            "pages/2_EDA_Analytics.py",
            label="Explore the data",
            width="stretch",
        )
with nav3:
    with st.container(border=True):
        st.markdown("#### 🤖 Model performance")
        st.caption("Compare the saved classification and regression results.")
        st.page_link(
            "pages/3_Model_Performance.py",
            label="View metrics",
            width="stretch",
        )

st.divider()
st.subheader("Production models")

model_left, model_right = st.columns(2)
with model_left:
    with st.container(border=True):
        st.markdown("### Classification")
        st.success(artifacts.classification_model_name)
        st.write("Predicts EMI eligibility:")
        st.write(", ".join(artifacts.classification_classes))
        st.metric(
            "Held-out F1 score",
            f"{classification_metrics.get('F1 Score', 0.0):.4f}",
        )
        st.metric(
            "Held-out ROC-AUC",
            f"{classification_metrics.get('ROC-AUC', 0.0):.4f}",
        )
with model_right:
    with st.container(border=True):
        st.markdown("### Regression")
        st.success(artifacts.regression_model_name)
        st.write("Predicts maximum safe monthly EMI.")
        st.metric(
            "Held-out R²",
            f"{regression_metrics.get('R2 Score', 0.0):.4f}",
        )
        st.metric(
            "Held-out RMSE",
            format_currency(regression_metrics.get("RMSE", 0.0), 2),
        )

st.divider()
st.subheader("Pipeline")
st.markdown(
    """
    **Dataset** → **Cleaning** → **EDA & feature engineering** →
    **Classification + regression** → **Model comparison** →
    **Saved artifacts** → **Real-time prediction**
    """
)

st.divider()
st.subheader("Dataset snapshot")

try:
    sample = load_eda_sample()
except DataAssetError as exc:
    st.warning(f"Dataset preview unavailable: {exc}")
else:
    preview_columns = [
        "emi_scenario",
        "monthly_salary",
        "credit_score",
        "requested_amount",
        "requested_tenure",
        "max_monthly_emi",
        "emi_eligibility",
    ]
    available = [column for column in preview_columns if column in sample.columns]
    st.caption(
        f"Stratified sample: {len(sample):,} rows · "
        f"full cleaned dataset: {total_records:,} rows"
    )
    st.dataframe(
        sample[available].head(8),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Full-dataset class balance: "
        + " · ".join(
            f"{name}: {format_number(count)}"
            for name, count in class_counts.items()
        )
    )

with st.expander("Model card, artifact provenance, and limitations"):
    st.markdown(
        f"""
        **Artifacts**

        | Artifact | File | Loaded |
        |---|---|---|
        | Classification | `{artifacts.classification_path.name}` | {artifacts.classification_model_name} |
        | Regression | `{artifacts.regression_path.name}` | {artifacts.regression_model_name} |
        | Label encoder | `{config.LABEL_ENCODER_PATH.name}` | {len(artifacts.classification_classes)} classes |
        | Feature metadata | `{config.FEATURE_INFO_PATH.name}` | {len(artifacts.feature_names)} features |

        **Known modelling issue**

        During training, `affordability_ratio` was computed as
        `max_monthly_emi / monthly_salary`. Because `max_monthly_emi` is the
        regression target, this is target leakage. The saved model cannot be
        evaluated honestly without that value. This application starts from a
        transparent income-based estimate and iterates the saved regression
        model to a self-consistent ratio. Reported held-out metrics are the
        original notebook metrics and are therefore optimistic.

        **Not financial advice.** This is a demonstration application; use a
        validated, leakage-free model and human underwriting for real decisions.
        """
    )
    st.markdown(
        f"Runtime: scikit-learn `{artifacts.runtime_versions.get('scikit-learn')}`, "
        f"xgboost `{artifacts.runtime_versions.get('xgboost')}`, "
        f"pandas `{artifacts.runtime_versions.get('pandas')}`, "
        f"Streamlit `{artifacts.runtime_versions.get('streamlit')}`."
    )

footer()
