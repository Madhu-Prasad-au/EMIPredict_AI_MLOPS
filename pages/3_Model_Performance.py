"""Model performance page backed by the saved comparison reports."""

from __future__ import annotations

import json

import streamlit as st

from emi_app import config
from emi_app.data import load_reports
from emi_app.ui import (
    footer,
    format_currency,
    format_pct,
    get_artifacts_or_stop,
    get_data_or_stop,
    page_header,
    setup_page,
    sidebar,
)

setup_page("Model Performance", icon="🤖")
artifacts = get_artifacts_or_stop()
sidebar(artifacts)

if artifacts.profile == "lite":
    metrics = artifacts.metrics
    classification = metrics.get("classification", {})
    regression = metrics.get("regression", {})
    page_header(
        "🤖 Model performance — free/lite profile",
        "Held-out test metrics for the compact, leakage-free model pair used on "
        "small 512 MB instances.",
    )
    st.warning(
        "This is a **new compact model**, not the saved full artifact. It was "
        "trained without the target-derived `affordability_ratio`, so its "
        "predictions and metrics differ from the original notebook results. "
        "Do not compare these numbers directly to the full model's leaky metrics."
    )

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("### Classification")
            st.success(artifacts.classification_model_name)
            st.metric("Held-out F1", f"{classification.get('F1 Score', 0):.4f}")
            st.metric("Held-out ROC-AUC", f"{classification.get('ROC-AUC', 0):.4f}")
            st.caption(
                f"Accuracy {classification.get('Accuracy', 0):.4f} · "
                f"Precision {classification.get('Precision', 0):.4f} · "
                f"Recall {classification.get('Recall', 0):.4f}"
            )
    with c2:
        with st.container(border=True):
            st.markdown("### Regression")
            st.success(artifacts.regression_model_name)
            st.metric(
                "Held-out RMSE",
                format_currency(regression.get("RMSE", 0), 2),
            )
            st.metric("Held-out R²", f"{regression.get('R2 Score', 0):.4f}")
            st.caption(
                f"MAE {format_currency(regression.get('MAE', 0), 2)} · "
                f"MAPE {regression.get('MAPE (%)', 0):.2f}%"
            )

    st.subheader("Evaluation details")
    st.json(metrics, expanded=False)
    st.caption(
        f"Training rows: {metrics.get('training_rows', 0):,} · "
        f"test rows: {metrics.get('test_rows', 0):,} · "
        f"features: {metrics.get('feature_count', len(artifacts.feature_names))}. "
        "The compact profile is intentionally less expressive than the full "
        "Random Forest, but it fits a 512 MB instance and has no target leakage."
    )
    st.download_button(
        "⬇️ Download lite metrics (JSON)",
        data=json.dumps(metrics, indent=2, default=str),
        file_name="lite_metrics.json",
        mime="application/json",
        width="stretch",
    )
    footer()
    st.stop()

reports = get_data_or_stop(load_reports, "Model performance reports")

page_header(
    "🤖 Model performance",
    "Saved validation and held-out test results from the training notebook. "
    "Values are read from reports/*.csv; nothing is re-evaluated at runtime.",
)

classification_final = reports.classification_final_metrics
regression_final = reports.regression_final_metrics
best_classification = reports.best_classification_row
best_regression = reports.best_regression_row

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("### Best classification model")
        st.success(str(best_classification["Model"]))
        st.metric("Held-out F1", f"{classification_final.get('F1 Score', 0):.4f}")
        st.metric("Held-out ROC-AUC", f"{classification_final.get('ROC-AUC', 0):.4f}")
        st.caption(
            f"Selected on validation F1 ({float(best_classification['F1 Score']):.4f})."
        )
with c2:
    with st.container(border=True):
        st.markdown("### Best regression model")
        st.success(str(best_regression["Model"]))
        st.metric(
            "Held-out RMSE",
            format_currency(regression_final.get("RMSE", 0), 2),
        )
        st.metric("Held-out R²", f"{regression_final.get('R2 Score', 0):.4f}")
        st.caption(
            f"Selected on validation RMSE ({format_currency(float(best_regression['RMSE']), 2)})."
        )

st.warning(
    "**Important limitation:** `affordability_ratio` was engineered from the "
    "regression target (`max_monthly_emi / monthly_salary`) during training. "
    "The regression metrics below are therefore optimistic and cannot be "
    "reproduced in production without the target. The application reconstructs "
    "the feature iteratively; see the notes tab and README."
)

tab_class, tab_reg, tab_notes = st.tabs(
    ["Classification", "Regression", "Evaluation notes"]
)

with tab_class:
    st.subheader("Held-out test metrics")
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        st.metric("Accuracy", f"{classification_final.get('Accuracy', 0):.4f}")
    with f2:
        st.metric("Precision", f"{classification_final.get('Precision', 0):.4f}")
    with f3:
        st.metric("Recall", f"{classification_final.get('Recall', 0):.4f}")
    with f4:
        st.metric("F1 score", f"{classification_final.get('F1 Score', 0):.4f}")
    with f5:
        st.metric("ROC-AUC", f"{classification_final.get('ROC-AUC', 0):.4f}")

    st.subheader("Validation comparison")
    st.dataframe(
        reports.classification_comparison,
        width="stretch",
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### F1 score by model")
        st.bar_chart(
            reports.classification_comparison.set_index("Model")[["F1 Score"]]
        )
    with right:
        st.markdown("#### ROC-AUC by model")
        st.bar_chart(
            reports.classification_comparison.set_index("Model")[["ROC-AUC"]]
        )

    st.caption(
        "Classes: " + ", ".join(artifacts.classification_classes) + ". "
        "Metrics are weighted averages where applicable."
    )

with tab_reg:
    st.subheader("Held-out test metrics")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric(
            "RMSE",
            format_currency(regression_final.get("RMSE", 0), 2),
        )
    with r2:
        st.metric("MAE", format_currency(regression_final.get("MAE", 0), 2))
    with r3:
        st.metric("R²", f"{regression_final.get('R2 Score', 0):.4f}")
    with r4:
        st.metric("MAPE", format_pct(regression_final.get("MAPE (%)", 0) / 100))

    st.subheader("Validation comparison")
    st.dataframe(
        reports.regression_comparison,
        width="stretch",
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### RMSE by model (lower is better)")
        st.bar_chart(
            reports.regression_comparison.set_index("Model")[["RMSE"]]
        )
    with right:
        st.markdown("#### R² by model (higher is better)")
        st.bar_chart(
            reports.regression_comparison.set_index("Model")[["R2 Score"]]
        )

    st.caption(
        "MAPE is shown as a percentage. RMSE and MAE are in Indian rupees (₹)."
    )

with tab_notes:
    st.markdown(
        f"""
        **Artifacts used**

        - Classification: `{artifacts.classification_path.name}` →
          **{artifacts.classification_model_name}**
        - Regression: `{artifacts.regression_path.name}` →
          **{artifacts.regression_model_name}**
        - Label encoder: `{len(artifacts.classification_classes)}` classes
        - Saved reports: `reports/*.csv` (validation comparison and held-out test)

        **Evaluation protocol (from the notebook)**

        - 70% train, 15% validation, 15% test; random seed 42.
        - Classification splits are stratified; regression splits are not.
        - The comparison tables are validation results used for model selection.
        - The final metric cards are the held-out test results saved in
          `reports/final_*_metrics.csv`.

        **Why the regression metrics are optimistic**

        The training notebook created:

        ```text
        affordability_ratio = max_monthly_emi / monthly_salary
        ```

        and then included `affordability_ratio` in the feature matrix for both
        models. `max_monthly_emi` is the regression target. The Random Forest
        can therefore partly read the answer from an input. The 0.998 R² is not a
        realistic production estimate. Do not quote it as model accuracy.

        The app's inference-time reconstruction (see the Prediction page) starts
        from income and iterates the saved model. A leakage-free retrain is the
        correct long-term fix, but this project intentionally does not retrain
        because the saved artifacts are loadable.

        **Other caveats**

        - The dataset appears synthetic/simulated; validate on real lending data
          before any production credit use.
        - No fairness, calibration, or out-of-time stability analysis is
          included in the saved reports.
        - This application is decision support, not an automated credit decision.
        """
    )

st.divider()
st.subheader("Download saved reports")
d1, d2, d3, d4 = st.columns(4)
for column, (label, key) in zip(
    (d1, d2, d3, d4),
    (
        ("Classification comparison", "classification_comparison"),
        ("Regression comparison", "regression_comparison"),
        ("Final classification", "classification_final"),
        ("Final regression", "regression_final"),
    ),
):
    with column:
        path = config.REPORTS[key]
        st.download_button(
            label,
            data=path.read_bytes(),
            file_name=path.name,
            mime="text/csv",
            width="stretch",
        )

footer()
