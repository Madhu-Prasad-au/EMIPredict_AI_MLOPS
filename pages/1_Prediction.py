"""EMI prediction and classification page."""

from __future__ import annotations

import json
import logging
from datetime import datetime

import pandas as pd
import streamlit as st

from emi_app import config
from emi_app.preprocessing import (
    feature_engineering_table,
    raw_input_table,
)
from emi_app.schema import (
    NUMERIC_SPECS,
    ApplicationInput,
    ValidationError,
    build_default_application,
    validate_application,
)
from emi_app.service import PredictionError, PredictionResult, predict
from emi_app.ui import (
    footer,
    format_currency,
    format_pct,
    get_artifacts_or_stop,
    page_header,
    setup_page,
    sidebar,
)

logger = logging.getLogger("emi_predict_ai.prediction")

setup_page("EMI Prediction", icon="🔮")
artifacts = get_artifacts_or_stop()
sidebar(artifacts)

page_header(
    "🔮 EMI Prediction & Classification",
    "Assess eligibility and estimate the maximum affordable monthly EMI. "
    "Requested EMI is calculated as loan amount ÷ tenure (principal only).",
)

default = build_default_application(artifacts.stats, artifacts.categorical_values)
categories = artifacts.categorical_values


def _category_index(field: str, value: str) -> int:
    options = categories[field]
    try:
        return options.index(value)
    except ValueError:
        return 0


def _number(label: str, field: str, *, integer: bool = False, step: float = 1.0, help_text: str | None = None):
    spec = NUMERIC_SPECS[field]
    default_value = getattr(default, field)
    return st.number_input(
        label,
        min_value=int(spec.minimum) if integer else float(spec.minimum),
        max_value=int(spec.maximum) if integer else float(spec.maximum),
        value=int(default_value) if integer else float(default_value),
        step=int(step) if integer else float(step),
        format="%d" if integer else "%.2f",
        help=help_text,
        key=f"predict_{field}",
    )


def _select(label: str, field: str, help_text: str | None = None):
    options = categories[field]
    default_value = getattr(default, field)
    return st.selectbox(
        label,
        options,
        index=_category_index(field, default_value),
        help=help_text,
        key=f"predict_{field}",
    )


def _render_probabilities(result: PredictionResult) -> None:
    if not result.probabilities:
        st.info("The classification model did not expose probabilities.")
        return
    probability_frame = pd.DataFrame(
        {
            "Class": list(result.probabilities.keys()),
            "Probability": [result.probabilities[name] for name in result.probabilities],
        }
    )
    probability_frame["Probability (%)"] = probability_frame["Probability"] * 100
    display_frame = probability_frame[["Class", "Probability (%)"]].copy()
    display_frame["Probability (%)"] = display_frame["Probability (%)"].map(
        lambda value: f"{value:.2f}%"
    )
    st.dataframe(display_frame, width="stretch", hide_index=True)
    st.bar_chart(
        probability_frame.set_index("Class")["Probability"],
        y_label="Probability",
        horizontal=False,
    )


def _render_result(result: PredictionResult, application: ApplicationInput) -> None:
    tone = result.decision_tone
    heading = f"{result.decision_icon} **{result.decision_label}**"
    if tone == "success":
        st.success(heading)
    elif tone == "warning":
        st.warning(heading)
    elif tone == "error":
        st.error(heading)
    else:
        st.info(heading)

    st.markdown(result.recommendation)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Eligibility class", result.eligibility)
    with m2:
        st.metric(
            "Maximum affordable EMI",
            format_currency(result.predicted_max_emi, 2),
        )
    with m3:
        st.metric("Requested EMI", format_currency(result.requested_emi, 2))
    with m4:
        st.metric(
            "Affordability gap",
            format_currency(result.affordability_gap, 2),
            delta="within limit" if result.is_affordable else "over limit",
            delta_color="normal" if result.is_affordable else "inverse",
        )

    if result.warnings:
        unique_warnings = list(dict.fromkeys(result.warnings))
        st.warning(
            "**Please review**\n\n"
            + "\n".join(f"- {warning}" for warning in unique_warnings)
        )

    st.divider()
    left, right = st.columns([1, 1.15])
    with left:
        st.subheader("Classification probabilities")
        _render_probabilities(result)
        st.caption(
            f"Predicted class confidence: **{format_pct(result.eligibility_probability)}**"
        )
    with right:
        st.subheader("Affordability assessment")
        comparison = pd.DataFrame(
            {
                "EMI": ["Requested EMI", "Maximum affordable EMI"],
                "Amount (₹)": [result.requested_emi, result.predicted_max_emi],
            }
        )
        st.bar_chart(comparison.set_index("EMI")["Amount (₹)"])
        st.caption(
            "Requested EMI = requested loan amount ÷ requested tenure. "
            "The dataset and saved models do not include an interest rate, so "
            "this is a principal-only estimate."
        )
        st.write(
            f"**Reconstructed affordability ratio:** {result.affordability_ratio:.4f} "
            f"(start: {result.base_affordability_ratio:.4f}, "
            f"iterations: {result.ratio_iterations}, "
            f"converged: {'yes' if result.ratio_converged else 'no'})"
        )
        st.write(
            f"**Models:** {result.model_names['classification']} + "
            f"{result.model_names['regression']} · "
            f"inference {result.inference_ms:.0f} ms"
        )

    st.divider()
    st.subheader("Financial ratios")
    ratios = pd.DataFrame(
        [
            {
                "Metric": "Total monthly expenses",
                "Value": format_currency(application.total_monthly_expenses),
            },
            {
                "Metric": "Total debt burden",
                "Value": format_currency(application.total_debt_burden),
            },
            {
                "Metric": "Remaining monthly income",
                "Value": format_currency(application.remaining_income),
            },
            {
                "Metric": "Affordability ratio (reconstructed)",
                "Value": f"{result.affordability_ratio:.2%}",
            },
            {
                "Metric": "Debt-to-income ratio",
                "Value": f"{result.engineered_features['debt_to_income_ratio']:.2%}"
                if result.engineered_features.get("debt_to_income_ratio") is not None
                else "—",
            },
            {
                "Metric": "Expense-to-income ratio",
                "Value": f"{result.engineered_features['expense_to_income_ratio']:.2%}"
                if result.engineered_features.get("expense_to_income_ratio") is not None
                else "—",
            },
        ]
    )
    st.dataframe(ratios, width="stretch", hide_index=True)

    with st.expander("🔧 Model input details"):
        st.write("**Raw applicant fields**")
        st.dataframe(raw_input_table(application), width="stretch", hide_index=True)
        st.write("**Engineered features (training reproduction)**")
        engineered = feature_engineering_table(application, result.affordability_ratio)
        engineered["Value"] = engineered["Value"].map(
            lambda value: f"{value:,.4f}" if isinstance(value, (int, float)) else "—"
        )
        st.dataframe(engineered, width="stretch", hide_index=True)
        st.caption(
            f"The saved pipeline expects {len(artifacts.feature_names)} features. "
            "The classifier and regressor each contain their own imputer, scaler, "
            "and one-hot encoder, so raw values are passed to the saved pipeline."
        )

    with st.expander("ℹ️ How this result is produced"):
        st.markdown(
            f"""
            **Classification** — `{artifacts.classification_model_name}` predicts
            one of {', '.join(artifacts.classification_classes)} and returns class
            probabilities.

            **Regression** — `{artifacts.regression_model_name}` predicts the
            maximum affordable monthly EMI.

            **Target-derived feature reconstruction** — at training time the
            feature `affordability_ratio` was `max_monthly_emi / monthly_salary`.
            Because the target is unknown at prediction time, the app starts from
            a conservative estimate (44% of positive remaining income over salary)
            and repeatedly runs the saved regression model, setting the ratio to
            `predicted_emi / salary`, until it stabilises. The reported prediction
            is always made with the final reconstructed ratio.

            **Interpretation** — the decision combines the predicted class with
            the affordability check. It is decision support, not a credit
            approval.
            """
        )

    st.download_button(
        "⬇️ Download this assessment (JSON)",
        data=json.dumps(result.to_dict(), indent=2, default=str),
        file_name="emi_assessment.json",
        mime="application/json",
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------
with st.form("emi_assessment_form", clear_on_submit=False):
    st.subheader("👤 Applicant profile")
    row1, row2, row3 = st.columns(3)
    with row1:
        age = _number("Age", "age", integer=True, step=1)
        gender = _select("Gender", "gender")
        marital_status = _select("Marital status", "marital_status")
    with row2:
        education = _select("Education", "education")
        employment_type = _select("Employment type", "employment_type")
        company_type = _select("Company type", "company_type")
    with row3:
        years_of_employment = _number(
            "Years of employment", "years_of_employment", step=0.5
        )
        house_type = _select("House type", "house_type")
        family_size = _number("Family size", "family_size", integer=True, step=1)

    st.subheader("💵 Income and monthly obligations")
    inc1, inc2, inc3 = st.columns(3)
    with inc1:
        monthly_salary = _number(
            "Monthly salary (₹)",
            "monthly_salary",
            step=1000.0,
            help_text="Must be greater than zero.",
        )
        monthly_rent = _number("Monthly rent (₹)", "monthly_rent", step=500.0)
        dependents = _number("Dependents", "dependents", integer=True, step=1)
    with inc2:
        school_fees = _number("School fees (₹)", "school_fees", step=500.0)
        college_fees = _number("College fees (₹)", "college_fees", step=500.0)
        travel_expenses = _number("Travel expenses (₹)", "travel_expenses", step=500.0)
    with inc3:
        groceries_utilities = _number(
            "Groceries & utilities (₹)", "groceries_utilities", step=500.0
        )
        other_monthly_expenses = _number(
            "Other monthly expenses (₹)", "other_monthly_expenses", step=500.0
        )

    st.subheader("🏦 Credit and loan request")
    cr1, cr2, cr3 = st.columns(3)
    with cr1:
        existing_loans = _select("Existing loans", "existing_loans")
        current_emi_amount = _number(
            "Current EMI amount (₹)", "current_emi_amount", step=500.0
        )
        credit_score = _number("Credit score", "credit_score", integer=True, step=1)
    with cr2:
        bank_balance = _number("Bank balance (₹)", "bank_balance", step=5000.0)
        emergency_fund = _number("Emergency fund (₹)", "emergency_fund", step=5000.0)
        emi_scenario = _select("EMI scenario", "emi_scenario")
    with cr3:
        requested_amount = _number(
            "Requested loan amount (₹)",
            "requested_amount",
            step=5000.0,
            help_text="Must be greater than zero.",
        )
        requested_tenure = _number(
            "Requested tenure (months)",
            "requested_tenure",
            integer=True,
            step=1,
            help_text="1–120 months.",
        )

    submitted = st.form_submit_button(
        "🔮 Run EMI assessment",
        type="primary",
        width="stretch",
    )

if submitted:
    payload = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
    }

    try:
        application = validate_application(payload, categories)
    except ValidationError as exc:
        st.error("Please fix the following input issues:")
        for error in exc.errors:
            st.markdown(f"- {error}")
    else:
        try:
            with st.spinner("Running the saved models..."):
                result = predict(application, artifacts)
        except PredictionError as exc:
            logger.exception("Prediction failed")
            st.error(f"Prediction could not be completed: {exc}")
            if config.DEBUG:
                st.exception(exc)
        else:
            st.session_state["last_prediction"] = {
                "application": application,
                "result": result,
            }
            history = st.session_state.setdefault("assessment_history", [])
            history.append(
                {
                    "timestamp": datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                    "application": application.to_summary_dict(),
                    "raw_input": application.to_raw_dict(),
                    "result": result.to_dict(),
                }
            )
            del history[:-50]
            st.divider()
            _render_result(result, application)

# Show the last result when the page reruns without a new submission (for
# example after opening an expander or downloading the JSON).
if not submitted and "last_prediction" in st.session_state:
    cached = st.session_state["last_prediction"]
    st.divider()
    st.caption("Showing the most recent assessment from this session.")
    _render_result(cached["result"], cached["application"])

with st.expander("Data quality and model caveats"):
    st.markdown(
        """
        - Inputs are validated before inference. Categorical options come from
          the saved one-hot encoder, not from hardcoded UI lists.
        - `credit_score` is validated against the standard 300–900 range. The
          cleaned training data contains a few 0 and 1200 values; those are
          treated as data-quality issues rather than valid user input.
        - The training target-derived `affordability_ratio` means the original
          regression metrics are optimistic. See the Model Performance page.
        - This page uses session-only state; no applicant data is written to
          disk or persisted on the server.
        """
    )

footer()
