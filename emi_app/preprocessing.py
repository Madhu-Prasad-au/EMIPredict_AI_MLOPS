"""Exact inference-time reproduction of the training feature engineering.

The training notebook (EXP1(1).ipynb, cells 14 and 20) created 11 engineered
features before fitting the classification/regression pipelines. Those
pipelines contain the imputer, scaler and one-hot encoder, so the raw values
below are the only preprocessing this application must reproduce.

One training feature needs special handling:

    affordability_ratio = max_monthly_emi / monthly_salary

`max_monthly_emi` is the regression target and is not available at prediction
time. This application reconstructs it iteratively from the regression model's
own output (see emi_app/service.py). The reconstruction is documented in the UI
and README; it is the closest possible inference-time analogue of the training
feature.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from emi_app.schema import ApplicationInput, ENGINEERED_FIELDS, RAW_FIELDS

# Median target-to-disposable-income factor observed in the cleaned training
# data. Used only as the starting point of the fixed-point reconstruction.
BASE_AFFORDABILITY_FACTOR = 0.44
MAX_AFFORDABILITY_RATIO = 2.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Divide, returning NaN for a zero/invalid denominator.

    Training used `series / series.replace(0, np.nan)` and then replaced
    +/-inf with NaN. The saved pipeline's imputer is the component that turns
    those NaNs into training medians, so NaN is the faithful representation.
    """
    if denominator is None:
        return float("nan")
    try:
        den = float(denominator)
        num = float(numerator)
    except (TypeError, ValueError):
        return float("nan")
    if den == 0.0 or not math.isfinite(den) or not math.isfinite(num):
        return float("nan")
    value = num / den
    return value if math.isfinite(value) else float("nan")


def engineering_values(
    application: ApplicationInput,
    affordability_ratio: float,
) -> dict[str, float]:
    """Compute the 11 engineered features exactly as the notebook did."""
    salary = application.monthly_salary
    total_expenses = application.total_monthly_expenses
    total_debt = application.total_debt_burden
    remaining_income = application.remaining_income
    current_emi = application.current_emi_amount

    return {
        "total_monthly_expenses": total_expenses,
        "total_debt_burden": total_debt,
        "debt_to_income_ratio": _safe_ratio(total_debt, salary),
        "expense_to_income_ratio": _safe_ratio(total_expenses, salary),
        "current_emi_to_income": _safe_ratio(current_emi, salary),
        "loan_to_salary_ratio": _safe_ratio(application.requested_amount, salary),
        "balance_to_salary_ratio": _safe_ratio(application.bank_balance, salary),
        "emergency_coverage_ratio": _safe_ratio(
            application.emergency_fund, total_expenses
        ),
        "remaining_income": remaining_income,
        "estimated_requested_emi": _safe_ratio(
            application.requested_amount, application.requested_tenure
        ),
        "affordability_ratio": float(affordability_ratio),
    }


def base_affordability_ratio(application: ApplicationInput) -> float:
    """Transparent starting estimate for the target-derived feature.

    The training target is strongly related to income left after existing
    expenses and EMIs. The median ratio of target->remaining income is ~0.44,
    so 44% of positive remaining income is a conservative starting point. The
    service then iterates the regression model to a fixed point.
    """
    salary = application.monthly_salary
    if salary <= 0:
        return 0.0
    remaining = max(application.remaining_income, 0.0)
    ratio = BASE_AFFORDABILITY_FACTOR * remaining / salary
    return float(min(max(ratio, 0.0), MAX_AFFORDABILITY_RATIO))


def build_feature_frame(
    application: ApplicationInput,
    feature_names: Sequence[str],
    affordability_ratio: float,
    fill_values: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build the one-row DataFrame consumed by the saved sklearn pipelines.

    Args:
        application: validated applicant.
        feature_names: exact training-time feature order.
        affordability_ratio: reconstructed target-derived feature.
        fill_values: imputer statistics extracted from the saved pipeline.

    Returns:
        A one-row DataFrame with the exact feature names and order.
    """
    row: dict[str, object] = application.to_raw_dict()
    row.update(engineering_values(application, affordability_ratio))

    missing = [name for name in feature_names if name not in row]
    if missing:
        raise ValueError(
            "Model expects features that are not produced by the application: "
            + ", ".join(sorted(missing))
        )

    frame = pd.DataFrame([{name: row[name] for name in feature_names}])

    # Match the notebook's inf -> NaN step. The saved pipeline's imputer then
    # handles genuine NaNs with the training medians. We additionally fill
    # explicitly with those same imputer statistics where available, so the
    # pipeline receives the exact values it was fitted to expect.
    frame = frame.replace([np.inf, -np.inf], np.nan)

    if fill_values:
        for column in frame.columns:
            if column in fill_values and pd.isna(frame.at[0, column]):
                frame.at[0, column] = float(fill_values[column])

    # Keep categorical fields as strings for the OneHotEncoder.
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype(str)

    return frame


def feature_engineering_table(
    application: ApplicationInput, affordability_ratio: float
) -> pd.DataFrame:
    """Human-readable engineered feature table for the prediction page."""
    values = engineering_values(application, affordability_ratio)
    labels = {
        "total_monthly_expenses": "Total monthly expenses",
        "total_debt_burden": "Total debt burden (expenses + existing EMI)",
        "debt_to_income_ratio": "Debt-to-income ratio",
        "expense_to_income_ratio": "Expense-to-income ratio",
        "current_emi_to_income": "Current EMI-to-income ratio",
        "loan_to_salary_ratio": "Requested loan-to-salary ratio",
        "balance_to_salary_ratio": "Bank balance-to-salary ratio",
        "emergency_coverage_ratio": "Emergency-fund coverage (months)",
        "remaining_income": "Remaining monthly income",
        "estimated_requested_emi": "Estimated requested EMI (principal / tenure)",
        "affordability_ratio": "Affordability ratio (reconstructed)",
    }
    rows = []
    for name in ENGINEERED_FIELDS:
        value = values[name]
        rows.append(
            {
                "Feature": labels.get(name, name),
                "Field": name,
                "Value": value,
            }
        )
    return pd.DataFrame(rows)


def raw_input_table(application: ApplicationInput) -> pd.DataFrame:
    labels = {
        "age": "Age",
        "gender": "Gender",
        "marital_status": "Marital status",
        "education": "Education",
        "monthly_salary": "Monthly salary",
        "employment_type": "Employment type",
        "years_of_employment": "Years of employment",
        "company_type": "Company type",
        "house_type": "House type",
        "monthly_rent": "Monthly rent",
        "family_size": "Family size",
        "dependents": "Dependents",
        "school_fees": "School fees",
        "college_fees": "College fees",
        "travel_expenses": "Travel expenses",
        "groceries_utilities": "Groceries & utilities",
        "other_monthly_expenses": "Other monthly expenses",
        "existing_loans": "Existing loans",
        "current_emi_amount": "Current EMI amount",
        "credit_score": "Credit score",
        "bank_balance": "Bank balance",
        "emergency_fund": "Emergency fund",
        "emi_scenario": "EMI scenario",
        "requested_amount": "Requested amount",
        "requested_tenure": "Requested tenure (months)",
    }
    raw = application.to_raw_dict()

    def display(value: object) -> str:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)):
            return f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
        return str(value)

    return pd.DataFrame(
        [
            {
                "Feature": labels[name],
                "Field": name,
                "Value": display(raw[name]),
            }
            for name in RAW_FIELDS
        ]
    )
