"""Input schema, validation, and default applicant construction."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

# Raw fields exactly as they appeared in the training dataset (25 columns).
RAW_FIELDS: tuple[str, ...] = (
    "age",
    "gender",
    "marital_status",
    "education",
    "monthly_salary",
    "employment_type",
    "years_of_employment",
    "company_type",
    "house_type",
    "monthly_rent",
    "family_size",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "existing_loans",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "emi_scenario",
    "requested_amount",
    "requested_tenure",
)

# Features engineered by the training notebook, in training order.
ENGINEERED_FIELDS: tuple[str, ...] = (
    "total_monthly_expenses",
    "total_debt_burden",
    "debt_to_income_ratio",
    "expense_to_income_ratio",
    "current_emi_to_income",
    "loan_to_salary_ratio",
    "balance_to_salary_ratio",
    "emergency_coverage_ratio",
    "remaining_income",
    "estimated_requested_emi",
    "affordability_ratio",
)

CATEGORICAL_FIELDS: tuple[str, ...] = (
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "company_type",
    "house_type",
    "existing_loans",
    "emi_scenario",
)

FIELD_LABELS = {
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
    "requested_amount": "Requested loan amount",
    "requested_tenure": "Requested tenure (months)",
}


@dataclass(frozen=True)
class NumericFieldSpec:
    label: str
    minimum: float
    maximum: float
    integer: bool = False
    unit: str = ""


NUMERIC_SPECS: dict[str, NumericFieldSpec] = {
    "age": NumericFieldSpec("Age", 18, 100, True, "years"),
    "monthly_salary": NumericFieldSpec("Monthly salary", 1, 100_000_000, False, "₹"),
    "years_of_employment": NumericFieldSpec(
        "Years of employment", 0, 60, False, "years"
    ),
    "monthly_rent": NumericFieldSpec("Monthly rent", 0, 10_000_000, False, "₹"),
    "family_size": NumericFieldSpec("Family size", 1, 30, True, "people"),
    "dependents": NumericFieldSpec("Dependents", 0, 30, True, "people"),
    "school_fees": NumericFieldSpec("School fees", 0, 10_000_000, False, "₹"),
    "college_fees": NumericFieldSpec("College fees", 0, 10_000_000, False, "₹"),
    "travel_expenses": NumericFieldSpec("Travel expenses", 0, 10_000_000, False, "₹"),
    "groceries_utilities": NumericFieldSpec(
        "Groceries & utilities", 0, 10_000_000, False, "₹"
    ),
    "other_monthly_expenses": NumericFieldSpec(
        "Other monthly expenses", 0, 10_000_000, False, "₹"
    ),
    "current_emi_amount": NumericFieldSpec(
        "Current EMI amount", 0, 10_000_000, False, "₹"
    ),
    "credit_score": NumericFieldSpec("Credit score", 300, 900, True, "points"),
    "bank_balance": NumericFieldSpec("Bank balance", 0, 1_000_000_000, False, "₹"),
    "emergency_fund": NumericFieldSpec("Emergency fund", 0, 1_000_000_000, False, "₹"),
    "requested_amount": NumericFieldSpec(
        "Requested loan amount", 1, 100_000_000, False, "₹"
    ),
    "requested_tenure": NumericFieldSpec(
        "Requested tenure", 1, 120, True, "months"
    ),
}


class ValidationError(ValueError):
    """Raised when one or more applicant fields are invalid."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class ApplicationInput:
    """Validated, normalized applicant data."""

    age: int
    gender: str
    marital_status: str
    education: str
    monthly_salary: float
    employment_type: str
    years_of_employment: float
    company_type: str
    house_type: str
    monthly_rent: float
    family_size: int
    dependents: int
    school_fees: float
    college_fees: float
    travel_expenses: float
    groceries_utilities: float
    other_monthly_expenses: float
    existing_loans: str
    current_emi_amount: float
    credit_score: int
    bank_balance: float
    emergency_fund: float
    emi_scenario: str
    requested_amount: float
    requested_tenure: int

    @property
    def total_monthly_expenses(self) -> float:
        return (
            self.monthly_rent
            + self.school_fees
            + self.college_fees
            + self.travel_expenses
            + self.groceries_utilities
            + self.other_monthly_expenses
        )

    @property
    def total_debt_burden(self) -> float:
        return self.current_emi_amount + self.total_monthly_expenses

    @property
    def remaining_income(self) -> float:
        return self.monthly_salary - self.total_debt_burden

    @property
    def requested_emi(self) -> float:
        """Principal-only requested EMI (amount / tenure).

        The training data does not contain an interest rate, so this is the
        only requested-EMI definition reproducible from the saved artifacts.
        """
        return self.requested_amount / max(self.requested_tenure, 1)

    def to_raw_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "age": self.age,
            "gender": self.gender,
            "monthly_salary": self.monthly_salary,
            "credit_score": self.credit_score,
            "emi_scenario": self.emi_scenario,
            "requested_amount": self.requested_amount,
            "requested_tenure": self.requested_tenure,
        }


def _coerce_number(field: str, value: Any, spec: NumericFieldSpec) -> float:
    label = spec.label
    if value is None:
        raise ValueError(f"{label} is required.")
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number, not a boolean.")
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            raise ValueError(f"{label} is required.")
        try:
            number = float(cleaned)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number.") from exc
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise ValueError(f"{label} must be a number.")

    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    if spec.integer and not float(number).is_integer():
        raise ValueError(f"{label} must be a whole number.")
    if number < spec.minimum or number > spec.maximum:
        low = f"{spec.minimum:,.0f}" if spec.integer else f"{spec.minimum:,.2f}"
        high = f"{spec.maximum:,.0f}" if spec.integer else f"{spec.maximum:,.2f}"
        raise ValueError(f"{label} must be between {low} and {high}.")
    return number


def _coerce_category(
    field: str, value: Any, allowed: Sequence[str], label: str
) -> str:
    if value is None:
        raise ValueError(f"{label} is required.")
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")
    candidate = value.strip()
    if not candidate:
        raise ValueError(f"{label} is required.")
    if candidate in allowed:
        return candidate
    folded = candidate.casefold()
    matches = [item for item in allowed if item.casefold() == folded]
    if len(matches) == 1:
        return matches[0]
    options = ", ".join(f"'{item}'" for item in allowed)
    raise ValueError(f"{label} must be one of: {options}. Got '{candidate}'.")


def validate_application(
    payload: Mapping[str, Any],
    categories: Mapping[str, Sequence[str]],
) -> ApplicationInput:
    """Validate and normalize a raw applicant payload.

    Raises:
        ValidationError: with one human-readable message per invalid field.
    """
    if not isinstance(payload, Mapping):
        raise ValidationError(["Application data must be a mapping of fields."])

    errors: list[str] = []
    values: dict[str, Any] = {}

    missing_categories = [f for f in CATEGORICAL_FIELDS if f not in categories]
    if missing_categories:
        raise ValidationError(
            [
                "Model categorical metadata is incomplete for: "
                + ", ".join(missing_categories)
            ]
        )

    for field in RAW_FIELDS:
        if field not in payload:
            errors.append(f"{FIELD_LABELS.get(field, field)} is required.")
            continue

        value = payload[field]
        try:
            if field in NUMERIC_SPECS:
                number = _coerce_number(field, value, NUMERIC_SPECS[field])
                if NUMERIC_SPECS[field].integer:
                    values[field] = int(number)
                else:
                    values[field] = float(number)
            else:
                allowed = list(categories[field])
                values[field] = _coerce_category(
                    field, value, allowed, FIELD_LABELS.get(field, field)
                )
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        raise ValidationError(errors)

    return ApplicationInput(**{field: values[field] for field in RAW_FIELDS})


def application_warnings(
    application: ApplicationInput,
    stats: Mapping[str, Any] | None = None,
) -> list[str]:
    """Non-blocking data-quality warnings shown next to the prediction."""
    warnings: list[str] = []

    if (
        application.existing_loans.strip().casefold() == "no"
        and application.current_emi_amount > 0
    ):
        warnings.append(
            "Existing loans is 'No' but the current EMI is greater than zero. "
            "The model will use both values as provided."
        )

    expenses = application.total_monthly_expenses
    if expenses > application.monthly_salary:
        warnings.append(
            "Monthly expenses exceed monthly salary. The applicant is likely to "
            "be classified as high risk."
        )

    if application.requested_emi > application.monthly_salary:
        warnings.append(
            "The requested monthly EMI is larger than the applicant's salary."
        )

    if application.emergency_fund < expenses:
        warnings.append(
            "The emergency fund covers less than one month of expenses."
        )

    if stats:
        ranges = stats.get("observed_ranges", {})
        tenure_range = ranges.get("requested_tenure", {})
        observed_max = tenure_range.get("max")
        if observed_max and application.requested_tenure > observed_max:
            warnings.append(
                f"Requested tenure ({application.requested_tenure} months) is above "
                f"the maximum seen in training ({int(observed_max)} months). "
                "Treat the prediction as an extrapolation."
            )
        amount_range = ranges.get("requested_amount", {})
        observed_amount_max = amount_range.get("max")
        if observed_amount_max and application.requested_amount > observed_amount_max:
            warnings.append(
                "Requested amount is above the maximum seen in training; the "
                "prediction is an extrapolation."
            )

    return warnings


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_default_application(
    stats: Mapping[str, Any],
    categories: Mapping[str, Sequence[str]],
) -> ApplicationInput:
    """Build a realistic default applicant from training medians/modes."""
    medians: Mapping[str, Any] = stats.get("numeric_medians", {})
    modes: Mapping[str, Any] = stats.get("categorical_modes", {})

    def numeric(field: str, fallback: float) -> float:
        value = medians.get(field)
        if value is None:
            return fallback
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def category(field: str, fallback: str) -> str:
        allowed = list(categories.get(field, []))
        preferred = str(modes.get(field, fallback)).strip()
        if preferred in allowed:
            return preferred
        if fallback in allowed:
            return fallback
        return allowed[0] if allowed else fallback

    payload: dict[str, Any] = {
        "age": int(round(_clamp(numeric("age", 35), 18, 100))),
        "gender": category("gender", "Male"),
        "marital_status": category("marital_status", "Married"),
        "education": category("education", "Graduate"),
        "monthly_salary": _clamp(numeric("monthly_salary", 50_000), 1, 100_000_000),
        "employment_type": category("employment_type", "Private"),
        "years_of_employment": _clamp(
            numeric("years_of_employment", 5.0), 0, 60
        ),
        "company_type": category("company_type", "Large Indian"),
        "house_type": category("house_type", "Rented"),
        "monthly_rent": _clamp(numeric("monthly_rent", 10_000), 0, 10_000_000),
        "family_size": int(round(_clamp(numeric("family_size", 3), 1, 30))),
        "dependents": int(round(_clamp(numeric("dependents", 2), 0, 30))),
        "school_fees": _clamp(numeric("school_fees", 3_000), 0, 10_000_000),
        "college_fees": _clamp(numeric("college_fees", 0), 0, 10_000_000),
        "travel_expenses": _clamp(
            numeric("travel_expenses", 3_000), 0, 10_000_000
        ),
        "groceries_utilities": _clamp(
            numeric("groceries_utilities", 8_000), 0, 10_000_000
        ),
        "other_monthly_expenses": _clamp(
            numeric("other_monthly_expenses", 3_000), 0, 10_000_000
        ),
        "existing_loans": category("existing_loans", "No"),
        "current_emi_amount": _clamp(
            numeric("current_emi_amount", 0), 0, 10_000_000
        ),
        "credit_score": int(round(_clamp(numeric("credit_score", 700), 300, 900))),
        "bank_balance": _clamp(
            numeric("bank_balance", 100_000), 0, 1_000_000_000
        ),
        "emergency_fund": _clamp(
            numeric("emergency_fund", 50_000), 0, 1_000_000_000
        ),
        "emi_scenario": category("emi_scenario", "Home Appliances EMI"),
        "requested_amount": _clamp(
            numeric("requested_amount", 200_000), 1, 100_000_000
        ),
        "requested_tenure": int(
            round(_clamp(numeric("requested_tenure", 24), 1, 120))
        ),
    }

    try:
        return validate_application(payload, categories)
    except ValidationError:
        # Last-resort defaults, still schema-valid, if metadata is unusual.
        payload.update(
            {
                "age": 35,
                "gender": "Male" if "Male" in categories.get("gender", []) else categories["gender"][0],
                "marital_status": "Single"
                if "Single" in categories.get("marital_status", [])
                else categories["marital_status"][0],
                "education": "Graduate"
                if "Graduate" in categories.get("education", [])
                else categories["education"][0],
                "employment_type": "Private"
                if "Private" in categories.get("employment_type", [])
                else categories["employment_type"][0],
                "company_type": "MNC"
                if "MNC" in categories.get("company_type", [])
                else categories["company_type"][0],
                "house_type": "Rented"
                if "Rented" in categories.get("house_type", [])
                else categories["house_type"][0],
                "existing_loans": "No"
                if "No" in categories.get("existing_loans", [])
                else categories["existing_loans"][0],
                "emi_scenario": "Personal Loan EMI"
                if "Personal Loan EMI" in categories.get("emi_scenario", [])
                else categories["emi_scenario"][0],
                "requested_tenure": 24,
            }
        )
        return validate_application(payload, categories)
