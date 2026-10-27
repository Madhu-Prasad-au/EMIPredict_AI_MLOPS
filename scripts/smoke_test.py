"""End-to-end artifact + prediction smoke test.

Run from the project root:
    python scripts/smoke_test.py

Exits non-zero on failure. This is safe to run inside the Docker image; it does
not need the cleaned training CSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emi_app.artifacts import get_artifacts  # noqa: E402
from emi_app.schema import (  # noqa: E402
    ValidationError,
    build_default_application,
    validate_application,
)
from emi_app.service import (  # noqa: E402
    DECISION_META,
    default_application,
    predict,
)


def expect_invalid(payload: dict, categories: dict, expected_fragment: str) -> None:
    try:
        validate_application(payload, categories)
    except ValidationError as exc:
        message = str(exc)
        if expected_fragment.casefold() not in message.casefold():
            raise AssertionError(
                f"Expected {expected_fragment!r} in validation error, got: {message}"
            )
        return
    raise AssertionError(f"Expected invalid input for {expected_fragment!r}")


def main() -> int:
    artifacts = get_artifacts()
    categories = artifacts.categorical_values
    application = default_application(artifacts)

    print("Artifacts")
    print(f"  classification : {artifacts.classification_model_name}")
    print(f"  regression     : {artifacts.regression_model_name}")
    print(f"  features       : {len(artifacts.feature_names)}")
    print(f"  classes        : {', '.join(artifacts.classification_classes)}")
    print(f"  load time      : {artifacts.load_seconds:.2f}s")

    result = predict(application, artifacts)
    assert result.eligibility in artifacts.classification_classes
    assert 0.0 <= result.eligibility_probability <= 1.0
    assert result.predicted_max_emi >= 0
    assert result.requested_emi > 0
    assert result.decision in DECISION_META
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6
    assert len(result.model_input) == len(artifacts.feature_names)

    result_again = predict(application, artifacts)
    assert result.eligibility == result_again.eligibility
    assert result.predicted_max_emi == result_again.predicted_max_emi

    invalid_payload = application.to_raw_dict()
    invalid_payload["monthly_salary"] = 0
    expect_invalid(invalid_payload, categories, "Monthly salary")

    invalid_payload = application.to_raw_dict()
    invalid_payload["credit_score"] = 200
    expect_invalid(invalid_payload, categories, "Credit score")

    invalid_payload = application.to_raw_dict()
    invalid_payload["company_type"] = "Medium"
    expect_invalid(invalid_payload, categories, "Company type")

    invalid_payload = application.to_raw_dict()
    invalid_payload.pop("requested_amount")
    expect_invalid(invalid_payload, categories, "Requested loan amount")

    invalid_payload = application.to_raw_dict()
    invalid_payload["monthly_rent"] = -1
    expect_invalid(invalid_payload, categories, "Monthly rent")

    invalid_payload = application.to_raw_dict()
    invalid_payload["requested_tenure"] = 0
    expect_invalid(invalid_payload, categories, "Requested tenure")

    # Extreme but schema-valid applicant: must not crash or return NaN.
    extreme_payload = application.to_raw_dict()
    extreme_payload.update(
        {
            "monthly_salary": 1.0,
            "monthly_rent": 10_000_000.0,
            "requested_amount": 100_000_000.0,
            "requested_tenure": 120,
            "emergency_fund": 0.0,
        }
    )
    extreme = validate_application(extreme_payload, categories)
    extreme_result = predict(extreme, artifacts)
    assert extreme_result.predicted_max_emi >= 0
    assert extreme_result.requested_emi > 0
    assert extreme_result.eligibility in artifacts.classification_classes

    # Default applicant returned by the stats metadata must validate.
    rebuilt = build_default_application(artifacts.stats, categories)
    assert rebuilt.to_raw_dict() == application.to_raw_dict()

    print("\nDefault prediction")
    print(json.dumps(
        {
            "eligibility": result.eligibility,
            "predicted_max_emi": result.predicted_max_emi,
            "requested_emi": result.requested_emi,
            "decision": result.decision_label,
            "affordability_ratio": result.affordability_ratio,
            "ratio_iterations": result.ratio_iterations,
            "ratio_converged": result.ratio_converged,
            "inference_ms": result.inference_ms,
            "warnings": result.warnings,
        },
        indent=2,
    ))
    print("\nValidation and extreme-input tests passed.")
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
