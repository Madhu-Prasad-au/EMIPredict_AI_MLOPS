"""Prediction service: reconstruct training features and run both models."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from emi_app import config
from emi_app.artifacts import ModelArtifacts
from emi_app.preprocessing import (
    MAX_AFFORDABILITY_RATIO,
    base_affordability_ratio,
    build_feature_frame,
    engineering_values,
)
from emi_app.schema import (
    ApplicationInput,
    application_warnings,
    build_default_application,
)

logger = logging.getLogger("emi_predict_ai.service")


class PredictionError(RuntimeError):
    """Raised when a loaded model cannot produce a valid prediction."""


DECISION_RECOMMENDED = "recommended"
DECISION_REVIEW = "review"
DECISION_MANUAL_REVIEW = "manual_review"
DECISION_NOT_ELIGIBLE = "not_eligible"

DECISION_META: dict[str, dict[str, str]] = {
    DECISION_RECOMMENDED: {
        "label": "Recommended",
        "tone": "success",
        "icon": "✅",
    },
    DECISION_REVIEW: {
        "label": "Review required",
        "tone": "warning",
        "icon": "⚠️",
    },
    DECISION_MANUAL_REVIEW: {
        "label": "Manual review",
        "tone": "warning",
        "icon": "🕵️",
    },
    DECISION_NOT_ELIGIBLE: {
        "label": "Not eligible",
        "tone": "error",
        "icon": "⛔",
    },
}


@dataclass(frozen=True)
class PredictionResult:
    eligibility: str
    eligibility_probability: float
    probabilities: dict[str, float]
    predicted_max_emi: float
    requested_emi: float
    affordability_gap: float
    is_affordable: bool
    affordability_ratio: float
    base_affordability_ratio: float
    ratio_iterations: int
    ratio_converged: bool
    uses_reconstructed_ratio: bool
    decision: str
    decision_label: str
    recommendation: str
    engineered_features: dict[str, float]
    raw_input: dict[str, Any]
    model_names: dict[str, str]
    model_input: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    inference_ms: float = 0.0

    @property
    def decision_tone(self) -> str:
        return DECISION_META.get(self.decision, {}).get("tone", "info")

    @property
    def decision_icon(self) -> str:
        return DECISION_META.get(self.decision, {}).get("icon", "•")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip_ratio(value: float) -> float:
    return float(min(max(value, 0.0), MAX_AFFORDABILITY_RATIO))


def _ratio_from_emi(emi: float, salary: float) -> float:
    if salary <= 0 or not math.isfinite(salary):
        raise PredictionError("Monthly salary must be greater than zero.")
    return _clip_ratio(max(emi, 0.0) / salary)


def _regression_prediction(
    artifacts: ModelArtifacts,
    frame,
) -> float:
    try:
        raw = artifacts.regression_model.predict(frame)
    except Exception as exc:  # noqa: BLE001 - convert to a user-safe error
        raise PredictionError(
            f"Regression model failed: {type(exc).__name__}: {exc}"
        ) from exc
    values = np.asarray(raw).reshape(-1)
    if values.size != 1:
        raise PredictionError(
            f"Regression model returned {values.size} predictions for one row."
        )
    value = float(values[0])
    if not math.isfinite(value):
        raise PredictionError("Regression model returned a non-finite value.")
    return max(value, 0.0)


def _classification_prediction(
    artifacts: ModelArtifacts,
    frame,
) -> tuple[str, dict[str, float]]:
    try:
        encoded = artifacts.classification_model.predict(frame)
        probabilities = np.asarray(
            artifacts.classification_model.predict_proba(frame)
        )
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(
            f"Classification model failed: {type(exc).__name__}: {exc}"
        ) from exc

    encoded_values = np.asarray(encoded).reshape(-1)
    if encoded_values.size != 1:
        raise PredictionError(
            f"Classification model returned {encoded_values.size} predictions."
        )
    try:
        eligibility = str(
            artifacts.label_encoder.inverse_transform(encoded_values)[0]
        )
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(
            f"Could not decode the predicted class: {type(exc).__name__}: {exc}"
        ) from exc

    if probabilities.ndim != 2 or probabilities.shape[0] != 1:
        raise PredictionError("Classification model returned invalid probabilities.")
    if probabilities.shape[1] != len(artifacts.classification_classes):
        raise PredictionError(
            "Classification probabilities do not match the label encoder classes."
        )

    model_classes = getattr(
        getattr(artifacts.classification_model, "named_steps", {}).get("model"),
        "classes_",
        None,
    )
    if model_classes is not None:
        class_names = [
            str(name)
            for name in artifacts.label_encoder.inverse_transform(model_classes)
        ]
    else:  # pragma: no cover - all supported artifacts expose classes_
        class_names = list(artifacts.classification_classes)

    if tuple(class_names) != tuple(artifacts.classification_classes):
        raise PredictionError(
            "Model probability column order does not match the label encoder."
        )

    probability_map = {
        name: float(probabilities[0, index])
        for index, name in enumerate(class_names)
    }
    return eligibility, probability_map


def _recommendation(
    *,
    eligibility: str,
    is_affordable: bool,
    predicted_max_emi: float,
    requested_emi: float,
) -> tuple[str, str]:
    """Return (decision key, human-readable recommendation)."""
    eligible = eligibility.strip().casefold() == "eligible"
    high_risk = eligibility.strip().casefold() in {"high_risk", "high risk"}
    not_eligible = eligibility.strip().casefold() in {
        "not_eligible",
        "not eligible",
        "ineligible",
    }

    if eligible and is_affordable:
        return (
            DECISION_RECOMMENDED,
            "The applicant is classified as eligible and the requested EMI is "
            "within the model's estimated affordability limit. This is a model "
            "assessment, not a credit approval.",
        )
    if eligible and not is_affordable:
        return (
            DECISION_REVIEW,
            f"The applicant is classified as eligible, but the requested EMI "
            f"(₹{requested_emi:,.2f}) exceeds the estimated affordable EMI "
            f"(₹{predicted_max_emi:,.2f}). Consider reducing the loan amount or "
            "extending the tenure, then re-run the assessment.",
        )
    if high_risk:
        return (
            DECISION_MANUAL_REVIEW,
            "The applicant is classified as high risk. A manual underwriting "
            "review is recommended before any decision.",
        )
    if not_eligible:
        return (
            DECISION_NOT_ELIGIBLE,
            f"The applicant is classified as not eligible. The model estimates a "
            f"maximum affordable EMI of ₹{predicted_max_emi:,.2f} against a "
            f"requested EMI of ₹{requested_emi:,.2f}. Consider a smaller amount, "
            "a longer tenure, or a co-applicant.",
        )
    return (
        DECISION_REVIEW,
        f"The model returned the class '{eligibility}'. A manual review is "
        "recommended before relying on this result.",
    )


def predict(
    application: ApplicationInput,
    artifacts: ModelArtifacts,
) -> PredictionResult:
    """Run classification + regression for one validated applicant.

    Full profile: the target-derived `affordability_ratio` is reconstructed with
    a fixed-point loop, because the saved full model expects it.

    Lite profile: the compact leakage-free model does not use that feature, so a
    single regression call is made.
    """
    started = time.perf_counter()
    salary = application.monthly_salary
    if salary <= 0 or not math.isfinite(salary):
        raise PredictionError("Monthly salary must be a positive finite number.")

    uses_reconstructed_ratio = "affordability_ratio" in artifacts.feature_names
    base_ratio = 0.0
    ratio = 0.0
    converged = True
    iterations = 0

    if uses_reconstructed_ratio:
        base_ratio = base_affordability_ratio(application)
        ratio = base_ratio
        converged = False
        for iteration in range(1, config.AFFORDABILITY_ITERATIONS + 1):
            iterations = iteration
            frame = build_feature_frame(
                application,
                artifacts.feature_names,
                ratio,
                artifacts.numeric_fill_values,
            )
            emi = _regression_prediction(artifacts, frame)
            new_ratio = _ratio_from_emi(emi, salary)
            if abs(new_ratio - ratio) <= config.AFFORDABILITY_TOLERANCE:
                ratio = new_ratio
                converged = True
                break
            # Direct update: the forest response is piecewise constant, so the
            # ratio either reaches a stable self-consistent value quickly or the
            # loop is reported as approximate.
            ratio = _clip_ratio(new_ratio)

    # Always report a prediction made with the final ratio.
    final_frame = build_feature_frame(
        application,
        artifacts.feature_names,
        ratio,
        artifacts.numeric_fill_values,
    )
    predicted_max_emi = _regression_prediction(artifacts, final_frame)
    eligibility, probabilities = _classification_prediction(
        artifacts, final_frame
    )

    requested_emi = application.requested_emi
    affordability_gap = predicted_max_emi - requested_emi
    is_affordable = affordability_gap >= 0
    decision, recommendation = _recommendation(
        eligibility=eligibility,
        is_affordable=is_affordable,
        predicted_max_emi=predicted_max_emi,
        requested_emi=requested_emi,
    )

    warnings_list = application_warnings(application, artifacts.stats)
    confidence = max(probabilities.values()) if probabilities else 0.0
    if confidence < 0.60:
        warnings_list.append(
            "The classification confidence is below 60%. Treat the eligibility "
            "result as uncertain and review the probability distribution."
        )
    if uses_reconstructed_ratio and not converged:
        warnings_list.append(
            "The reconstructed affordability ratio did not fully converge within "
            f"{config.AFFORDABILITY_ITERATIONS} iterations. The EMI estimate is "
            "approximate."
        )

    engineered = engineering_values(application, ratio)
    model_input = {
        name: final_frame.iloc[0][name] for name in artifacts.feature_names
    }
    model_input = {
        key: (
            float(value)
            if isinstance(value, (int, float, np.integer, np.floating))
            else value
        )
        for key, value in model_input.items()
    }

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "Prediction completed in %.1f ms: profile=%s class=%s max_emi=%.2f "
        "requested=%.2f ratio_iterations=%d converged=%s",
        elapsed_ms,
        artifacts.profile,
        eligibility,
        predicted_max_emi,
        requested_emi,
        iterations,
        converged,
    )

    return PredictionResult(
        eligibility=eligibility,
        eligibility_probability=float(probabilities.get(eligibility, 0.0)),
        probabilities=probabilities,
        predicted_max_emi=round(float(predicted_max_emi), 2),
        requested_emi=round(float(requested_emi), 2),
        affordability_gap=round(float(affordability_gap), 2),
        is_affordable=bool(is_affordable),
        affordability_ratio=round(float(ratio), 6),
        base_affordability_ratio=round(float(base_ratio), 6),
        ratio_iterations=iterations,
        ratio_converged=bool(converged),
        uses_reconstructed_ratio=uses_reconstructed_ratio,
        decision=decision,
        decision_label=DECISION_META[decision]["label"],
        recommendation=recommendation,
        engineered_features={
            key: (round(float(value), 6) if math.isfinite(float(value)) else None)
            for key, value in engineered.items()
        },
        raw_input=application.to_raw_dict(),
        model_names={
            "classification": artifacts.classification_model_name,
            "regression": artifacts.regression_model_name,
        },
        model_input=model_input,
        warnings=warnings_list,
        inference_ms=round(elapsed_ms, 2),
    )


def default_application(artifacts: ModelArtifacts) -> ApplicationInput:
    return build_default_application(artifacts.stats, artifacts.categorical_values)


def run_smoke_test(artifacts: ModelArtifacts) -> PredictionResult:
    """Predict a realistic default applicant; used by the health page/tests."""
    return predict(default_application(artifacts), artifacts)
