"""Health snapshot used by the Health page and startup logging."""

from __future__ import annotations

import time
from typing import Any

from emi_app import config
from emi_app.artifacts import ModelArtifacts
from emi_app.data import asset_status
from emi_app.service import run_smoke_test


def build_health_snapshot(artifacts: ModelArtifacts) -> dict[str, Any]:
    data_assets = asset_status()
    missing_assets = [item["label"] for item in data_assets if not item["exists"]]
    expected_features = 35 if artifacts.profile == "lite" else 36
    checks = {
        "artifacts_loaded": True,
        "classification_predict": callable(
            getattr(artifacts.classification_model, "predict", None)
        ),
        "classification_predict_proba": callable(
            getattr(artifacts.classification_model, "predict_proba", None)
        ),
        "regression_predict": callable(
            getattr(artifacts.regression_model, "predict", None)
        ),
        "feature_count_matches": len(artifacts.feature_names) == expected_features,
        "classes_match": tuple(artifacts.classification_classes)
        == ("Eligible", "High_Risk", "Not_Eligible"),
        "categorical_metadata_present": all(
            artifacts.categorical_values.get(feature)
            for feature in artifacts.categorical_features
        ),
        "eda_and_report_assets_present": not missing_assets,
    }
    healthy = all(checks.values())
    return {
        "status": "ok" if healthy else "degraded",
        "healthy": healthy,
        "profile": artifacts.profile,
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": config.ENVIRONMENT,
        "app_version": config.APP_VERSION,
        "health_endpoint": "/_stcore/health",
        "checks": checks,
        "missing_data_assets": missing_assets,
        "artifacts": artifacts.to_health_dict(),
        "data_assets": data_assets,
    }


def run_health_smoke_test(artifacts: ModelArtifacts) -> dict[str, Any]:
    """Run one real prediction and return a JSON-friendly result."""
    started = time.perf_counter()
    result = run_smoke_test(artifacts)
    return {
        "ok": True,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "result": result.to_dict(),
    }
