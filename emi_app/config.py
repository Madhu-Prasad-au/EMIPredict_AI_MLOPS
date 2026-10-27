"""Paths and runtime configuration for EMIPredict AI.

All paths are anchored to the repository/application root (the parent of this
package), never to the process working directory. That makes the app behave the
same locally and inside the Docker image, where the working directory is /app.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "EMIPredict AI"
APP_SLUG = "emi-predict-ai"
APP_VERSION = "1.0.0"

# "full" = the original saved pipelines (large, leaks affordability_ratio).
# "lite" = compact, leakage-free HistGradientBoosting pair for 512 MB free
# instances. Select with EMI_PROFILE=lite.
PROFILE = os.environ.get("EMI_PROFILE", "full").strip().lower()
VALID_PROFILES = frozenset({"full", "lite"})

ROOT_DIR = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (ROOT_DIR / candidate).resolve()
    return candidate


MODELS_DIR = _path_from_env("EMI_MODELS_DIR", ROOT_DIR / "models")
DATA_DIR = _path_from_env("EMI_DATA_DIR", ROOT_DIR / "data")
REPORTS_DIR = _path_from_env("EMI_REPORTS_DIR", ROOT_DIR / "reports")

CLASSIFICATION_MODEL_PATH = MODELS_DIR / "best_classification_model.pkl"
REGRESSION_MODEL_PATH = MODELS_DIR / "best_regression_model_compressed.pkl"
REGRESSION_MODEL_FALLBACK_PATH = MODELS_DIR / "best_regression_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "classification_label_encoder.pkl"
FEATURE_INFO_PATH = MODELS_DIR / "feature_info.pkl"
PREPROCESSING_STATS_PATH = MODELS_DIR / "preprocessing_stats.json"

# Compact "free" profile artifacts.
LITE_CLASSIFICATION_MODEL_PATH = MODELS_DIR / "lite_classification_model.pkl"
LITE_REGRESSION_MODEL_PATH = MODELS_DIR / "lite_regression_model.pkl"
LITE_FEATURE_INFO_PATH = MODELS_DIR / "lite_feature_info.json"
LITE_METRICS_PATH = MODELS_DIR / "lite_metrics.json"

EDA_SAMPLE_PATH = DATA_DIR / "eda_sample.csv.gz"
EDA_SAMPLE_LITE_PATH = DATA_DIR / "eda_sample_lite.csv.gz"
EDA_SUMMARY_PATH = DATA_DIR / "eda_summary.json"

REPORTS = {
    "classification_comparison": REPORTS_DIR / "classification_model_comparison.csv",
    "regression_comparison": REPORTS_DIR / "regression_model_comparison.csv",
    "classification_final": REPORTS_DIR / "final_classification_metrics.csv",
    "regression_final": REPORTS_DIR / "final_regression_metrics.csv",
}

LOG_LEVEL = os.environ.get("EMI_LOG_LEVEL", "INFO").upper()
ENVIRONMENT = os.environ.get("EMI_ENV", "development").lower()
DEBUG = os.environ.get("EMI_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}

# Model artifacts are large. Keep them out of git but inside the Docker image.
REGRESSION_MODEL_URL = os.environ.get(
    "EMI_REGRESSION_MODEL_URL",
    (
        "https://huggingface.co/M-Prasad/EMIPredict_AI/resolve/main/"
        "best_regression_model_compressed.pkl?download=true"
    ),
)
REGRESSION_MODEL_SHA256 = os.environ.get(
    "EMI_REGRESSION_MODEL_SHA256",
    "a321224177d43c5e7de64c8bc98fa0f15689240ba7f521c29139ca412cd9e401",
)

# Number of regression iterations used to reconstruct the training-time
# affordability_ratio feature. See emi_app/preprocessing.py for details.
AFFORDABILITY_ITERATIONS = 10
AFFORDABILITY_TOLERANCE = 1e-4
