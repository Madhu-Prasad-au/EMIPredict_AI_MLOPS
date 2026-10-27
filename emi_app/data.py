"""Readers for the compact EDA assets and saved model-comparison reports.

The full 143 MB cleaned CSV is intentionally not shipped. `data/eda_sample.csv.gz`
is a 60k-row proportional stratified sample; `data/eda_summary.json` contains
full-dataset aggregates (404,800 records) used for headline metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pandas as pd

from emi_app import config


class DataAssetError(RuntimeError):
    """Raised when a shipped EDA/report asset is missing or malformed."""


@dataclass(frozen=True)
class ReportBundle:
    classification_comparison: pd.DataFrame
    regression_comparison: pd.DataFrame
    classification_final: pd.DataFrame
    regression_final: pd.DataFrame

    @property
    def classification_final_metrics(self) -> dict[str, float]:
        return _metrics_to_dict(self.classification_final)

    @property
    def regression_final_metrics(self) -> dict[str, float]:
        return _metrics_to_dict(self.regression_final)

    @property
    def best_classification_row(self) -> pd.Series:
        return self.classification_comparison.sort_values(
            "F1 Score", ascending=False
        ).iloc[0]

    @property
    def best_regression_row(self) -> pd.Series:
        return self.regression_comparison.sort_values("RMSE").iloc[0]


def _metrics_to_dict(frame: pd.DataFrame) -> dict[str, float]:
    if "Metric" not in frame.columns or "Value" not in frame.columns:
        raise DataAssetError(
            "Metric report must contain 'Metric' and 'Value' columns."
        )
    result: dict[str, float] = {}
    for _, row in frame.iterrows():
        try:
            result[str(row["Metric"])] = float(row["Value"])
        except (TypeError, ValueError):
            continue
    return result


@lru_cache(maxsize=1)
def load_summary() -> dict[str, Any]:
    path = config.EDA_SUMMARY_PATH
    if not path.is_file():
        raise DataAssetError(
            f"EDA summary not found: {path}. Run scripts/prepare_assets.py "
            "or rebuild the Docker image."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataAssetError(f"Could not read EDA summary: {exc}") from exc


@lru_cache(maxsize=1)
def load_eda_sample() -> pd.DataFrame:
    path = config.EDA_SAMPLE_PATH
    if not path.is_file():
        raise DataAssetError(
            f"EDA sample not found: {path}. Run scripts/prepare_assets.py "
            "or rebuild the Docker image."
        )
    try:
        frame = pd.read_csv(path, compression="gzip", low_memory=False)
    except Exception as exc:  # noqa: BLE001
        raise DataAssetError(f"Could not read EDA sample: {exc}") from exc
    required = {
        "monthly_salary",
        "credit_score",
        "max_monthly_emi",
        "emi_eligibility",
        "emi_scenario",
        "debt_to_income_ratio",
        "expense_to_income_ratio",
        "affordability_ratio",
        "requested_amount",
        "requested_tenure",
        "current_emi_amount",
        "bank_balance",
        "emergency_fund",
    }
    missing = required - set(frame.columns)
    if missing:
        raise DataAssetError(
            "EDA sample is missing columns: " + ", ".join(sorted(missing))
        )
    return frame


@lru_cache(maxsize=1)
def load_reports() -> ReportBundle:
    paths = config.REPORTS
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise DataAssetError(
            "Model performance reports are missing:\n  " + "\n  ".join(missing)
        )

    try:
        classification_comparison = pd.read_csv(paths["classification_comparison"])
        regression_comparison = pd.read_csv(paths["regression_comparison"])
        classification_final = pd.read_csv(paths["classification_final"])
        regression_final = pd.read_csv(paths["regression_final"])
    except Exception as exc:  # noqa: BLE001
        raise DataAssetError(f"Could not read model performance reports: {exc}") from exc

    required_classification = {"Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"}
    required_regression = {"Model", "RMSE", "MAE", "R2 Score", "MAPE (%)"}
    if not required_classification.issubset(classification_comparison.columns):
        raise DataAssetError(
            "classification_model_comparison.csv is missing expected columns."
        )
    if not required_regression.issubset(regression_comparison.columns):
        raise DataAssetError(
            "regression_model_comparison.csv is missing expected columns."
        )

    return ReportBundle(
        classification_comparison=classification_comparison,
        regression_comparison=regression_comparison,
        classification_final=classification_final,
        regression_final=regression_final,
    )


def asset_status() -> list[dict[str, Any]]:
    """Small status list for the health page."""
    items = [
        ("EDA summary", config.EDA_SUMMARY_PATH, True),
        ("EDA sample", config.EDA_SAMPLE_PATH, True),
        ("Classification comparison", config.REPORTS["classification_comparison"], True),
        ("Regression comparison", config.REPORTS["regression_comparison"], True),
        ("Final classification metrics", config.REPORTS["classification_final"], True),
        ("Final regression metrics", config.REPORTS["regression_final"], True),
    ]
    output = []
    for label, path, required in items:
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        output.append(
            {
                "label": label,
                "path": str(path),
                "required": required,
                "exists": exists,
                "size_mb": round(size / 1024 / 1024, 2) if size else None,
            }
        )
    return output
