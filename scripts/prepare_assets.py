"""Build compact, deployment-friendly EDA assets from the cleaned training data.

The full cleaned dataset is excluded from git/Docker. This script writes:
  - data/eda_sample.csv.gz   stratified sample used by the interactive EDA page
  - data/eda_summary.json    full-dataset aggregates used for headline metrics
  - models/preprocessing_stats.json  inference-time medians / observed ranges

Run from the project root:
    python scripts/prepare_assets.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "emi_prediction_cleaned.csv"
SAMPLE_OUT = ROOT / "data" / "eda_sample.csv.gz"
SAMPLE_LITE_OUT = ROOT / "data" / "eda_sample_lite.csv.gz"
SUMMARY_OUT = ROOT / "data" / "eda_summary.json"
STATS_OUT = ROOT / "models" / "preprocessing_stats.json"

SAMPLE_SIZE = 60_000
SAMPLE_SIZE_LITE = 20_000
RANDOM_STATE = 42

CATEGORICAL_COLUMNS = [
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "company_type",
    "house_type",
    "existing_loans",
    "emi_scenario",
    "emi_eligibility",
]

NUMERIC_COLUMNS = [
    "age",
    "monthly_salary",
    "years_of_employment",
    "monthly_rent",
    "family_size",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "requested_amount",
    "requested_tenure",
    "max_monthly_emi",
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
]

ENGINEERED_COLUMNS = [
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
]

CORRELATION_COLUMNS = [
    "monthly_salary",
    "current_emi_amount",
    "bank_balance",
    "emergency_fund",
    "requested_amount",
    "requested_tenure",
    "debt_to_income_ratio",
    "expense_to_income_ratio",
    "affordability_ratio",
    "credit_score",
    "max_monthly_emi",
]


def _round(value: float, digits: int = 6) -> float:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), digits)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Cleaned dataset not found: {SOURCE}")

    print(f"Reading {SOURCE} ...")
    df = pd.read_csv(SOURCE, low_memory=False)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    missing = [c for c in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Cleaned dataset is missing expected columns: {missing}")

    # ------------------------------------------------------------
    # Full-data summary
    # ------------------------------------------------------------
    numeric_summary = {}
    for col in NUMERIC_COLUMNS:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            numeric_summary[col] = {}
            continue
        numeric_summary[col] = {
            "min": _round(series.min()),
            "p01": _round(series.quantile(0.01)),
            "p05": _round(series.quantile(0.05)),
            "median": _round(series.median()),
            "mean": _round(series.mean()),
            "p95": _round(series.quantile(0.95)),
            "p99": _round(series.quantile(0.99)),
            "max": _round(series.max()),
        }

    categorical_summary = {}
    for col in CATEGORICAL_COLUMNS:
        counts = df[col].astype(str).value_counts(dropna=False)
        categorical_summary[col] = {
            str(k): int(v) for k, v in counts.items()
        }

    corr = (
        df[CORRELATION_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .corr()
        .round(4)
    )
    correlation = {
        row: {col: _round(corr.loc[row, col], 4) for col in corr.columns}
        for row in corr.index
    }

    class_counts = {
        str(k): int(v)
        for k, v in df["emi_eligibility"].astype(str).value_counts().items()
    }
    scenario_counts = {
        str(k): int(v)
        for k, v in df["emi_scenario"].astype(str).value_counts().items()
    }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "data/emi_prediction_cleaned.csv",
        "records": int(len(df)),
        "columns": int(df.shape[1]),
        "class_counts": class_counts,
        "scenario_counts": scenario_counts,
        "numeric": numeric_summary,
        "categorical": categorical_summary,
        "correlation": correlation,
        "sample": {
            "method": "proportional stratified sample by emi_eligibility + emi_scenario",
            "rows": SAMPLE_SIZE,
            "random_state": RANDOM_STATE,
        },
    }
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {SUMMARY_OUT} ({SUMMARY_OUT.stat().st_size / 1024:.1f} KiB)")

    # ------------------------------------------------------------
    # Preprocessing metadata for inference
    # ------------------------------------------------------------
    stats = {
        "generated_at_utc": summary["generated_at_utc"],
        "source": "data/emi_prediction_cleaned.csv",
        "records": int(len(df)),
        "engineered_medians": {
            col: _round(pd.to_numeric(df[col], errors="coerce").median())
            for col in ENGINEERED_COLUMNS
        },
        "numeric_medians": {
            col: _round(pd.to_numeric(df[col], errors="coerce").median())
            for col in NUMERIC_COLUMNS
        },
        "categorical_modes": {
            col: str(df[col].astype(str).mode().iloc[0])
            for col in CATEGORICAL_COLUMNS
        },
        "observed_ranges": {
            col: {
                "min": _round(pd.to_numeric(df[col], errors="coerce").min()),
                "max": _round(pd.to_numeric(df[col], errors="coerce").max()),
                "p01": _round(pd.to_numeric(df[col], errors="coerce").quantile(0.01)),
                "p99": _round(pd.to_numeric(df[col], errors="coerce").quantile(0.99)),
            }
            for col in NUMERIC_COLUMNS
        },
        "categorical_values": {
            col: sorted(df[col].astype(str).dropna().unique().tolist())
            for col in CATEGORICAL_COLUMNS
        },
        "notes": {
            "affordability_ratio": (
                "During training this feature was computed as "
                "max_monthly_emi / monthly_salary, i.e. it contains the regression "
                "target. At inference it is reconstructed iteratively from the "
                "regression model's own prediction divided by salary. The stored "
                "median below is for reference only and is not used as a fill value."
            )
        },
    }
    STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
    STATS_OUT.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {STATS_OUT} ({STATS_OUT.stat().st_size / 1024:.1f} KiB)")

    # ------------------------------------------------------------
    # Stratified samples
    # ------------------------------------------------------------
    _write_stratified_sample(df, SAMPLE_SIZE, SAMPLE_OUT, "full")
    _write_stratified_sample(df, SAMPLE_SIZE_LITE, SAMPLE_LITE_OUT, "lite")

    print("Done.")


def _write_stratified_sample(
    df: pd.DataFrame, size: int, output: Path, label: str
) -> None:
    print(f"Building {label} stratified sample ({size:,} rows) ...")
    frac = min(1.0, size / len(df))
    parts = []
    rng = np.random.default_rng(RANDOM_STATE)
    for _, group in df.groupby(["emi_eligibility", "emi_scenario"], sort=False):
        n = max(1, int(round(len(group) * frac)))
        n = min(n, len(group))
        if n == len(group):
            parts.append(group)
        else:
            idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
            parts.append(group.loc[idx])
    sample = pd.concat(parts).sample(frac=1.0, random_state=RANDOM_STATE)
    sample = sample.reset_index(drop=True)
    sample.to_csv(output, index=False, compression="gzip")
    print(
        f"Wrote {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB, "
        f"{len(sample):,} rows)"
    )
    print("Class balance in sample:")
    print(sample["emi_eligibility"].value_counts().to_string())


if __name__ == "__main__":
    main()
