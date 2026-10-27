"""Train the compact, leakage-free "lite" model pair for Render Free.

The full artifacts are retained and remain the default profile. This script
creates separate, small HistGradientBoosting pipelines that fit comfortably in a
512 MB instance:

  models/lite_classification_model.pkl
  models/lite_regression_model.pkl
  models/lite_feature_info.json
  models/lite_metrics.json

The target-derived `affordability_ratio` feature is deliberately excluded.
This is a new model, not a modification of the saved production artifacts.

Run from the project root (requires the full cleaned CSV):
    python scripts/train_lite_models.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "emi_prediction_cleaned.csv"
LABEL_ENCODER_PATH = ROOT / "models" / "classification_label_encoder.pkl"
CLASSIFICATION_PATH = ROOT / "models" / "lite_classification_model.pkl"
REGRESSION_PATH = ROOT / "models" / "lite_regression_model.pkl"
FEATURE_INFO_PATH = ROOT / "models" / "lite_feature_info.json"
METRICS_PATH = ROOT / "models" / "lite_metrics.json"

RANDOM_STATE = 42
TARGET_CLASSIFICATION = "emi_eligibility"
TARGET_REGRESSION = "max_monthly_emi"
LEAKY_FEATURE = "affordability_ratio"

CATEGORICAL_FEATURES = [
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "company_type",
    "house_type",
    "existing_loans",
    "emi_scenario",
]


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def main() -> int:
    if not DATA_PATH.is_file():
        raise SystemExit(
            f"Full cleaned dataset not found: {DATA_PATH}\n"
            "This training script needs data/emi_prediction_cleaned.csv."
        )
    if not LABEL_ENCODER_PATH.is_file():
        raise SystemExit(f"Label encoder not found: {LABEL_ENCODER_PATH}")

    started = time.time()
    print(f"Reading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    full_features = [
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
        LEAKY_FEATURE,
    ]
    lite_features = [feature for feature in full_features if feature != LEAKY_FEATURE]
    numeric_features = [f for f in lite_features if f not in CATEGORICAL_FEATURES]
    categorical_features = [f for f in CATEGORICAL_FEATURES if f in lite_features]

    missing = [f for f in lite_features + [TARGET_CLASSIFICATION, TARGET_REGRESSION] if f not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {missing}")

    # Align the training credit-score range with the app's validation range.
    df["credit_score"] = pd.to_numeric(df["credit_score"], errors="coerce").clip(300, 900)

    X = df[lite_features].copy()
    y_class_raw = df[TARGET_CLASSIFICATION].astype(str)
    y_reg = pd.to_numeric(df[TARGET_REGRESSION], errors="coerce")

    valid = y_reg.notna()
    X = X.loc[valid].reset_index(drop=True)
    y_class_raw = y_class_raw.loc[valid].reset_index(drop=True)
    y_reg = y_reg.loc[valid].reset_index(drop=True)

    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    y_class = label_encoder.transform(y_class_raw)

    print(f"Lite features: {len(lite_features)} "
          f"({len(numeric_features)} numeric + {len(categorical_features)} categorical)")
    print(f"Excluded feature: {LEAKY_FEATURE}")

    X_train, X_temp, yc_train, yc_temp = train_test_split(
        X,
        y_class,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y_class,
    )
    X_val, X_test, yc_val, yc_test = train_test_split(
        X_temp,
        yc_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=yc_temp,
    )

    Xr_train, Xr_temp, yr_train, yr_temp = train_test_split(
        X,
        y_reg,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )
    Xr_val, Xr_test, yr_val, yr_test = train_test_split(
        Xr_temp,
        yr_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    print("Training compact HistGradientBoostingClassifier ...")
    classifier = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    classifier.fit(X_train, yc_train)

    print("Training compact HistGradientBoostingRegressor ...")
    regressor = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    regressor.fit(Xr_train, yr_train)

    def classification_metrics(X_eval, y_eval):
        pred = classifier.predict(X_eval)
        proba = classifier.predict_proba(X_eval)
        return {
            "Accuracy": _round(accuracy_score(y_eval, pred)),
            "Precision": _round(
                precision_score(y_eval, pred, average="weighted", zero_division=0)
            ),
            "Recall": _round(
                recall_score(y_eval, pred, average="weighted", zero_division=0)
            ),
            "F1 Score": _round(
                f1_score(y_eval, pred, average="weighted", zero_division=0)
            ),
            "ROC-AUC": _round(
                roc_auc_score(
                    y_eval, proba, multi_class="ovr", average="weighted"
                )
            ),
        }

    def regression_metrics(X_eval, y_eval):
        pred = regressor.predict(X_eval)
        return {
            "RMSE": _round(root_mean_squared_error(y_eval, pred), 4),
            "MAE": _round(mean_absolute_error(y_eval, pred), 4),
            "R2 Score": _round(r2_score(y_eval, pred)),
            "MAPE (%)": _round(mean_absolute_percentage_error(y_eval, pred) * 100, 4),
        }

    val_classification = classification_metrics(X_val, yc_val)
    test_classification = classification_metrics(X_test, yc_test)
    val_regression = regression_metrics(Xr_val, yr_val)
    test_regression = regression_metrics(Xr_test, yr_test)

    print("\nValidation classification:", val_classification)
    print("Test classification:      ", test_classification)
    print("Validation regression:    ", val_regression)
    print("Test regression:          ", test_regression)

    joblib.dump(classifier, CLASSIFICATION_PATH, compress=3)
    joblib.dump(regressor, REGRESSION_PATH, compress=3)

    feature_info = {
        "profile": "lite",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "feature_count": len(lite_features),
        "classification_features": lite_features,
        "regression_features": lite_features,
        "categorical_features": categorical_features,
        "numerical_features": numeric_features,
        "classification_classes": [str(c) for c in label_encoder.classes_],
        "classification_target": TARGET_CLASSIFICATION,
        "regression_target": TARGET_REGRESSION,
        "best_classification_model": "HistGradientBoostingClassifier (lite)",
        "best_regression_model": "HistGradientBoostingRegressor (lite)",
        "excluded_features": [LEAKY_FEATURE],
        "notes": (
            "Compact, leakage-free profile for small instances. "
            "affordability_ratio (target-derived) is excluded. This is a new "
            "model pair and does not replace the saved full artifacts."
        ),
    }
    metrics = {
        "profile": "lite",
        "generated_at_utc": feature_info["generated_at_utc"],
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "feature_count": len(lite_features),
        "model_names": {
            "classification": feature_info["best_classification_model"],
            "regression": feature_info["best_regression_model"],
        },
        "validation": {
            "classification": val_classification,
            "regression": val_regression,
        },
        "classification": test_classification,
        "regression": test_regression,
        "notes": feature_info["notes"],
    }

    FEATURE_INFO_PATH.write_text(json.dumps(feature_info, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nSaved:")
    for path in (CLASSIFICATION_PATH, REGRESSION_PATH, FEATURE_INFO_PATH, METRICS_PATH):
        print(f"  {path}  ({path.stat().st_size / 1024 / 1024:.2f} MiB)")
    print(f"Total elapsed: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
