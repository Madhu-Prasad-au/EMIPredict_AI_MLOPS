"""Load, validate, and expose the saved model artifacts exactly once.

This module is deliberately independent of Streamlit. `entrypoint.py` calls
`get_artifacts()` before the web server starts, and every page reuses the same
in-process singleton. Streamlit reruns do not re-import this module, so the
295 MB regression pipeline is never reloaded per request or per session.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from emi_app import config

logger = logging.getLogger("emi_predict_ai.artifacts")


class ArtifactLoadError(RuntimeError):
    """Raised when a required artifact cannot be found or validated."""


@dataclass(frozen=True)
class ArtifactFile:
    key: str
    label: str
    path: Path
    required: bool
    exists: bool
    size_bytes: int | None = None
    sha256: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "path": str(self.path),
            "required": self.required,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_bytes / 1024 / 1024, 2)
            if self.size_bytes
            else None,
            "sha256": self.sha256,
            "error": self.error,
        }


@dataclass
class ModelArtifacts:
    classification_model: Any
    regression_model: Any
    label_encoder: Any
    feature_info: dict[str, Any]
    stats: dict[str, Any]

    feature_names: tuple[str, ...]
    categorical_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_values: dict[str, list[str]]
    numeric_fill_values: dict[str, float]

    classification_model_name: str
    regression_model_name: str
    classification_classes: tuple[str, ...]

    artifact_files: tuple[ArtifactFile, ...]
    artifact_warnings: tuple[str, ...]
    load_seconds: float
    loaded_at_utc: str

    regression_path: Path
    classification_path: Path

    runtime_versions: dict[str, str] = field(default_factory=dict)

    def to_health_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "loaded_at_utc": self.loaded_at_utc,
            "load_seconds": round(self.load_seconds, 3),
            "classification_model": self.classification_model_name,
            "regression_model": self.regression_model_name,
            "classification_classes": list(self.classification_classes),
            "feature_count": len(self.feature_names),
            "categorical_features": list(self.categorical_features),
            "numeric_features": len(self.numeric_features),
            "categorical_values": self.categorical_values,
            "regression_artifact": str(self.regression_path),
            "runtime_versions": self.runtime_versions,
            "artifact_warnings": list(self.artifact_warnings),
            "files": [item.to_dict() for item in self.artifact_files],
        }


_LOCK = threading.Lock()
_ARTIFACTS: ModelArtifacts | None = None


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(
    key: str,
    label: str,
    path: Path,
    required: bool,
    *,
    hash_file: bool = True,
) -> ArtifactFile:
    exists = path.is_file()
    size = path.stat().st_size if exists else None
    digest = None
    error = None
    if exists and hash_file:
        try:
            digest = _sha256(path)
        except OSError as exc:  # pragma: no cover - unusual I/O failure
            error = f"Could not hash artifact: {exc}"
    elif not exists and required:
        error = "Missing required artifact"
    return ArtifactFile(
        key=key,
        label=label,
        path=path,
        required=required,
        exists=exists,
        size_bytes=size,
        sha256=digest,
        error=error,
    )


def _resolve_version(package: str, *fallbacks: str) -> str:
    for name in (package, *fallbacks):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "not installed"


def _load_joblib(path: Path, label: str) -> Any:
    if not path.is_file():
        raise ArtifactLoadError(f"{label} not found: {path}")
    if path.stat().st_size == 0:
        raise ArtifactLoadError(f"{label} is empty: {path}")
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - surface a clear artifact error
        raise ArtifactLoadError(
            f"Failed to load {label} from {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _get_step(pipeline: Any, name: str, label: str) -> Any:
    steps = getattr(pipeline, "named_steps", None)
    if steps is None or name not in steps:
        raise ArtifactLoadError(
            f"{label} does not contain the expected '{name}' step. "
            f"Found: {list(steps) if steps else 'none'}"
        )
    return steps[name]


def _extract_preprocessor_metadata(pipeline: Any, label: str) -> dict[str, Any]:
    pre = _get_step(pipeline, "preprocessor", label)
    transformers = getattr(pre, "transformers_", None)
    if not transformers:
        raise ArtifactLoadError(f"{label} preprocessor has no fitted transformers.")

    named = {name: (transformer, columns) for name, transformer, columns in transformers}
    if "num" in named:
        num_transformer, num_columns = named["num"]
    else:
        num_transformer, num_columns = next(
            (t, c) for _, t, c in transformers if "num" in _
        )
    if "cat" in named:
        cat_transformer, cat_columns = named["cat"]
    else:
        cat_transformer, cat_columns = next(
            (t, c) for _, t, c in transformers if "cat" in _
        )

    num_steps = getattr(num_transformer, "named_steps", {})
    cat_steps = getattr(cat_transformer, "named_steps", {})
    imputer = num_steps.get("imputer")
    onehot = cat_steps.get("onehot")
    if imputer is None:
        raise ArtifactLoadError(f"{label} numeric pipeline has no imputer.")
    if onehot is None:
        raise ArtifactLoadError(f"{label} categorical pipeline has no onehot encoder.")

    categories = getattr(onehot, "categories_", None)
    if categories is None:
        raise ArtifactLoadError(f"{label} one-hot encoder is not fitted.")
    if len(categories) != len(cat_columns):
        raise ArtifactLoadError(
            f"{label} one-hot categories do not match categorical columns."
        )

    statistics = getattr(imputer, "statistics_", None)
    if statistics is None:
        raise ArtifactLoadError(f"{label} numeric imputer is not fitted.")
    if len(statistics) != len(num_columns):
        raise ArtifactLoadError(
            f"{label} imputer statistics do not match numeric columns."
        )

    return {
        "numeric_columns": tuple(str(c) for c in num_columns),
        "categorical_columns": tuple(str(c) for c in cat_columns),
        "categories": {
            str(column): [str(value) for value in values]
            for column, values in zip(cat_columns, categories)
        },
        "fill_values": {
            str(column): float(value)
            for column, value in zip(num_columns, statistics)
            if value is not None
        },
        "n_output_features": getattr(pre, "n_features_in_", None),
    }


def _validate_feature_info(feature_info: Any) -> dict[str, Any]:
    if not isinstance(feature_info, dict):
        raise ArtifactLoadError("feature_info.pkl does not contain a dictionary.")
    required = {
        "classification_features",
        "regression_features",
        "categorical_features",
        "classification_classes",
        "best_classification_model",
        "best_regression_model",
    }
    missing = sorted(required - set(feature_info))
    if missing:
        raise ArtifactLoadError(
            "feature_info.pkl is missing keys: " + ", ".join(missing)
        )
    return feature_info


def _load_artifacts_impl() -> ModelArtifacts:
    start = time.perf_counter()

    classification_path = config.CLASSIFICATION_MODEL_PATH
    regression_path = config.REGRESSION_MODEL_PATH
    if not regression_path.is_file() and config.REGRESSION_MODEL_FALLBACK_PATH.is_file():
        regression_path = config.REGRESSION_MODEL_FALLBACK_PATH
        logger.warning(
            "Compressed regression artifact not found; using the uncompressed "
            "fallback at %s. Expect slower startup and higher memory use.",
            regression_path,
        )

    required_paths = {
        "classification_model": classification_path,
        "regression_model": regression_path,
        "label_encoder": config.LABEL_ENCODER_PATH,
        "feature_info": config.FEATURE_INFO_PATH,
    }
    missing = [f"{key}: {path}" for key, path in required_paths.items() if not path.is_file()]
    if missing:
        raise ArtifactLoadError(
            "Required model artifacts are missing:\n  "
            + "\n  ".join(missing)
            + "\n\nExpected model directory: "
            + str(config.MODELS_DIR)
            + "\nFor the regression artifact, see README.md (download URL and "
            "SHA-256 checksum)."
        )

    feature_info = _validate_feature_info(_load_joblib(config.FEATURE_INFO_PATH, "feature info"))
    label_encoder = _load_joblib(config.LABEL_ENCODER_PATH, "label encoder")

    captured: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classification_model = _load_joblib(classification_path, "classification model")
        regression_model = _load_joblib(regression_path, "regression model")
        for item in caught:
            message = str(item.message)
            if "version" in message.lower() or "unpickle" in message.lower():
                captured.append(message)

    # Single-threaded inference keeps memory predictable on small instances and
    # avoids joblib/multiprocessing surprises in restricted container runtimes.
    for pipeline, label in (
        (classification_model, "classification model"),
        (regression_model, "regression model"),
    ):
        model_step = getattr(pipeline, "named_steps", {}).get("model")
        if model_step is None:
            raise ArtifactLoadError(f"{label} has no 'model' step.")
        if hasattr(model_step, "n_jobs"):
            model_step.n_jobs = 1

    if not hasattr(classification_model, "predict"):
        raise ArtifactLoadError("Classification artifact has no predict().")
    if not hasattr(classification_model, "predict_proba"):
        raise ArtifactLoadError("Classification artifact has no predict_proba().")
    if not hasattr(regression_model, "predict"):
        raise ArtifactLoadError("Regression artifact has no predict().")

    clf_pre = _extract_preprocessor_metadata(classification_model, "classification model")
    reg_pre = _extract_preprocessor_metadata(regression_model, "regression model")

    feature_names = tuple(str(name) for name in feature_info["classification_features"])
    regression_features = tuple(
        str(name) for name in feature_info["regression_features"]
    )
    if feature_names != regression_features:
        raise ArtifactLoadError(
            "Classification and regression feature lists differ. This application "
            "expects the shared feature pipeline used by the training notebook."
        )

    expected_numeric = set(clf_pre["numeric_columns"])
    expected_categorical = set(clf_pre["categorical_columns"])
    missing_from_pipeline = [
        name
        for name in feature_names
        if name not in expected_numeric and name not in expected_categorical
    ]
    if missing_from_pipeline:
        raise ArtifactLoadError(
            "Model pipeline does not consume these declared features: "
            + ", ".join(missing_from_pipeline)
        )

    if set(feature_info["categorical_features"]) != expected_categorical:
        raise ArtifactLoadError(
            "feature_info categorical_features do not match the saved encoder."
        )

    if tuple(clf_pre["categorical_columns"]) != tuple(reg_pre["categorical_columns"]):
        raise ArtifactLoadError(
            "Classification and regression categorical columns differ."
        )
    if tuple(clf_pre["numeric_columns"]) != tuple(reg_pre["numeric_columns"]):
        raise ArtifactLoadError(
            "Classification and regression numeric columns differ."
        )

    for column, values in clf_pre["categories"].items():
        reg_values = reg_pre["categories"].get(column)
        if reg_values != values:
            raise ArtifactLoadError(
                f"Classification and regression encoder categories differ for '{column}'."
            )

    classes = tuple(str(name) for name in label_encoder.classes_)
    declared_classes = tuple(
        str(name) for name in feature_info["classification_classes"]
    )
    if classes != declared_classes:
        raise ArtifactLoadError(
            "Label encoder classes do not match feature_info.classification_classes."
        )

    model_classes = getattr(
        getattr(classification_model, "named_steps", {}).get("model"),
        "classes_",
        None,
    )
    if model_classes is None:
        raise ArtifactLoadError("Classification model has no classes_ attribute.")
    encoded_classes = tuple(
        str(name) for name in label_encoder.inverse_transform(model_classes)
    )
    if encoded_classes != classes:
        raise ArtifactLoadError(
            "Classification model class order does not match the label encoder."
        )

    pipeline_n_features = getattr(classification_model, "n_features_in_", None)
    if pipeline_n_features is not None and int(pipeline_n_features) != len(feature_names):
        raise ArtifactLoadError(
            f"Classification pipeline expects {pipeline_n_features} features, but "
            f"feature_info declares {len(feature_names)}."
        )

    stats: dict[str, Any] = {}
    if config.PREPROCESSING_STATS_PATH.is_file():
        try:
            stats = json.loads(
                config.PREPROCESSING_STATS_PATH.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactLoadError(
                f"preprocessing_stats.json is invalid: {exc}"
            ) from exc
    else:
        logger.warning("preprocessing_stats.json not found; using pipeline defaults.")
        captured.append(
            "preprocessing_stats.json is missing; defaults will fall back to "
            "pipeline metadata only."
        )

    artifact_files = (
        _artifact_record(
            "classification_model",
            "Classification model (sklearn Pipeline + XGBoost)",
            classification_path,
            True,
            hash_file=True,
        ),
        _artifact_record(
            "regression_model",
            "Regression model (sklearn Pipeline + RandomForest)",
            regression_path,
            True,
            hash_file=True,
        ),
        _artifact_record(
            "label_encoder",
            "Classification label encoder",
            config.LABEL_ENCODER_PATH,
            True,
            hash_file=True,
        ),
        _artifact_record(
            "feature_info",
            "Feature metadata",
            config.FEATURE_INFO_PATH,
            True,
            hash_file=True,
        ),
        _artifact_record(
            "preprocessing_stats",
            "Inference preprocessing statistics",
            config.PREPROCESSING_STATS_PATH,
            False,
            hash_file=True,
        ),
    )

    runtime_versions = {
        "python": ".".join(str(part) for part in __import__("sys").version_info[:3]),
        "scikit-learn": _resolve_version("scikit-learn"),
        "xgboost": _resolve_version("xgboost", "xgboost-cpu"),
        "pandas": _resolve_version("pandas"),
        "numpy": _resolve_version("numpy"),
        "joblib": _resolve_version("joblib"),
        "streamlit": _resolve_version("streamlit"),
    }

    load_seconds = time.perf_counter() - start
    logger.info(
        "Loaded artifacts in %.2fs (%s + %s); regression=%.1f MB",
        load_seconds,
        feature_info.get("best_classification_model"),
        feature_info.get("best_regression_model"),
        regression_path.stat().st_size / 1024 / 1024,
    )

    return ModelArtifacts(
        classification_model=classification_model,
        regression_model=regression_model,
        label_encoder=label_encoder,
        feature_info=feature_info,
        stats=stats,
        feature_names=feature_names,
        categorical_features=tuple(clf_pre["categorical_columns"]),
        numeric_features=tuple(clf_pre["numeric_columns"]),
        categorical_values=clf_pre["categories"],
        numeric_fill_values=clf_pre["fill_values"],
        classification_model_name=str(feature_info["best_classification_model"]),
        regression_model_name=str(feature_info["best_regression_model"]),
        classification_classes=classes,
        artifact_files=artifact_files,
        artifact_warnings=tuple(captured),
        load_seconds=load_seconds,
        loaded_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        regression_path=regression_path,
        classification_path=classification_path,
        runtime_versions=runtime_versions,
    )


def get_artifacts() -> ModelArtifacts:
    """Return the process-wide artifacts, loading them on first use."""
    global _ARTIFACTS
    if _ARTIFACTS is not None:
        return _ARTIFACTS
    with _LOCK:
        if _ARTIFACTS is None:
            _ARTIFACTS = _load_artifacts_impl()
    return _ARTIFACTS


def load_artifacts_for_testing(force: bool = True) -> ModelArtifacts:
    """Test helper that forces a reload."""
    global _ARTIFACTS
    with _LOCK:
        if force or _ARTIFACTS is None:
            _ARTIFACTS = _load_artifacts_impl()
    return _ARTIFACTS


def artifacts_loaded() -> bool:
    return _ARTIFACTS is not None
