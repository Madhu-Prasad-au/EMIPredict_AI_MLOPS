"""Shared Streamlit UI helpers."""

from __future__ import annotations

import logging
from typing import Any, Callable

import streamlit as st

from emi_app import config
from emi_app.artifacts import ArtifactLoadError, ModelArtifacts, get_artifacts
from emi_app.data import DataAssetError

logger = logging.getLogger("emi_predict_ai.ui")

_CSS = """
<style>
    .block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1280px; }
    .emi-hero {
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #0f766e 0%, #115e59 55%, #134e4a 100%);
        color: #ffffff;
        margin-bottom: 1.1rem;
    }
    .emi-hero h1 { color: #ffffff; margin: 0 0 .25rem 0; font-size: 2.1rem; }
    .emi-hero p { color: #d1fae5; margin: 0; font-size: 1.02rem; }
    .emi-muted { color: #64748b; font-size: .9rem; }
    .emi-pill {
        display: inline-block; padding: .18rem .6rem; border-radius: 999px;
        font-size: .78rem; font-weight: 600; letter-spacing: .01em;
    }
    .emi-pill-ok { background: #dcfce7; color: #166534; }
    .emi-pill-warn { background: #fef3c7; color: #92400e; }
    .emi-pill-bad { background: #fee2e2; color: #991b1b; }
    div[data-testid="stMetric"] { border: 1px solid #e2e8f0; border-radius: 12px; padding: .65rem .8rem; }
    footer { visibility: hidden; }
</style>
"""


def setup_page(title: str, icon: str = "💰", layout: str = "wide") -> None:
    st.set_page_config(
        page_title=f"{title} · {config.APP_NAME}",
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="emi-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f'<p class="emi-muted">{subtitle}</p>', unsafe_allow_html=True)


def get_artifacts_or_stop() -> ModelArtifacts:
    try:
        return get_artifacts()
    except ArtifactLoadError as exc:
        logger.exception("Artifact load failed")
        st.error("The model artifacts could not be loaded.")
        st.code(str(exc), language="text")
        st.info(
            "Check the `models/` directory and the artifact instructions in "
            "README.md. The regression artifact can be downloaded from the "
            "Hugging Face URL documented there."
        )
        if config.DEBUG:
            st.exception(exc)
        st.stop()


def get_data_or_stop(loader: Callable[[], Any], label: str) -> Any:
    try:
        return loader()
    except DataAssetError as exc:
        logger.exception("%s could not be loaded", label)
        st.error(f"{label} could not be loaded.")
        st.code(str(exc), language="text")
        st.info(
            "The shipped EDA/report assets may be missing. Run "
            "`python scripts/prepare_assets.py` or rebuild the Docker image."
        )
        if config.DEBUG:
            st.exception(exc)
        st.stop()


def sidebar(artifacts: ModelArtifacts | None = None) -> None:
    with st.sidebar:
        st.markdown("### 💰 EMIPredict AI")
        st.caption("EMI eligibility + affordability assessment")
        if artifacts is not None:
            st.success(
                f"**Models loaded once**\n\n"
                f"Classification: `{artifacts.classification_model_name}`  \n"
                f"Regression: `{artifacts.regression_model_name}`"
            )
            st.caption(
                f"{len(artifacts.feature_names)} features · "
                f"{len(artifacts.classification_classes)} classes · "
                f"loaded in {artifacts.load_seconds:.1f}s"
            )
        st.divider()
        st.caption(f"v{config.APP_VERSION} · {config.ENVIRONMENT}")


def format_currency(value: float | int | None, decimals: int = 0) -> str:
    if value is None:
        return "—"
    try:
        return f"₹{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def format_pct(value: float | int | None, decimals: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:,.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def format_number(value: float | int | None, decimals: int = 0) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def human_bytes(size: int | None) -> str:
    if not size:
        return "—"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:,.1f} GiB"


def status_pill(ok: bool, ok_text: str = "Healthy", bad_text: str = "Degraded") -> str:
    css = "emi-pill-ok" if ok else "emi-pill-bad"
    text = ok_text if ok else bad_text
    return f'<span class="emi-pill {css}">{text}</span>'


def footer() -> None:
    st.divider()
    st.caption(
        "EMIPredict AI is a machine-learning demonstration. Predictions are not "
        "credit approvals and must not be used as the sole basis for a lending "
        "decision. See the Model Performance page for limitations."
    )
