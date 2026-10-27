"""Health and readiness page.

Render's platform health check should use the built-in Streamlit endpoint
`/_stcore/health`. This page adds an application-level view: artifact hashes,
runtime versions, metadata checks, and an on-demand real prediction.
"""

from __future__ import annotations

import json
import logging

import pandas as pd
import streamlit as st

from emi_app import config
from emi_app.health import build_health_snapshot, run_health_smoke_test
from emi_app.ui import (
    footer,
    format_currency,
    get_artifacts_or_stop,
    human_bytes,
    page_header,
    setup_page,
    sidebar,
    status_pill,
)

logger = logging.getLogger("emi_predict_ai.health")

setup_page("System Health", icon="🩺")
artifacts = get_artifacts_or_stop()
sidebar(artifacts)

snapshot = build_health_snapshot(artifacts)

page_header(
    "🩺 System health",
    "Readiness of the loaded artifacts, runtime versions, shipped data assets, "
    "and an on-demand prediction smoke test.",
)

healthy = bool(snapshot["healthy"])
st.markdown(
    status_pill(
        healthy,
        "All checks passed",
        "Degraded — review the checks below",
    ),
    unsafe_allow_html=True,
)
st.write("")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Status", snapshot["status"].upper())
with c2:
    st.metric("Artifact load", f"{artifacts.load_seconds:.2f} s")
with c3:
    st.metric("Features", len(artifacts.feature_names))
with c4:
    st.metric(
        "Regression artifact",
        human_bytes(
            next(
                (
                    item["size_bytes"]
                    for item in snapshot["artifacts"]["files"]
                    if item["key"] == "regression_model"
                ),
                None,
            )
        ),
    )

st.info(
    f"Platform health endpoint: `{snapshot['health_endpoint']}`. "
    "It reports whether the Streamlit server is accepting connections. "
    "The smoke test below additionally exercises the saved models."
)

st.subheader("Readiness checks")
checks_frame = pd.DataFrame(
    [
        {"Check": name.replace("_", " "), "Passed": bool(value)}
        for name, value in snapshot["checks"].items()
    ]
)
st.dataframe(checks_frame, width="stretch", hide_index=True)

if snapshot["missing_data_assets"]:
    st.warning(
        "Missing data/report assets: " + ", ".join(snapshot["missing_data_assets"])
    )

st.subheader("Artifacts")
files_frame = pd.DataFrame(
    [
        {
            "Artifact": item["label"],
            "Required": item["required"],
            "Exists": item["exists"],
            "Size": human_bytes(item["size_bytes"]),
            "SHA-256 (prefix)": (item["sha256"][:16] + "…") if item["sha256"] else "—",
            "Path": item["path"],
        }
        for item in snapshot["artifacts"]["files"]
    ]
)
st.dataframe(files_frame, width="stretch", hide_index=True)

with st.expander("Full artifact hashes and model metadata"):
    st.json(snapshot["artifacts"], expanded=False)

st.subheader("Runtime")
version_frame = pd.DataFrame(
    [
        {"Package": name, "Version": version}
        for name, version in snapshot["artifacts"]["runtime_versions"].items()
    ]
)
st.dataframe(version_frame, width="stretch", hide_index=True)
st.caption(
    f"Environment: `{snapshot['environment']}` · App version: "
    f"`{snapshot['app_version']}` · Checked at `{snapshot['checked_at_utc']}`."
)

st.subheader("Shipped data and report assets")
st.dataframe(
    pd.DataFrame(
        [
            {
                "Asset": item["label"],
                "Exists": item["exists"],
                "Size": human_bytes(
                    int(item["size_mb"] * 1024 * 1024)
                    if item["size_mb"] is not None
                    else None
                ),
                "Path": item["path"],
            }
            for item in snapshot["data_assets"]
        ]
    ),
    width="stretch",
    hide_index=True,
)

if artifacts.artifact_warnings:
    st.warning(
        "Artifact warnings:\n\n"
        + "\n".join(f"- {warning}" for warning in artifacts.artifact_warnings)
    )

st.divider()
st.subheader("Prediction smoke test")
st.write(
    "Run one real prediction through the loaded classification and regression "
    "pipelines using a median applicant. This verifies inference, not just "
    "artifact loading."
)

if st.button("▶️ Run prediction smoke test", type="primary", width="stretch"):
    try:
        with st.spinner("Running the saved models..."):
            smoke = run_health_smoke_test(artifacts)
    except Exception as exc:  # noqa: BLE001 - surface to operator
        logger.exception("Health smoke test failed")
        st.error(f"Smoke test failed: {type(exc).__name__}: {exc}")
        if config.DEBUG:
            st.exception(exc)
    else:
        result = smoke["result"]
        st.success(
            f"Smoke test passed in {smoke['elapsed_ms']:.0f} ms · "
            f"eligibility **{result['eligibility']}** · "
            f"max EMI **{format_currency(result['predicted_max_emi'], 2)}**"
        )
        with st.expander("Smoke test result"):
            st.json(result, expanded=False)

st.download_button(
    "⬇️ Download health snapshot (JSON)",
    data=json.dumps(snapshot, indent=2, default=str),
    file_name="emi_health.json",
    mime="application/json",
    width="stretch",
)

st.caption(
    "This page is diagnostic. The Render health check should remain on "
    "`/_stcore/health` so deploys are validated quickly without loading the "
    "regression model again."
)

footer()
