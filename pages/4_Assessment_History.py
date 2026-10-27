"""Session-only assessment history.

The previous project wrote CRUD records to a CSV on disk. Render's container
filesystem is ephemeral, so this page keeps history in Streamlit session state
instead. No applicant data is persisted server-side.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from emi_app.ui import (
    footer,
    format_currency,
    format_pct,
    get_artifacts_or_stop,
    page_header,
    setup_page,
    sidebar,
)

setup_page("Assessment History", icon="🗂️")
artifacts = get_artifacts_or_stop()
sidebar(artifacts)

page_header(
    "🗂️ Assessment history",
    "Assessments completed in this browser session. History is not written to "
    "disk and is lost when the session ends or the server restarts.",
)

history = st.session_state.get("assessment_history", [])

if not history:
    st.info(
        "No assessments yet. Run an EMI assessment first; results will appear "
        "here for this session."
    )
    st.info("Open **EMI Prediction** from the sidebar to run an assessment.")
    footer()
    st.stop()

rows = []
for index, entry in enumerate(history):
    application = entry.get("application", {})
    result = entry.get("result", {})
    rows.append(
        {
            "#": index + 1,
            "Timestamp": entry.get("timestamp", ""),
            "Eligibility": result.get("eligibility", ""),
            "Decision": result.get("decision_label", ""),
            "Max affordable EMI (₹)": result.get("predicted_max_emi"),
            "Requested EMI (₹)": result.get("requested_emi"),
            "Salary (₹)": application.get("monthly_salary"),
            "Credit score": application.get("credit_score"),
            "Scenario": application.get("emi_scenario"),
        }
    )
history_frame = pd.DataFrame(rows)

st.subheader("This session")
st.dataframe(history_frame, width="stretch", hide_index=True)

col1, col2 = st.columns([1, 1])
with col1:
    st.download_button(
        "⬇️ Download history (CSV)",
        data=history_frame.to_csv(index=False).encode("utf-8"),
        file_name="emi_assessment_history.csv",
        mime="text/csv",
        width="stretch",
    )
with col2:
    if st.button("🗑️ Clear this session's history", width="stretch"):
        st.session_state["assessment_history"] = []
        st.session_state.pop("last_prediction", None)
        st.rerun()

st.divider()
st.subheader("Assessment detail")

labels = [
    f"#{index + 1} · {entry.get('result', {}).get('decision_label', '')} · "
    f"{entry.get('timestamp', '')}"
    for index, entry in enumerate(history)
]
selected_label = st.selectbox("Select an assessment", options=labels, index=len(labels) - 1)
selected_index = labels.index(selected_label)
entry = history[selected_index]
result = entry.get("result", {})
application = entry.get("application", {})
raw_input = entry.get("raw_input", {})

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Eligibility", result.get("eligibility", "—"))
with m2:
    st.metric(
        "Maximum affordable EMI",
        format_currency(result.get("predicted_max_emi"), 2),
    )
with m3:
    st.metric("Requested EMI", format_currency(result.get("requested_emi"), 2))
with m4:
    st.metric("Decision", result.get("decision_label", "—"))

st.markdown(result.get("recommendation", ""))

probabilities = result.get("probabilities", {})
if probabilities:
    probability_frame = pd.DataFrame(
        {
            "Class": list(probabilities.keys()),
            "Probability": [
                format_pct(value) for value in probabilities.values()
            ],
        }
    )
    st.dataframe(probability_frame, width="stretch", hide_index=True)

if result.get("warnings"):
    st.warning("\n".join(f"- {warning}" for warning in result["warnings"]))

with st.expander("Applicant input"):
    st.dataframe(
        pd.DataFrame(
            [{"Field": key, "Value": str(value)} for key, value in raw_input.items()]
        ),
        width="stretch",
        hide_index=True,
    )

with st.expander("Raw result JSON"):
    st.code(json.dumps(result, indent=2, default=str), language="json")

footer()
