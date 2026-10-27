"""Exploratory data analysis page over the shipped stratified sample."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from emi_app.data import load_eda_sample, load_summary
from emi_app.ui import (
    footer,
    format_currency,
    format_number,
    format_pct,
    get_artifacts_or_stop,
    get_data_or_stop,
    page_header,
    setup_page,
    sidebar,
)

setup_page("EDA Analytics", icon="📊")
artifacts = get_artifacts_or_stop()
sidebar(artifacts)
summary = get_data_or_stop(load_summary, "EDA summary")
sample = get_data_or_stop(load_eda_sample, "EDA sample")

page_header(
    "📊 EDA & financial analytics",
    "Interactive exploration of the cleaned EMI dataset. Charts use a "
    "proportional stratified sample; headline cards use full-dataset aggregates.",
)

full_records = int(summary.get("records", 0))
numeric_summary = summary.get("numeric", {})
sample_meta = summary.get("sample", {})

st.caption(
    f"Sample: {len(sample):,} of {full_records:,} records · "
    f"{sample_meta.get('method', 'stratified sample')}. "
    "The full 143 MB cleaned CSV is not shipped in the image."
)

# ---------------------------------------------------------------------------
# Headline metrics (full dataset)
# ---------------------------------------------------------------------------
st.subheader("Full-dataset overview")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Records", f"{full_records:,}")
with c2:
    st.metric(
        "Average salary",
        format_currency(numeric_summary.get("monthly_salary", {}).get("mean")),
    )
with c3:
    st.metric(
        "Median credit score",
        format_number(numeric_summary.get("credit_score", {}).get("median")),
    )
with c4:
    st.metric(
        "Average max EMI",
        format_currency(numeric_summary.get("max_monthly_emi", {}).get("mean")),
    )
with c5:
    eligible = summary.get("class_counts", {}).get("Eligible", 0)
    st.metric("Eligible share", format_pct(eligible / full_records if full_records else None))

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
st.sidebar.header("📌 Filters")
scenario_options = sorted(sample["emi_scenario"].dropna().unique().tolist())
eligibility_options = sorted(sample["emi_eligibility"].dropna().unique().tolist())

selected_scenarios = st.sidebar.multiselect(
    "EMI scenario",
    options=scenario_options,
    default=scenario_options,
)
selected_eligibility = st.sidebar.multiselect(
    "Eligibility",
    options=eligibility_options,
    default=eligibility_options,
)
credit_range = st.sidebar.slider(
    "Credit score range",
    min_value=300,
    max_value=900,
    value=(300, 900),
    step=10,
    help="The standard credit-score range. A few out-of-range values in the "
    "cleaned data are excluded from this view.",
)
st.sidebar.caption(f"Showing up to {len(sample):,} sampled records.")

filtered = sample[
    sample["emi_scenario"].isin(selected_scenarios)
    & sample["emi_eligibility"].isin(selected_eligibility)
    & sample["credit_score"].between(credit_range[0], credit_range[1])
].copy()

if filtered.empty:
    st.warning("No records match the selected filters. Widen the filters to continue.")
    footer()
    st.stop()

# ---------------------------------------------------------------------------
# Filtered summary
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Filtered sample")
f1, f2, f3, f4 = st.columns(4)
with f1:
    st.metric("Records", f"{len(filtered):,}")
with f2:
    st.metric("Average salary", format_currency(filtered["monthly_salary"].mean()))
with f3:
    st.metric("Average credit score", format_number(filtered["credit_score"].mean()))
with f4:
    st.metric("Average max EMI", format_currency(filtered["max_monthly_emi"].mean()))

tab_dist, tab_rel, tab_data, tab_notes = st.tabs(
    ["Distributions", "Relationships", "Filtered data", "Data notes"]
)

with tab_dist:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Eligibility distribution")
        eligibility_counts = (
            filtered["emi_eligibility"]
            .value_counts()
            .rename_axis("Eligibility")
            .to_frame("Records")
        )
        st.bar_chart(eligibility_counts)
        st.dataframe(eligibility_counts, width="stretch")
    with right:
        st.markdown("#### EMI scenario distribution")
        scenario_counts = (
            filtered["emi_scenario"]
            .value_counts()
            .rename_axis("Scenario")
            .to_frame("Records")
        )
        st.bar_chart(scenario_counts)
        st.dataframe(scenario_counts, width="stretch")

    st.markdown("#### Credit score distribution")
    counts, edges = np.histogram(
        filtered["credit_score"].to_numpy(),
        bins=np.arange(300, 901, 25),
    )
    histogram = pd.DataFrame(
        {
            "Credit score bucket": [
                f"{int(edges[index])}–{int(edges[index + 1])}"
                for index in range(len(counts))
            ],
            "Records": counts,
        }
    )
    st.bar_chart(histogram.set_index("Credit score bucket")["Records"])

    st.markdown("#### Maximum EMI distribution")
    emi_counts, emi_edges = np.histogram(
        filtered["max_monthly_emi"].to_numpy(),
        bins=30,
    )
    emi_hist = pd.DataFrame(
        {
            "Maximum EMI bucket": [
                f"{format_currency(emi_edges[index], 0)}–{format_currency(emi_edges[index + 1], 0)}"
                for index in range(len(emi_counts))
            ],
            "Records": emi_counts,
        }
    )
    st.bar_chart(emi_hist.set_index("Maximum EMI bucket")["Records"])

with tab_rel:
    st.markdown("#### Salary vs maximum affordable EMI")
    scatter_data = filtered.sample(
        n=min(5_000, len(filtered)), random_state=42
    )
    st.scatter_chart(
        scatter_data,
        x="monthly_salary",
        y="max_monthly_emi",
        color="emi_eligibility",
        x_label="Monthly salary (₹)",
        y_label="Maximum monthly EMI (₹)",
    )
    st.caption(
        f"Showing {len(scatter_data):,} sampled rows. Colour = eligibility class."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Average max EMI by eligibility")
        by_class = (
            filtered.groupby("emi_eligibility", as_index=True)["max_monthly_emi"]
            .mean()
            .to_frame("Average max EMI (₹)")
        )
        st.bar_chart(by_class)
    with right:
        st.markdown("#### Average max EMI by scenario")
        by_scenario = (
            filtered.groupby("emi_scenario", as_index=True)["max_monthly_emi"]
            .mean()
            .to_frame("Average max EMI (₹)")
        )
        st.bar_chart(by_scenario)

    st.markdown("#### Correlation with maximum affordable EMI")
    correlation_columns = [
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
    ]
    present = [column for column in correlation_columns if column in filtered.columns]
    correlations = (
        filtered[present + ["max_monthly_emi"]]
        .apply(pd.to_numeric, errors="coerce")
        .corr()["max_monthly_emi"]
        .drop("max_monthly_emi")
        .sort_values()
        .to_frame("Correlation")
    )
    st.bar_chart(correlations, horizontal=True)
    st.caption(
        "Pearson correlation on the filtered sample. The target-derived "
        "`affordability_ratio` appears strong by construction; treat it as a "
        "leakage diagnostic, not a usable predictor."
    )

with tab_data:
    st.markdown("#### Filtered records")
    display_columns = [
        "emi_scenario",
        "emi_eligibility",
        "age",
        "gender",
        "monthly_salary",
        "credit_score",
        "existing_loans",
        "requested_amount",
        "requested_tenure",
        "max_monthly_emi",
    ]
    display_columns = [column for column in display_columns if column in filtered.columns]
    st.dataframe(
        filtered[display_columns].head(500),
        width="stretch",
        hide_index=True,
    )
    st.caption("Showing the first 500 filtered rows. Download the full filtered sample below.")
    st.download_button(
        "⬇️ Download filtered sample (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="emi_filtered_sample.csv",
        mime="text/csv",
        width="stretch",
    )

with tab_notes:
    st.markdown(
        f"""
        **Source and sampling**

        - Full cleaned dataset: **{full_records:,} rows**, 38 columns.
        - Shipped sample: **{len(sample):,} rows**, proportional stratified by
          `emi_eligibility` × `emi_scenario`, random seed 42.
        - Full-data aggregates live in `data/eda_summary.json`; the sample lives
          in `data/eda_sample.csv.gz`. The raw 75 MB dataset and 143 MB cleaned
          CSV are intentionally excluded from git and Docker.

        **Data-quality notes**

        - `credit_score` in the cleaned data ranges from 0 to 1200; this page
          filters to the standard 300–900 range.
        - `affordability_ratio` was derived from the regression target during
          training. It is retained in the sample for leakage analysis only.
        - Currency values are Indian rupees (₹); requested EMI is principal-only
          (amount ÷ tenure) because no interest rate is present.
        """
    )
    with st.expander("Full-dataset summary JSON"):
        st.json(summary)

footer()
