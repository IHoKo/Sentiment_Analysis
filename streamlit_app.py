from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = PROJECT_ROOT / "Script"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sentiment_methods import (  # noqa: E402
    NLP_OUTPUT_PATH,
    RULE_OUTPUT_PATH,
    add_nlp_sentiment,
    add_rule_based_sentiment,
    compare_methods,
    load_event_data,
)


st.set_page_config(
    page_title="Event Sentiment Method Comparison",
    page_icon="",
    layout="wide",
)

PLOT_BG = "#111827"
POSITIVE_COLOR = "#2E7D32"
NEUTRAL_COLOR = "#757575"
NEGATIVE_COLOR = "#C62828"
LABEL_COLORS = {
    "Positive": POSITIVE_COLOR,
    "Neutral": NEUTRAL_COLOR,
    "Negative": NEGATIVE_COLOR,
}


@st.cache_data
def load_comparison_data() -> pd.DataFrame:
    event_data = load_event_data()
    rule_data = add_rule_based_sentiment(event_data)
    nlp_data = add_nlp_sentiment(event_data)

    return pd.DataFrame(
        {
            "Event": event_data["Event"],
            "Type": event_data["Type"],
            "Registers": event_data["Registers"],
            "Room Available/Not Available": event_data["Room Available/Not Available"],
            "Accept/Reject": event_data["Accept/Reject"],
            "Rule Score": rule_data["Rule Score"],
            "Rule Label": rule_data["Rule Label"],
            "Rule Reason": rule_data["Rule Reason"],
            "NLP Score": nlp_data["NLP Score"],
            "NLP Label": nlp_data["NLP Label"],
            "NLP Reason": nlp_data["NLP Reason"],
            "NLP Text": nlp_data["NLP Text"],
        }
    )


def style_plot(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font_color="#F9FAFB",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def label_badge(label: str) -> str:
    color = LABEL_COLORS.get(label, NEUTRAL_COLOR)
    return f"<span style='background:{color}; color:white; padding:0.2rem 0.5rem; border-radius:0.25rem;'>{label}</span>"


comparison_data = load_comparison_data()

st.title("Event Sentiment Comparison")
st.caption("Compare deterministic rule-based scoring with VADER NLP sentiment analysis.")

summary_cols = st.columns(4)
summary_cols[0].metric("Events", len(comparison_data))
summary_cols[1].metric("Rule Avg Score", f"{comparison_data['Rule Score'].mean():.2f}")
summary_cols[2].metric("NLP Avg Score", f"{comparison_data['NLP Score'].mean():.2f}")
agreement_rate = (comparison_data["Rule Label"] == comparison_data["NLP Label"]).mean() * 100
summary_cols[3].metric("Label Agreement", f"{agreement_rate:.1f}%")

tab_new, tab_dataset, tab_charts = st.tabs(["New Entry", "Dataset Comparison", "Charts"])

with tab_new:
    st.subheader("Compare Both Methods For A New Event")

    with st.form("new_event_form"):
        left, right = st.columns(2)
        with left:
            event = st.text_input("Event", value="AI workshop")
            faculty_id = st.text_input("Faculty ID", value="FAC010999")
            faculty_name = st.text_input("Faculty Name", value="New Faculty")
            faculty_designation = st.text_input("Faculty Designation", value="Associate Professor")
            registers = st.number_input("Registers", min_value=0, max_value=10000, value=50, step=1)
            priority = st.selectbox("Type", ["Normal", "Important", "Very important"], index=1)
        with right:
            booking_requested = st.checkbox("Request for Booking", value=True)
            cancellation_requested = st.checkbox("Request for Cancelling", value=False)
            requested_room = st.number_input("Requested Room No", min_value=1, max_value=9999, value=202, step=1)
            request_date = st.date_input("Request Date")
            requested_time = st.text_input("Requested Time", value="3:00pm")
            duration = st.text_input("Duration", value="120min")
            room_available = st.selectbox("Room Available/Not Available", ["Available", "Not Available"])
            decision = st.selectbox("Accept/Reject", ["Accept", "Reject"])

        submitted = st.form_submit_button("Analyze New Entry", type="primary")

    if submitted:
        new_row = pd.Series(
            {
                "SNO": "New",
                "Event": event,
                "Faculty_id": faculty_id,
                "Faculty_name": faculty_name,
                "Faculty_Designation": faculty_designation,
                "Registers": registers,
                "Type": priority,
                "Request for Booking": int(booking_requested),
                "Request for Cancelling": int(cancellation_requested),
                "Requested Room No": requested_room,
                "Request Date": request_date.strftime("%m/%d/%Y"),
                "Requested Time": requested_time,
                "Duration": duration,
                "Room Available/Not Available": room_available,
                "Accept/Reject": decision,
            }
        )

        new_comparison = compare_methods(new_row)
        score_fig = px.bar(
            new_comparison,
            x="Method",
            y="Score",
            color="Label",
            text="Score",
            color_discrete_map=LABEL_COLORS,
            title="New Entry Score Comparison",
        )
        score_fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#E5E7EB")
        score_fig.update_layout(yaxis_range=[-100, 100])
        st.plotly_chart(style_plot(score_fig), use_container_width=True)

        result_cols = st.columns(2)
        for index, result in new_comparison.iterrows():
            with result_cols[index]:
                st.markdown(f"### {result['Method']}")
                st.markdown(label_badge(result["Label"]), unsafe_allow_html=True)
                st.metric("Score", f"{result['Score']:.2f}")
                st.write(result["Reason"])

        with st.expander("New entry data"):
            st.dataframe(pd.DataFrame([new_row]), use_container_width=True)
    else:
        st.info("Enter event details and click Analyze New Entry.")

with tab_dataset:
    st.subheader("Existing Dataset Method Comparison")

    selected_label = st.multiselect(
        "Filter by final labels",
        ["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"],
    )
    filtered_data = comparison_data[
        comparison_data["Rule Label"].isin(selected_label)
        | comparison_data["NLP Label"].isin(selected_label)
    ]

    st.dataframe(
        filtered_data[
            [
                "Event",
                "Type",
                "Registers",
                "Room Available/Not Available",
                "Accept/Reject",
                "Rule Score",
                "Rule Label",
                "NLP Score",
                "NLP Label",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    event_choice = st.selectbox("Inspect one event", filtered_data["Event"].tolist())
    event_row = filtered_data.loc[filtered_data["Event"] == event_choice].iloc[0]
    inspect_cols = st.columns(2)
    with inspect_cols[0]:
        st.markdown("### Rule-based")
        st.metric("Score", f"{event_row['Rule Score']:.2f}")
        st.markdown(label_badge(event_row["Rule Label"]), unsafe_allow_html=True)
        st.write(event_row["Rule Reason"])
    with inspect_cols[1]:
        st.markdown("### VADER NLP")
        st.metric("Score", f"{event_row['NLP Score']:.2f}")
        st.markdown(label_badge(event_row["NLP Label"]), unsafe_allow_html=True)
        st.write(event_row["NLP Reason"])
        with st.expander("NLP text sent to VADER"):
            st.write(event_row["NLP Text"])

with tab_charts:
    st.subheader("Method-Level Charts")

    long_scores = pd.concat(
        [
            comparison_data[["Event", "Rule Score", "Rule Label"]].rename(
                columns={"Rule Score": "Score", "Rule Label": "Label"}
            ).assign(Method="Rule-based"),
            comparison_data[["Event", "NLP Score", "NLP Label"]].rename(
                columns={"NLP Score": "Score", "NLP Label": "Label"}
            ).assign(Method="VADER NLP"),
        ],
        ignore_index=True,
    )

    score_fig = px.bar(
        long_scores.sort_values(["Method", "Score"]),
        x="Score",
        y="Event",
        color="Label",
        facet_col="Method",
        orientation="h",
        color_discrete_map=LABEL_COLORS,
        title="Scores by Event and Method",
    )
    score_fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#E5E7EB")
    score_fig.update_layout(xaxis_range=[-100, 100], xaxis2_range=[-100, 100], height=700)
    st.plotly_chart(style_plot(score_fig), use_container_width=True)

    scatter_fig = px.scatter(
        comparison_data,
        x="Rule Score",
        y="NLP Score",
        color="Rule Label",
        hover_name="Event",
        hover_data=["NLP Label", "Registers", "Accept/Reject"],
        color_discrete_map=LABEL_COLORS,
        title="Rule-Based Score vs. VADER NLP Score",
    )
    scatter_fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#E5E7EB")
    scatter_fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#E5E7EB")
    scatter_fig.update_layout(xaxis_range=[-100, 100], yaxis_range=[-100, 100])
    st.plotly_chart(style_plot(scatter_fig), use_container_width=True)

    label_counts = long_scores.groupby(["Method", "Label"]).size().reset_index(name="Events")
    count_fig = px.bar(
        label_counts,
        x="Method",
        y="Events",
        color="Label",
        barmode="group",
        text="Events",
        color_discrete_map=LABEL_COLORS,
        title="Label Counts by Method",
    )
    count_fig.update_traces(textposition="outside")
    st.plotly_chart(style_plot(count_fig), use_container_width=True)

with st.sidebar:
    st.header("Files")
    st.write(f"Rule output: `{RULE_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`")
    st.write(f"NLP output: `{NLP_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`")
    st.header("Methods")
    st.write("Rule-based scoring uses explicit structured signals.")
    st.write("VADER NLP analyzes generated natural-language text with a sentiment lexicon and linguistic rules.")
