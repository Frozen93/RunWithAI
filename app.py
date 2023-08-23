import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import calendar


def setup_config():
    """Configures Streamlit app settings."""
    st.set_page_config(
        page_title="Run with AI",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_styles():
    """Applies custom styles to Streamlit app."""
    hide_default_format = """
           <style>
           #MainMenu {visibility: visible; }
           footer {visibility: hidden;}
           </style>
           """
    st.markdown(hide_default_format, unsafe_allow_html=True)


@st.cache_data
def load_data(data_source: str) -> pd.DataFrame:
    """Loads and preprocesses running data."""
    data = pd.read_csv(data_source)
    data["Date"] = pd.to_datetime(data["Date"])
    data["Pace"] = data["Pace"].apply(convert_pace)

    return data.sort_values(by="Date")


def convert_pace(pace):
    minutes = int(pace)
    fraction = pace - minutes
    seconds_fraction = fraction * 60 / 100
    return minutes + seconds_fraction


def pace_threshold(df):
    pace_threshold = st.number_input(
        "Enter a pace threshold (exclude records with pace above this value):",
        value=7.0,
        step=0.1,
    )
    df = df[df["Pace"] <= pace_threshold]
    return df


def distance_threshold(df):
    dist_threshold = st.number_input(
        "Enter a distance threshold (exclude records with distance below this value):",
        value=10,
        step=1,
    )
    df = df[df["Distance"] >= dist_threshold]
    return df


def plot_selected_metrics(df: pd.DataFrame, metrics: list):
    """Plots the selected metrics over time."""
    st.subheader("Metrics Over Time")  #
    selected_metrics = st.multiselect(
        "Select metrics to plot:", metrics, default=["Distance"]
    )
    if not selected_metrics:
        st.warning("Please select at least one metric to plot.")
        return
    fig = go.Figure()
    for metric in selected_metrics:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df[metric],
                mode="markers",
                name=metric,
            )
        )
    fig.update_layout(title="Metrics Over Time", xaxis_title="Date")
    fig.update_xaxes(range=[df["Date"].min(), df["Date"].max()])
    st.plotly_chart(fig, use_container_width=True)


def plot_monthly_avg_pace(df: pd.DataFrame):
    """Plots the average pace for every month using plotly."""

    # Extract month and year from "Date" and create a new "Month-Year" column
    df["Month-Year"] = df["Date"].dt.strftime("%Y-%m")

    # Group by "Month-Year" and calculate the average pace
    monthly_avg = df.groupby("Month-Year")["Pace"].mean().reset_index()

    # Plot the monthly average pace using plotly
    fig = go.Figure(
        data=[
            go.Bar(
                x=monthly_avg["Month-Year"],
                y=monthly_avg["Pace"],
                marker_color="#7209b7",
            )
        ]
    )
    fig.update_layout(
        title="Average Pace per Month",
        xaxis_title="Month-Year",
        yaxis_title="Average Pace",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_cumulative_kms_per_month(df: pd.DataFrame):
    """Plots the total distance for every month using a box plot in plotly."""

    # Extract month and year from "Date" and create a new "Month-Year" column
    df["Month-Year"] = df["Date"].dt.strftime("%Y-%m")

    # Group by "Month-Year" and sum the distances for each month
    monthly_sum = df.groupby("Month-Year")["Distance"].sum().reset_index()

    # Create a box plot of the monthly summed distances
    fig = go.Figure(
        data=[
            go.Bar(
                y=monthly_sum["Distance"],
                x=monthly_sum["Month-Year"],
                name="Distance",
                marker_color="#3f37c9",
            )
        ]
    )

    fig.update_layout(
        title="Total Distance per Month",
        xaxis_title="Month-Year",
        yaxis_title="Total Distance",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_pace_distribution(df):
    df["YearMonth"] = df["Date"].dt.strftime("%Y-%m")

    # Create the box plot using graph_objects
    fig = go.Figure()

    months = sorted(df["YearMonth"].unique())
    for month in months:
        month_data = df[df["YearMonth"] == month]
        fig.add_trace(
            go.Box(
                y=month_data["Pace"],
                name=month,
                boxpoints=False,  # Disable showing all points
                hoverinfo="y+name",
                customdata=month_data["Pace"],
                showlegend=False,
                line=dict(color="#f77f00"),
                hovertemplate=(
                    "Min: %{customdata.min}<br>"
                    "Max: %{customdata.max}<br>"
                    "Median: %{customdata.median}<br>"
                    "Month-Year: %{name}"
                ),
            )
        )

    fig.update_layout(
        title="Distribution of Pace for Each Month",
        xaxis_title="Month-Year",
        yaxis_title="Pace (min/km)",
    )

    st.plotly_chart(fig, use_container_width=True)


def display_comparison_metrics(df: pd.DataFrame):
    """
    Displays a comparison of metrics for the last 30 days against the previous 30 days.
    Additionally, shows the overall metrics for the entire dataset.
    """

    # Extract the end date of the dataset for calculation reference
    end_date = df["Date"].max()

    # Compute metrics for the last 30 days
    last_30_days = df[
        (df["Date"] <= end_date) & (df["Date"] > end_date - pd.Timedelta(days=30))
    ]

    # Compute metrics for the previous 30 days
    previous_30_days = df[
        (df["Date"] <= end_date - pd.Timedelta(days=30))
        & (df["Date"] > end_date - pd.Timedelta(days=60))
    ]

    # Compute metrics for the entire dataset
    total_records_all = round(df.shape[0], 2)
    total_distance_all = round(df["Distance"].sum(), 2)
    avg_pace_all = round(df["Pace"].mean(), 2)

    # Define metrics
    metrics_last_30 = {
        "Total Records": round(last_30_days.shape[0], 2),
        "Total Distance": round(last_30_days["Distance"].sum(), 2),
        "Average Pace": round(last_30_days["Pace"].mean(), 2),
    }

    metrics_prev_30 = {
        "Total Records": round(previous_30_days.shape[0], 2),
        "Total Distance": round(previous_30_days["Distance"].sum(), 2),
        "Average Pace": round(previous_30_days["Pace"].mean(), 2),
    }

    metrics_all_time = {
        "Total Records": total_records_all,
        "Total Distance": total_distance_all,
        "Average Pace": avg_pace_all,
    }

    # Display metrics in Streamlit
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("All Time Metrics")
        for metric, value in metrics_all_time.items():
            st.metric(label=metric, value=value)

    with col2:
        st.subheader("Previous 30 Days")
        for metric, value in metrics_prev_30.items():
            st.metric(label=metric, value=value)

    with col3:
        st.subheader("Last 30 Days")
        for metric, value in metrics_last_30.items():
            if metric == "Average Pace":
                delta_val = value - metrics_prev_30[metric]
                st.metric(
                    label=metric,
                    value=value,
                    delta=round(delta_val, 2),
                    delta_color="inverse",
                )
            else:
                delta_val = value - metrics_prev_30[metric]
                st.metric(label=metric, value=value, delta=round(delta_val, 2))


def activity_heatmap(df):
    distances = [[0 for _ in range(52)] for _ in range(7)]
    full_dates = [["" for _ in range(52)] for _ in range(7)]

    for _, row in df.iterrows():
        week_of_year = row["Date"].isocalendar()[1] - 1
        day_of_week = row["Date"].weekday()

        full_dates[day_of_week][week_of_year] = row["Date"].strftime("%Y-%m-%d")
        distances[day_of_week][week_of_year] = row["Distance"]

    colorscale = [
        [0.0, "rgba(10, 10, 10, 1)"],
        [0.1, "rgba(30, 165, 30, 0.1)"],
        [0.3, "rgba(40, 180, 40, 0.4)"],
        [0.5, "rgba(50, 195, 50, 0.55)"],
        [0.7, "rgba(60, 210, 60, 0.7)"],
        [0.9, "rgba(65, 225, 65, 0.85)"],
        [1.0, "rgba(70, 236, 70, 1)"],
    ]
    hover = [
        [
            f"Day: {date}<br>Distance: {dist} km" if date else ""
            for date, dist in zip(date_row, dist_row)
        ]
        for date_row, dist_row in zip(full_dates, distances)
    ]

    fig = ff.create_annotated_heatmap(
        distances,
        colorscale=colorscale,
        text=hover,
        hoverinfo="text",
        xgap=3,
        ygap=3,
    )
    heatmap_annotations = []

    for ann in fig.layout.annotations:
        if ann.text != "0":
            try:
                rounded_text = str(round(float(ann.text), 1))
                ann.text = rounded_text
                heatmap_annotations.append(ann)
            except ValueError:
                heatmap_annotations.append(ann)
    all_annotations = heatmap_annotations + [
        dict(
            x=0.5,
            y=-0.01,
            text="Jan                 Feb                 Mar                 Apr                 Mai                 Jun                 Jul                 Aug                 Sep                 Okt                 Nov                 Dez",  # x-axis title
            xref="paper",
            yref="paper",
            align="center",
        )
    ]

    fig.update_layout(
        title="Distance Run - Every day of the year!",
        autosize=False,
        yaxis_title="Mon Tue Wed Thu Fr Sat Sun",
        width=1800,  # 7 boxes * 100 pixels/box + 30 pixels for padding
        height=500,  # 52 boxes * 7 pixels/box + 15 pixels for padding
        xaxis=dict(constrain="domain"),
        yaxis=dict(scaleanchor="x"),
        annotations=all_annotations,
    )
    fig.update_yaxes(
        tickvals=list(range(7)),
        ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    )
    for ann in fig.layout.annotations:
        ann.font.size = 9
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main function of the Streamlit App."""
    setup_config()
    apply_styles()

    l, r = st.columns(2)
    with l:
        st.header("AI Runner")
    with r:
        st_lottie(
            "https://lottie.host/a2b2ddf8-f030-46fa-b3b2-8c1727afb253/h2zfkvSzpy.json",
            height=100,
        )

    data_url = "https://docs.google.com/spreadsheets/d/139ckZPhjRzwmDayTSwSVXzIZUlwMGPqqTQwNg3EIKj0/export?format=csv&gid=0"
    df = load_data(data_url)
    df = df[df["Date"] >= "2023-01-01"]

    pace, threshold = st.columns(2)
    with pace:
        df = pace_threshold(df)
    with threshold:
        df = distance_threshold(df)

    activity_heatmap(df)

    with st.expander("Show raw data"):
        st.dataframe(
            df.drop(columns=["ID", "Average Speed", "Time_sec"]),
            use_container_width=True,
        )

    metrics_list = [
        "Distance",
        "HeartRate",
        "Pace",
        "Time",
        "ElevGain",
        "Temperature",
    ]
    plot_selected_metrics(df, metrics_list)
    display_comparison_metrics(df)
    plot_monthly_avg_pace(df)
    plot_cumulative_kms_per_month(df)
    plot_pace_distribution(df)


if __name__ == "__main__":
    main()
