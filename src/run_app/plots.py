import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np


def plot_scatter_metrics_with_regression(df: pd.DataFrame, metrics: list):
    st.subheader("Check for correlations with a regression line")
    metric_x = st.selectbox("Select metric for x-axis:", metrics, index=0)
    metric_y = st.selectbox("Select metric for y-axis:", metrics, index=1)
    df = df.dropna(subset=[metric_x, metric_y])
    df[metric_x] = pd.to_numeric(df[metric_x], errors='coerce')
    df[metric_y] = pd.to_numeric(df[metric_y], errors='coerce')

    correlation = df[metric_x].corr(df[metric_y])
    fig = go.Figure(data=go.Scatter(x=df[metric_x], y=df[metric_y], mode="markers", marker=dict(size=5), name="Data"))

    try:
        m, b = np.polyfit(df[metric_x], df[metric_y], 1)
        fig.add_trace(go.Scatter(x=df[metric_x], y=m * df[metric_x] + b, mode="lines", name="Regression Line"))
    except:
        st.warning("Something was wrong with the data, try another metric")

    fig.update_layout(
        title=f"Scatter Plot of {metric_x} vs {metric_y} with Regression Line",
        xaxis_title=metric_x,
        yaxis_title=metric_y,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Correlation Coefficient between {metric_x} and {metric_y}:** {correlation:.2f}")


def plot_distance_histogram(df):
    fig = go.Figure(
        data=[
            go.Histogram(
                x=df["distance_km"],
                nbinsx=50,
                marker_color="rgba(231, 29, 54, 0.5)",
                marker=dict(line=dict(color="rgba(238, 98, 116, 0.8)", width=1)),
            ),
        ]
    )
    fig.update_layout(
        title="Distribution of Running distance_meterss",
        xaxis_title="distance_meters (km)",
        yaxis_title="Number of Runs",
        bargap=0.1,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_selected_metrics(df: pd.DataFrame, metrics: list):
    st.subheader("Metrics Over Time")
    selected_metrics = st.multiselect("Select metrics to plot:", metrics, default=["distance_km"])
    months_back = st.slider("Select how many months to view:", 1, 24, 6)
    end_date = df["date"].max()
    start_date = end_date - pd.dateOffset(months=months_back)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    if not selected_metrics:
        st.warning("Please select at least one metric to plot.")
        return

    fig = go.Figure()
    for metric in selected_metrics:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[metric],
                mode="lines",
                name=metric,
            )
        )
    fig.update_layout(title="Metrics Over Time", xaxis_title="date")
    fig.update_xaxes(range=[df["date"].min(), df["date"].max()])
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_monthly_avg_pace(df: pd.DataFrame):
    monthly_avg = df.groupby("month-year")["pace"].mean().reset_index()
    fig = go.Figure(
        data=[
            go.Bar(
                x=monthly_avg["month-year"],
                y=monthly_avg["pace"],
                marker_color="rgba(164, 61, 174, 0.62)",
                marker=dict(line=dict(color="rgba(195, 108, 203, 0.8)", width=1)),
            )
        ]
    )
    fig.update_layout(
        title="Average pace per Month",
        xaxis_title="month-year",
        yaxis_title="Average pace",
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_cumulative_kms_per_month(df: pd.DataFrame):
    monthly_sum = df.groupby("month-year")["distance_km"].sum().reset_index()
    fig = go.Figure(
        data=[
            go.Bar(
                y=monthly_sum["distance_km"],
                x=monthly_sum["month-year"],
                name="distance_km",
                marker_color="rgba(60, 75, 255, 0.6)",
                marker=dict(line=dict(color="rgba(137, 146, 255, 0.8)", width=1)),
            )
        ]
    )

    fig.update_layout(
        title="Total distance_meters per Month",
        xaxis_title="month-year",
        yaxis_title="Total distance_meters",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_pace_distribution(df):
    df["YearMonth"] = df["date"].dt.strftime("%Y-%m")
    fig = go.Figure()
    months = sorted(df["YearMonth"].unique())
    for month in months:
        month_data = df[df["YearMonth"] == month]
        fig.add_trace(
            go.Box(
                y=month_data["pace"],
                name=month,
                boxpoints=False,
                hoverinfo="y+name",
                customdata=month_data["pace"],
                showlegend=False,
                line=dict(color="#f77f00"),
                hovertemplate=(
                    "Min: %{customdata.min}<br>"
                    "Max: %{customdata.max}<br>"
                    "Median: %{customdata.median}<br>"
                    "month-year: %{name}"
                ),
            )
        )

    fig.update_layout(
        title="Distribution of pace for Each Month",
        xaxis_title="month-year",
        yaxis_title="pace (min/km)",
    )

    st.plotly_chart(fig, use_container_width=True)
