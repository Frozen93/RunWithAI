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
    fig = go.Figure(
        data=go.Scatter(
            x=df[metric_x],
            y=df[metric_y],
            mode="markers",
            marker=dict(size=5, color="rgba(137, 146, 255, 0.8)"),
            name="Data",
        )
    )

    try:
        m, b = np.polyfit(df[metric_x], df[metric_y], 1)
        fig.add_trace(
            go.Scatter(
                x=df[metric_x],
                y=m * df[metric_x] + b,
                mode="lines",
                name="Regression Line",
                line=dict(color="#f77f00", width=1),
            )
        )
    except:
        st.warning("Something was wrong with the data, try another metric")

    fig.update_layout(
        title=f"Scatter Plot of {metric_x} vs {metric_y} with Regression Line",
        xaxis_title=metric_x,
        yaxis_title=metric_y,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Correlation Coefficient between {metric_x} and {metric_y}:** {correlation:.2f}")


def plot_distance_histogram(df):
    fig = go.Figure(
        data=[
            go.Histogram(
                x=df["distance_km"],
                nbinsx=30,
                marker_color="rgba(231, 29, 54, 0.5)",
                marker=dict(line=dict(color="rgba(238, 98, 116, 0.8)", width=1)),
            ),
        ]
    )
    fig.update_layout(
        title="Distribution of Running distance_meters",
        xaxis_title="distance_meters (km)",
        yaxis_title="Number of Runs",
        bargap=0.1,
    )
    st.plotly_chart(fig, use_container_width=True)


def adjust_heart_rate_for_cardiac_drift(row):
    # Assuming 5% adjustment at 60 minutes or 3600 seconds
    adjustment = 1 - 0.05 * (row['moving_time_seconds'] / 3600)

    # Clamping the adjustment between 0.95 (5% decrease) and 1 (no adjustment)
    adjustment = max(0.95, min(1, adjustment))

    return row['average_heartrate'] * adjustment


@st.cache_data
def plot_heart_rate_efficiency(df: pd.DataFrame):
    try:
        df = df.dropna(subset=['distance_km', 'total_elevation_gain', 'moving_time_seconds', 'average_heartrate'])

        ELEVATION_ADJUSTMENT_FACTOR = 11
        df['additional_distance'] = ELEVATION_ADJUSTMENT_FACTOR * df['total_elevation_gain']
        df['adjusted_distance'] = df['distance_km'] + df['additional_distance'] / 1000
        df['adjusted_speed'] = df['adjusted_distance'] / df['moving_time_seconds'] * 1000
        df['adjusted_heartrate'] = df.apply(adjust_heart_rate_for_cardiac_drift, axis=1)
        df['decay'] = df['distance_km'].apply(lambda d: 1 - 0.12 * (1 - d / 8) if d <= 6 else 1)
        df['heart_rate_efficiency'] = (df['adjusted_speed'] / df['adjusted_heartrate']) * df['decay']
        customdata = df[
            ["distance_km", "pace", "average_heartrate", "total_elevation_gain"]
        ].values
        hovertemplate = (
            "<b>Date:</b> %{x}<br><b>Efficiency:</b> %{y:.2f}<br>"
            "<b>Distance:</b> %{customdata[0]:.2f} km<br>"
            "<b>Pace:</b> %{customdata[1]:.2f} min/km<br>"
            "<b>Average Heartrate:</b> %{customdata[2]:.2f}<br>"
            "<b>Elevation Gain:</b> %{customdata[3]:.2f} m<br>"
            "<extra></extra>"  
        )
        # Regression Line Calculation
        df['heart_rate_efficiency'] = pd.to_numeric(df['heart_rate_efficiency'], errors='coerce')
        mask = pd.notna(df['heart_rate_efficiency'])  # Mask for non-NaN values
        slope, intercept = np.polyfit(df[mask]['date'].astype(np.int64), df[mask]['heart_rate_efficiency'], 1)
        df['regression_line'] = slope * df['date'].astype(np.int64) + intercept

        # Plotting
        fig = go.Figure(
            data=[
                go.Scatter(
                    y=df['heart_rate_efficiency'] * 10,
                    x=df['date'],
                    mode='lines+markers',
                    line=dict(color='rgba(60, 75, 255, 0.6)', width=2.5),
                    marker=dict(color="rgba(137, 146, 255, 0.8)", size=8),
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                    name="Data",
                ),
                # Adding Regression Line
                go.Scatter(
                    y=df['regression_line'] * 10,
                    x=df['date'],
                    mode='lines',
                    line=dict(color='rgba(231, 29, 54, 0.8)', width=1.5),
                    name="Regression Line",
                ),
            ]
        )

        fig.update_layout(
            title="Heart Rate Efficiency Over Time",
            yaxis_title="Heart Rate Efficiency",
            xaxis_title="Date",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning("A problem occured: " + str(e))


def plot_fatigue_sport(df):
    try:
        median_heartrate = df['average_heartrate'].replace('None', np.nan).dropna().median()
        df['average_heartrate'].replace({"None": np.nan}, inplace=True)
        df['average_heartrate'].fillna(median_heartrate, inplace=True)

        df['HRPR'] = df['average_heartrate'] / df['average_speed_metres_per_second']

        df['date'] = df['date'].dt.tz_localize(None)
        df['week'] = df['date'].dt.to_period('W-MON')

        df['days_since_last_workout'] = df['date'].diff().dt.days

        weekly_data = (
            df.groupby('week')
            .agg({'distance_km': 'sum', 'max_heartrate': 'mean', 'HRPR': 'mean', 'days_since_last_workout': 'mean'})
            .reset_index()
        )

        weekly_data.columns = ['week', 'Weekly Volume', 'Weekly Intensity', 'HRPR', 'Days Since Last']

        weekly_data['Normalized HRPR'] = (weekly_data['HRPR'] - weekly_data['HRPR'].min()) / (
            weekly_data['HRPR'].max() - weekly_data['HRPR'].min()
        )
        weekly_data['Normalized Volume'] = (weekly_data['Weekly Volume'] - weekly_data['Weekly Volume'].min()) / (
            weekly_data['Weekly Volume'].max() - weekly_data['Weekly Volume'].min()
        )
        weekly_data['Normalized Intensity'] = (
            weekly_data['Weekly Intensity'] - weekly_data['Weekly Intensity'].min()
        ) / (weekly_data['Weekly Intensity'].max() - weekly_data['Weekly Intensity'].min())
        DECAY_FACTOR = 0.7
        weekly_data['Fatigue Adjustment'] = 1 - (weekly_data['Days Since Last'] * (1 - DECAY_FACTOR))
        weekly_data['Fatigue'] = (
            100
            * weekly_data['Fatigue Adjustment']
            * (weekly_data['Normalized HRPR'] +  weekly_data['Normalized Volume'] +  weekly_data['Normalized Intensity'])
            / 3
            + 10
        )

        current_fatigue = weekly_data['Fatigue'].iloc[-1]
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=['Fatigue', 'Remaining'],
                    values=[current_fatigue, 100 - current_fatigue],
                    marker=dict(colors=['rgb(190, 15, 15)', 'rgb(38, 175, 38)']),
                    textfont=dict(color='white', size=15, family="Courier New, bold"),
                    showlegend=False,
                    hole=0.5,
                )
            ]
        )
        fig.update_layout(margin=dict(t=0, b=0, l=30, r=50))
        st.plotly_chart(
            fig,
            use_container_width=True,
        )
    except Exception as e:
        st.warning("A problem occured: " + str(e))


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
            go.color(
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
                    "Min: %{customdata.min}<br>" "Max: %{customdata.max}<br>" "Median: %{customdata.median}<br>"
                ),
            )
        )

    fig.update_layout(
        title="Distribution of pace for Each Month",
        yaxis_title="pace (min/km)",
    )

    st.plotly_chart(fig, use_container_width=True)
