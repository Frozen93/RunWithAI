import streamlit as st
import pandas as pd
import plotly.graph_objects as go


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
           #MainMenu {visibility: hidden; }
           footer {visibility: hidden;}
           </style>
           """
    st.markdown(hide_default_format, unsafe_allow_html=True)


def pace_to_minutes(pace: str) -> float:
    """Converts pace in 'X:YY' format to decimal minutes."""
    if ":" in pace:
        minutes, seconds = pace.split(":")
        return float(minutes) + float(seconds) / 60
    return float(pace)


@st.cache_data
def load_data(data_source: str) -> pd.DataFrame:
    """Loads and preprocesses running data."""
    data = pd.read_csv(data_source)
    data["Date"] = pd.to_datetime(data["Date"])
    data["Pace"] = data["Pace"].apply(pace_to_minutes)
    data["PacePerHeartRate"] = data["Pace"] / data["HeartRate"]
    return data.sort_values(by="Date")


def plot_selected_metrics(df: pd.DataFrame, metrics: list):
    """Plots the selected metrics over time."""
    selected_metrics = st.multiselect(
        "Select metrics to plot:", metrics, default=["Distance"]
    )
    st.subheader("Metrics Over Time")
    if not selected_metrics:
        st.warning("Please select at least one metric to plot.")
        return
    fig = go.Figure()
    for metric in selected_metrics:
        fig.add_trace(go.Scatter(x=df["Date"], y=df[metric], mode="lines", name=metric))
    fig.update_layout(title="Metrics Over Time", xaxis_title="Date")
    fig.update_xaxes(range=[df["Date"].min(), df["Date"].max()])
    st.plotly_chart(fig)


def display_statistics(df: pd.DataFrame):
    """Displays various statistics about the running data."""
    st.subheader("Running Statistics")
    st.write("Total Distance Ran:", df["Distance"].sum().round(2), "km")
    st.write("Average Heart Rate:", df["HeartRate"].mean(), "bpm")
    st.write("Best Pace:", df["Pace"].min(), "min/km")
    st.write("Total Elevation Gain:", df["ElevGain"].sum(), "meters")


def main():
    """Main function of the Streamlit App."""
    setup_config()
    apply_styles()

    st.header("AI Runner")

    data_url = "https://docs.google.com/spreadsheets/d/139ckZPhjRzwmDayTSwSVXzIZUlwMGPqqTQwNg3EIKj0/export?format=csv&gid=0"
    df = load_data(data_url)
    st.dataframe(df.drop(columns=["ID"]), use_container_width=True)

    metrics_list = [
        "Distance",
        "HeartRate",
        "Pace",
        "Time",
        "ElevGain",
        "PacePerHeartRate",
    ]
    plot_selected_metrics(df, metrics_list)
    display_statistics(df)


if __name__ == "__main__":
    main()
