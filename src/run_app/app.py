import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
import plotly.figure_factory as ff
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from pydantic import ValidationError
from langchain.agents.openai_functions_agent.base import OutputParserException
import requests
import json
import time
import textwrap
import plots


def setup_config():
    """Configures Streamlit app settings."""
    st.set_page_config(
        page_title="Run with AI",
        page_icon="💧",
        layout="wide",
    )
    st.markdown(
        """ <style>
    footer {visibility: hidden;}

    footer:hover,  footer:active {
        color: #ffffff;
        background-color: transparent;
        text-decoration: underline;
        transition: 400ms ease 0s;
    }
    </style>""",
        unsafe_allow_html=True,
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
    data["Month-Year"] = data["Date"].dt.strftime("%Y-%m")
    data["Pace"] = data["Pace"].apply(convert_pace)

    data = data[data["Date"] >= "2023-01-01"]
    return data.sort_values(by="Date")


def convert_pace(pace):
    minutes = int(pace)
    fraction = pace - minutes
    seconds_fraction = fraction * 60 / 100
    return minutes + seconds_fraction


def pace_threshold(df):
    pace_threshold = st.number_input(
        "Enter a pace threshold (exclude records with pace above/slower than this value):",
        value=7.2,
        step=0.1,
    )
    df = df[df["Pace"] <= pace_threshold]
    return df


def distance_threshold(df):
    dist_threshold = st.number_input(
        "Enter a distance threshold (exclude short runs below this value):",
        value=4,
        step=1,
    )
    df = df[df["Distance"] >= dist_threshold]
    return df


def display_comparison_metrics(df: pd.DataFrame):
    """
    Displays a comparison of metrics for the last 30 days against the previous 30 days.
    Additionally, shows the overall metrics for the entire dataset.
    """

    # Extract the end date of the dataset for calculation reference
    end_date = df["Date"].max()

    # Compute metrics for the last 30 days
    last_30_days = df[(df["Date"] <= end_date) & (df["Date"] > end_date - pd.Timedelta(days=30))]

    # Compute metrics for the previous 30 days
    previous_30_days = df[
        (df["Date"] <= end_date - pd.Timedelta(days=30)) & (df["Date"] > end_date - pd.Timedelta(days=60))
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
        [f"Day: {date}<br>Distance: {dist} km" if date else "" for date, dist in zip(date_row, dist_row)]
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
    months = [
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
        'July',
        'August',
        'September',
        'October',
        'November',
        'December',
    ]
    months_string = '                          '.join(months)
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
            y=0.1,
            text=months_string,
            xref="paper",
            yref="paper",
            align="center",
        )
    ]

    fig.update_layout(
        autosize=False,
        yaxis_title="Mon Tue Wed Thu Fr Sat Sun",
        width=1800,
        height=500,
        xaxis=dict(constrain="domain", showgrid=False, zeroline=False, showline=False),
        yaxis=dict(scaleanchor="x", showgrid=False, zeroline=False, showline=False),
        annotations=all_annotations,
        margin=dict(t=0, r=0, b=100, l=0),
    )
    fig.update_yaxes(
        tickvals=list(range(7)),
        ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    )
    for ann in fig.layout.annotations:
        ann.font.size = 14
    st.plotly_chart(fig, use_container_width=True)


def wrap_text(text, width=140):
    # Unwrap the text first to remove any existing line breaks
    unwrapped_text = ' '.join(text.splitlines())
    # Use textwrap to wrap the text
    return '\n'.join(textwrap.wrap(unwrapped_text, width=width))


def separate_table(content):
    # Split the content into lines
    lines = content.split('\n')

    # Find the delimiter row (which typically follows the header row)
    table_start_index = -1
    for index, line in enumerate(lines):
        if set(line.strip()) <= {'-', '|', ' '}:
            table_start_index = index - 1  # Assuming the previous line is the header
            break

    # If we found a table start
    if table_start_index != -1:
        # Gather the table lines
        table_lines = [lines[table_start_index]]
        for line in lines[table_start_index + 1 :]:
            if "|" in line:
                table_lines.append(line)
            else:
                break

        # Separate the table from the rest
        before_table = '\n'.join(lines[:table_start_index])
        table_content = '\n'.join(table_lines)
        after_table = '\n'.join(lines[table_start_index + len(table_lines) :])

        return before_table, table_content, after_table

    return content, "", ""


def fetch_gpt_response_test(query):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {st.secrets['gpt4_key']}"}
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": query}], "stream": True}

    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data, stream=True)

    accumulated_content = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data:'):
                try:
                    json_data = json.loads(decoded_line[5:])
                except json.JSONDecodeError:
                    continue
                choice = json_data.get('choices', [{}])[0]
                delta = choice.get('delta', {})
                content_chunk = delta.get('content', "")
                accumulated_content += content_chunk

                # Separate table content
                before_table, table_content, after_table = separate_table(accumulated_content)

                # Only wrap the non-table content
                formatted_content = wrap_text(before_table) + table_content + wrap_text(after_table)
                yield formatted_content


def fetch_gpt_response(query):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {st.secrets['gpt4_key']}"}
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": query}], "stream": True}

    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data, stream=True)

    accumulated_content = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data:'):
                try:
                    json_data = json.loads(decoded_line[5:])
                except json.JSONDecodeError:
                    continue
                choice = json_data.get('choices', [{}])[0]
                delta = choice.get('delta', {})
                content_chunk = delta.get('content', "")
                accumulated_content += content_chunk  # Accumulate the content
                formatted_content = wrap_text(accumulated_content)
                yield formatted_content


def init_langchain_agent(df):
    # Initialize agent
    return create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=st.secrets['gpt4_key']),
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )


def old_chatbot():
    st.markdown("___")
    base_promt = f"You are the running coach of the runner with the following data: {df.to_json()}, answer his questions short and to the point. Underline your statements with numbers to show improvement. Use the metric system and provide paces in min/km, distances in km . Mention times in minutes, not seconds. Provide a helpful table in the beginning. Question:  "
    user_input = st.text_input("Ask the AI about your running:")

    if st.button("Submit"):
        placeholder = st.empty()  # Placeholder for dynamic content
        for accumulated_response in fetch_gpt_response_test(base_promt + user_input):
            placeholder.markdown(accumulated_response)  # Update Streamlit with the accumulated text
            time.sleep(0.5)  # Optional: To slow down the streaming for better visualization

    st.markdown("___")


def main():
    """Main function of the Streamlit App."""
    setup_config()
    apply_styles()

    l, r, _ = st.columns((1, 1, 4))
    with l:
        st.header("AI Runner")
    with r:
        st_lottie(
            "https://lottie.host/a2b2ddf8-f030-46fa-b3b2-8c1727afb253/h2zfkvSzpy.json",
            height=120,
        )

    data_url = (
        "https://docs.google.com/spreadsheets/d/139ckZPhjRzwmDayTSwSVXzIZUlwMGPqqTQwNg3EIKj0/export?format=csv&gid=0"
    )
    df_raw = load_data(data_url)

    activity_heatmap(df_raw)
    pace, threshold = st.columns(2)
    with pace:
        df = pace_threshold(df_raw)
    with threshold:
        df = distance_threshold(df_raw)
    display_comparison_metrics(df)

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

    # plot_selected_metrics(df, metrics_list)
    plots.plot_scatter_metrics_with_regression(df, metrics_list)

    a, _, b = st.columns((6, 1, 6))
    with a:
        plots.plot_cumulative_kms_per_month(df)
        plots.plot_monthly_avg_pace(df)

    with b:
        plots.plot_pace_distribution(df)
        plots.plot_distance_histogram(df)

    st.subheader("Ask the AI any question related to your running data")

    user_input = st.text_input("Your question:", "")

    if user_input:
        try:
            # This is where you initialize and run your langchain agent.
            with st.spinner("AI at work!"):
                response = init_langchain_agent(df).run(user_input)
                st.markdown(response)

        except ValidationError:
            st.error("API Key Validation failed. Ensure your API key is correctly configured.")

        except ImportError:
            st.error("A required library is missing. Ensure you've installed all dependencies.")

        except OutputParserException:
            st.error("There was an error parsing the response. Please try a different query or check your data.")

        # Any other exceptions can be caught with a generic message
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
