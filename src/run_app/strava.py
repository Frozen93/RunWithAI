import base64
import arrow
import httpx
import streamlit as st
import pandas as pd
import os


# import sweat
from bokeh.models.widgets import Div


APP_URL = st.secrets["APP_URL"]
STRAVA_CLIENT_ID = st.secrets["STRAVA_CLIENT_ID"]
STRAVA_CLIENT_SECRET = st.secrets["STRAVA_CLIENT_SECRET"]
STRAVA_AUTHORIZATION_URL = "https://www.strava.com/oauth/authorize"
STRAVA_API_BASE_URL = "https://www.strava.com/api/v3"
DEFAULT_ACTIVITY_LABEL = "NO_ACTIVITY_SELECTED"
STRAVA_ORANGE = "#fc4c02"


@st.cache_data(show_spinner=False)
def load_image_as_base64(image_path):
    with open(image_path, "rb") as f:
        contents = f.read()
    return base64.b64encode(contents).decode("utf-8")


def powered_by_strava_logo():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_directory, 'images', 'by_strava.png')
    base64_image = load_image_as_base64(image_path)
    st.markdown(
        f'<img src="data:image/png;base64,{base64_image}" width="50%" alt="powered by strava">',
        unsafe_allow_html=True,
    )


def authorization_url():
    request = httpx.Request(
        method="GET",
        url=STRAVA_AUTHORIZATION_URL,
        params={
            "client_id": STRAVA_CLIENT_ID,
            "redirect_uri": APP_URL,
            "response_type": "code",
            "approval_prompt": "auto",
            "scope": "activity:read_all",
        },
    )

    return request.url


def login_header(header=None):
    strava_authorization_url = authorization_url()
    if header is None:
        base = st
    else:
        button = header
        base = button
    current_directory = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_directory, 'images', 'strava.png')
    base64_image = load_image_as_base64(image_path)

    base.markdown(
        (
            f"<a href=\"{strava_authorization_url}\">"
            f"  <img alt=\"strava login\" src=\"data:image/png;base64,{base64_image}\" width=\"20%\">"
            f"</a>"
        ),
        unsafe_allow_html=True,
    )


def logout_header(header=None):
    if header is None:
        base = st
    else:
        button = header
        base = button

    if base.button("Log out"):
        js = f"window.location.href = '{APP_URL}'"
        html = f"<img src onerror=\"{js}\">"
        div = Div(text=html)
        st.bokeh_chart(div)


def logged_in_title(strava_auth, header=None):
    if header is None:
        base = st
    else:
        col = header
        base = col

    first_name = strava_auth["athlete"]["firstname"]
    last_name = strava_auth["athlete"]["lastname"]
    col.markdown(f"*Welcome, {first_name} {last_name}!*")


@st.cache_data
def exchange_authorization_code(authorization_code):
    response = httpx.post(
        url="https://www.strava.com/oauth/token",
        json={
            "client_id": STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "code": authorization_code,
            "grant_type": "authorization_code",
        },
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        st.error("Something went wrong while authenticating with Strava. Please reload and try again")
        st.experimental_set_query_params()
        st.stop()
        return

    strava_auth = response.json()

    return strava_auth


def authenticate(header=None, stop_if_unauthenticated=True):
    query_params = st.experimental_get_query_params()
    authorization_code = query_params.get("code", [None])[0]

    if authorization_code is None:
        authorization_code = query_params.get("session", [None])[0]

    if authorization_code is None:
        login_header(header=header)
        if stop_if_unauthenticated:
            st.stop()
        return
    else:
        logout_header(header=header)
        strava_auth = exchange_authorization_code(authorization_code)
        logged_in_title(strava_auth, header)
        st.experimental_set_query_params(session=authorization_code)

        return strava_auth


def header():
    strava_button = st.empty()

    return strava_button


@st.cache_data
def get_activities(auth, page=1):
    access_token = auth["access_token"]
    response = httpx.get(
        url=f"{STRAVA_API_BASE_URL}/athlete/activities",
        params={
            "page": page,
        },
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )

    return response.json()


def activity_label(activity):
    if activity["name"] == DEFAULT_ACTIVITY_LABEL:
        return ""

    start_date = arrow.get(activity["start_date_local"])
    human_readable_date = start_date.humanize(granularity=["day"])
    date_string = start_date.format("YYYY-MM-DD")

    return f"{activity['name']} - {date_string} ({human_readable_date})"


def select_strava_activity(auth):
    col1, col2 = st.columns([1, 3])
    with col1:
        page = st.number_input(
            label="Activities page",
            min_value=1,
            help="The Strava API returns your activities in chunks of 30. Increment this field to go to the next page.",
        )

    with col2:
        activities = get_activities(auth=auth, page=page)
        if not activities:
            st.info("This Strava account has no activities or you ran out of pages.")
            st.stop()
        default_activity = {"name": DEFAULT_ACTIVITY_LABEL, "start_date_local": ""}

        activity = st.selectbox(
            label="Select an activity",
            options=[default_activity] + activities,
            format_func=activity_label,
        )

    if activity["name"] == DEFAULT_ACTIVITY_LABEL:
        st.write("No activity selected")
        st.stop()
        return

    activity_url = f"https://www.strava.com/activities/{activity['id']}"

    st.markdown(
        f"<a href=\"{activity_url}\" style=\"color:{STRAVA_ORANGE};\">View on Strava</a>", unsafe_allow_html=True
    )

    return activity


def dataframe_from_strava(auth, page=1):
    activities = get_activities(auth, page)

    # Initialize empty lists for each column
    dates = []
    names = []
    sport_types = []
    distances = []
    moving_times = []
    elapsed_times = []
    total_elevation_gains = []
    average_speeds = []
    max_speeds = []
    average_cadences = []
    average_watts_list = []
    average_heartrates = []
    max_heartrates = []
    elev_highs = []
    elev_lows = []
    suffer_scores = []

    for activity in activities:
        dates.append(activity.get("start_date_local", None))
        names.append(activity.get("name", None))
        sport_types.append(activity.get("type", None))
        distances.append(activity.get("distance", None))
        moving_times.append(activity.get("moving_time", None))
        elapsed_times.append(activity.get("elapsed_time", None))
        total_elevation_gains.append(activity.get("total_elevation_gain", None))
        average_speeds.append(activity.get("average_speed", None))
        max_speeds.append(activity.get("max_speed", None))
        average_cadences.append(activity.get("average_cadence", None))
        average_watts_list.append(activity.get("average_watts", None))
        average_heartrates.append(activity.get("average_heartrate", None))
        max_heartrates.append(activity.get("max_heartrate", None))
        elev_highs.append(activity.get("elev_high", None))
        elev_lows.append(activity.get("elev_low", None))
        suffer_scores.append(activity.get("suffer_score", None))

    # Create a dictionary to convert to a DataFrame
    data = {
        "date": dates,
        "name": names,
        "type": sport_types,
        "distance_meters": distances,
        "moving_time_seconds": moving_times,
        "elapsed_time seconds": elapsed_times,
        "total_elevation_gain": total_elevation_gains,
        "average_speed_metres_per_second": average_speeds,
        "max_speed_metres_per_second": max_speeds,
        "average_cadence": average_cadences,
        "average_watts": average_watts_list,
        "average_heartrate": average_heartrates,
        "max_heartrate": max_heartrates,
        "elev_high_meters": elev_highs,
        "elev_low_meters": elev_lows,
        "suffer_score": suffer_scores,
    }

    # Create a DataFrame
    return pd.DataFrame(data)


def speed_to_pace(speed):
    if speed == 0:
        return "inf"

    # Convert speed to seconds per meter
    seconds_per_meter = 1 / speed

    # Convert to seconds per kilometer
    seconds_per_kilometer = seconds_per_meter * 1000

    # Break down into minutes and seconds
    minutes = int(seconds_per_kilometer // 60)
    seconds = int(seconds_per_kilometer % 60)

    return float(f"{minutes}.{seconds:02}")


@st.cache_data
def load_strava_data(data: pd.DataFrame) -> pd.DataFrame:
    """Loads and preprocesses running data."""
    data = data.copy()
    data = data[data['type'] == "Run"]
    data["date"] = pd.to_datetime(data["date"], errors='coerce')
    data = data.dropna(subset=['date'])
    data["month-year"] = data["date"].dt.strftime("%Y-%m")
    data.loc[:, "pace"] = data["average_speed_metres_per_second"].apply(speed_to_pace)
    data = data[data["date"] >= "2023-01-01"]
    return data.sort_values(by="date")
