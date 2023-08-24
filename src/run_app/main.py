import streamlit as st
import strava
import pandas as pd

st.set_page_config(
    page_title="Streamlit Activity Viewer for Strava",
    page_icon=":circus_tent:",
)

strava_header = strava.header()

strava_auth = strava.authenticate(header=strava_header, stop_if_unauthenticated=False)

if strava_auth is None:
    st.stop()


activity = strava.select_strava_activity(strava_auth)


activity_data = {
    "name": [activity["name"]],
    "distance": [activity["distance"]],
    "moving_time": [activity["moving_time"]],
    "elapsed_time": [activity["elapsed_time"]],
    "total_elevation_gain": [activity["total_elevation_gain"]],
    "average_speed": [activity["average_speed"]],
    "max_speed": [activity["max_speed"]],
    "average_cadence": [activity["average_cadence"]],
    "average_watts": [activity["average_watts"]],
    "average_heartrate": [activity["average_heartrate"]],
    "max_heartrate": [activity["max_heartrate"]],
    "elev_high": [activity["elev_high"]],
    "elev_low": [activity["elev_low"]],
    "suffer_score": [activity["suffer_score"]],
}


st.write(activity)
st.write(pd.DataFrame.from_dict(activity_data))


df = strava.dataframe_from_strava(strava_auth)
st.write(strava.load_strava_data(df))
