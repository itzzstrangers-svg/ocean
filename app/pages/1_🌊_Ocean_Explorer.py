from pathlib import Path
import sys

import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# Project root: Oceanembed 2.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add src/ directly to Python's import path
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import predict_temperature, predict_temperature_batch
# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="OceanEmbed 2.0",
    page_icon="🌊",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPANDED_DIR = PROJECT_ROOT / "data" / "raw" / "expanded"

NC_FILES = list(EXPANDED_DIR.glob("*.nc"))

if not NC_FILES:
    st.error("No expanded Copernicus NetCDF file was found.")
    st.stop()

NC_FILE = NC_FILES[0]


# ============================================================
# STYLE
# ============================================================

st.html("""
<style>
.stApp {
    background:
    radial-gradient(circle at 8% 3%, rgba(0,145,190,.18), transparent 28%),
    radial-gradient(circle at 92% 10%, rgba(30,70,160,.16), transparent 28%),
    #06131f;
    color:#edfaff;
}

.block-container {
    max-width:1450px;
    padding-top:1.5rem;
    padding-bottom:3rem;
}

.hero {
    padding:30px 34px;
    border-radius:24px;
    background:linear-gradient(135deg,#0a3448,#071d2d);
    border:1px solid rgba(100,205,235,.18);
    margin-bottom:24px;
}

.eyebrow {
    color:#63c9e9;
    font-size:.74rem;
    font-weight:700;
    letter-spacing:.15em;
}

.hero h1 {
    color:#f2fcff;
    font-size:3rem;
    margin:5px 0;
}

.hero p {
    color:#91adbb;
    font-size:1rem;
    margin:0;
}

.section-title {
    color:#e7faff;
    font-size:1.3rem;
    font-weight:750;
    margin-top:22px;
}

.section-sub {
    color:#7895a4;
    font-size:.84rem;
    margin-bottom:15px;
}

.card {
    background:rgba(7,29,43,.90);
    border:1px solid rgba(100,195,225,.15);
    border-radius:19px;
    padding:20px;
}

.stat {
    background:linear-gradient(145deg,#0b4055,#082638);
    border:1px solid rgba(90,205,235,.20);
    border-radius:16px;
    padding:17px;
    text-align:center;
}

.stat-label {
    color:#82a5b5;
    font-size:.70rem;
    text-transform:uppercase;
    letter-spacing:.09em;
}

.stat-value {
    color:#effcff;
    font-size:1.8rem;
    font-weight:800;
    margin-top:4px;
}

.stat-unit {
    color:#61c9e9;
    font-size:.75rem;
}

.prediction {
    background:linear-gradient(135deg,#0b5065,#073448);
    border:1px solid rgba(100,220,245,.25);
    border-radius:20px;
    padding:24px;
    text-align:center;
}

.prediction-label {
    color:#91c3d2;
    font-size:.78rem;
    text-transform:uppercase;
    letter-spacing:.12em;
}

.prediction-value {
    color:#f3fdff;
    font-size:2.8rem;
    font-weight:850;
}

.prediction-unit {
    color:#67d2ed;
}

.note {
    color:#8ca8b5;
    font-size:.79rem;
    line-height:1.6;
}

.stButton > button {
    width:100%;
    min-height:45px;
    border-radius:12px;
    background:linear-gradient(90deg,#087e9f,#075b88);
    color:white;
    border:1px solid rgba(110,215,240,.35);
    font-weight:750;
}
</style>
""")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_ocean_dataset():
    return xr.open_dataset(NC_FILE)


ds = load_ocean_dataset()


# ============================================================
# DATA COVERAGE
# ============================================================

DATA_LAT_MIN = float(ds.latitude.min())
DATA_LAT_MAX = float(ds.latitude.max())

DATA_LON_MIN = float(ds.longitude.min())
DATA_LON_MAX = float(ds.longitude.max())

DATA_DEPTH_MIN = float(ds.depth.min())
DATA_DEPTH_MAX = float(ds.depth.max())

DATA_TIMES = pd.to_datetime(ds.time.values)

DATA_DATE_MIN = DATA_TIMES.min().date()
DATA_DATE_MAX = DATA_TIMES.max().date()


# ============================================================
# HEADER
# ============================================================

st.html("""
<div class="hero">
    <div class="eyebrow">
        OCEAN INTELLIGENCE / COPERNICUS MARINE / MACHINE LEARNING
    </div>

    <h1>
        OceanEmbed 2.0
    </h1>

    <p>
        Explore ocean temperature using real Copernicus observations
        and machine-learning predictions.
    </p>
</div>
""")


# ============================================================
# LOCATION
# ============================================================

st.html("""
<div class="section-title">1. Select location</div>
<div class="section-sub">
    Enter any valid geographic coordinate.
</div>
""")

c1, c2, c3 = st.columns([1, 1, .65])

with c1:
    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=18.5204,
        step=0.0001,
        format="%.4f",
    )

with c2:
    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=73.8567,
        step=0.0001,
        format="%.4f",
    )

with c3:
    st.markdown("<br>", unsafe_allow_html=True)
    locate = st.button("LOCATE", type="primary")


if "lat" not in st.session_state:
    st.session_state.lat = latitude

if "lon" not in st.session_state:
    st.session_state.lon = longitude

if locate:
    st.session_state.lat = latitude
    st.session_state.lon = longitude

selected_lat = st.session_state.lat
selected_lon = st.session_state.lon


# ============================================================
# GLOBAL MAP
# ============================================================

st.html("""
<div class="section-title">2. Location map</div>
<div class="section-sub">
    Selected geographic location.
</div>
""")

fig_map = go.Figure()

fig_map.add_trace(
    go.Scattergeo(
        lon=[selected_lon],
        lat=[selected_lat],
        mode="markers",
        marker=dict(
            size=17,
            color="#ff5968",
            line=dict(color="white", width=2),
        ),
        name="Selected location",
        text=[
            f"Latitude: {selected_lat:.4f}<br>"
            f"Longitude: {selected_lon:.4f}"
        ],
        hovertemplate="%{text}<extra></extra>",
    )
)

fig_map.update_geos(
    projection_type="natural earth",
    showland=True,
    landcolor="#19313d",
    showocean=True,
    oceancolor="#061c2c",
    showlakes=True,
    lakecolor="#08283a",
    showcountries=True,
    countrycolor="#45606d",
    showcoastlines=True,
    coastlinecolor="#7898a5",
    showframe=False,
)

fig_map.update_layout(
    height=520,
    margin=dict(l=0, r=0, t=5, b=5),
    paper_bgcolor="rgba(0,0,0,0)",
    geo=dict(bgcolor="rgba(0,0,0,0)"),
)

st.plotly_chart(fig_map, use_container_width=True)


# ============================================================
# DATA AVAILABILITY
# ============================================================

inside_data_region = (
    DATA_LAT_MIN <= selected_lat <= DATA_LAT_MAX
    and DATA_LON_MIN <= selected_lon <= DATA_LON_MAX
)

st.html("""
<div class="section-title">3. Ocean data availability</div>
""")

if inside_data_region:

    st.success(
        "Copernicus ocean data is available for this location "
        "in the current development dataset."
    )

else:

    st.warning(
        "This location is outside the currently downloaded "
        "Copernicus development region."
    )

    st.html(f"""
    <div class="card">
        <div class="note">

            <b>Current downloaded region</b><br><br>

            Latitude:
            <b>{DATA_LAT_MIN:.2f} to {DATA_LAT_MAX:.2f}</b><br>

            Longitude:
            <b>{DATA_LON_MIN:.2f} to {DATA_LON_MAX:.2f}</b><br>

            Depth:
            <b>{DATA_DEPTH_MIN:.2f} to {DATA_DEPTH_MAX:.2f} m</b><br>

            Dates:
            <b>{DATA_DATE_MIN} to {DATA_DATE_MAX}</b>

            <br><br>

            The next data-engineering stage is to download
            a larger geographic/depth region.

        </div>
    </div>
    """)

    st.stop()


# ============================================================
# OCEAN CONTROLS
# ============================================================

st.html("""
<div class="section-title">4. Ocean analysis</div>
<div class="section-sub">
    Select the date and depth for the analysis.
</div>
""")

o1, o2 = st.columns(2)

with o1:
    selected_date = st.date_input(
        "Observation date",
        value=DATA_DATE_MIN,
        min_value=DATA_DATE_MIN,
        max_value=DATA_DATE_MAX,
    )

with o2:
    selected_depth = st.number_input(
        "Prediction depth (m)",
        min_value=DATA_DEPTH_MIN,
        max_value=DATA_DEPTH_MAX,
        value=min(10.0, DATA_DEPTH_MAX),
        step=1.0,
        format="%.2f",
    )


# ============================================================
# REAL COPERNICUS PROFILE
# ============================================================

selected_timestamp = pd.Timestamp(selected_date)

profile = (
    ds["thetao"]
    .sel(
        time=selected_timestamp,
        latitude=selected_lat,
        longitude=selected_lon,
        method="nearest",
    )
    .to_dataframe()
    .reset_index()
)

profile = profile[["depth", "thetao"]].dropna()

profile.columns = [
    "Depth (m)",
    "Temperature (C)",
]

if profile.empty:
    st.error("No temperature profile is available for this selection.")
    st.stop()


# ============================================================
# REAL OBSERVATION
# ============================================================

nearest_index = np.abs(
    profile["Depth (m)"] - selected_depth
).argmin()

nearest_row = profile.iloc[nearest_index]

observed_depth = float(nearest_row["Depth (m)"])
observed_temperature = float(nearest_row["Temperature (C)"])


# ============================================================
# OPTIONAL ML PREDICTION
# ============================================================

prediction = None
difference = None

try:
    prediction = predict_temperature(
        year=selected_date.year,
        month=selected_date.month,
        day=selected_date.day,
        depth_m=selected_depth,
        latitude=selected_lat,
        longitude=selected_lon,
    )

    difference = prediction - observed_temperature

except (FileNotFoundError, OSError):
    prediction = None
    difference = None


# ============================================================
# RESULTS
# ============================================================

st.html("""
<div class="section-title">5. Ocean temperature result</div>
""")

if prediction is None:
    st.info(
        "The Copernicus observation is available. "
        "The trained machine-learning model is not included "
        "in this public deployment, so AI prediction is "
        "currently unavailable."
    )

prediction_display = (
    f"{prediction:.2f}"
    if prediction is not None
    else "Unavailable"
)

difference_display = (
    f"{difference:+.3f}"
    if difference is not None
    else "Unavailable"
)

r1, r2, r3, r4 = st.columns(4)

with r1:
    st.html(f"""
    <div class="prediction">
        <div class="prediction-label">ML prediction</div>
        <div class="prediction-value">{prediction_display}</div>
        <div class="prediction-unit">degrees C</div>
    </div>
    """)

with r2:
    st.html(f"""
    <div class="stat">
        <div class="stat-label">Real Copernicus</div>
        <div class="stat-value">{observed_temperature:.2f}</div>
        <div class="stat-unit">degrees C</div>
    </div>
    """)

with r3:
    st.html(f"""
    <div class="stat">
        <div class="stat-label">Observed depth</div>
        <div class="stat-value">{observed_depth:.2f}</div>
        <div class="stat-unit">metres</div>
    </div>
    """)

with r4:
    st.html(f"""
    <div class="stat">
        <div class="stat-label">Difference</div>
        <div class="stat-value">{difference_display}</div>
        <div class="stat-unit">degrees C</div>
    </div>
    """)


# ============================================================
# GRAPH
# ============================================================

st.html("""
<div class="section-title">6. Temperature vs depth</div>
""")

fig_profile = go.Figure()

fig_profile.add_trace(
    go.Scatter(
        x=profile["Temperature (C)"],
        y=profile["Depth (m)"],
        mode="lines+markers",
        name="Copernicus",
        line=dict(color="#35b9e2", width=3),
        marker=dict(size=7),
        hovertemplate=(
            "Temperature: %{x:.3f} C"
            "<br>Depth: %{y:.2f} m"
            "<extra>Copernicus</extra>"
        ),
    )
)

if prediction is not None:
    fig_profile.add_trace(
        go.Scatter(
            x=[prediction],
            y=[selected_depth],
            mode="markers",
            name="OceanEmbed prediction",
            marker=dict(
                size=16,
                color="#ff5968",
                line=dict(color="white", width=2),
            ),
            hovertemplate=(
                "Prediction: %{x:.3f} C"
                "<br>Depth: %{y:.2f} m"
                "<extra>OceanEmbed</extra>"
            ),
        )
    )

fig_profile.update_layout(
    height=560,
    margin=dict(l=55, r=25, t=20, b=55),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(5,25,38,.75)",
    font=dict(color="#b9d1dc"),
    xaxis=dict(
        title="Temperature (degrees C)",
        gridcolor="rgba(130,190,210,.13)",
    ),
    yaxis=dict(
        title="Depth (m)",
        autorange="reversed",
        gridcolor="rgba(130,190,210,.13)",
    ),
)

st.plotly_chart(fig_profile, use_container_width=True)


# ============================================================
# TABLE
# ============================================================

st.html("""
<div class="section-title">7. Temperature / depth table</div>
""")

table_df = profile.copy()

predictions = None

try:
    predictions = predict_temperature_batch(
        year=selected_date.year,
        month=selected_date.month,
        day=selected_date.day,
        depths=table_df["Depth (m)"].tolist(),
        latitude=selected_lat,
        longitude=selected_lon,
    )
except (FileNotFoundError, OSError):
    predictions = None

if predictions is not None:

    table_df["OceanEmbed Prediction (C)"] = predictions

    table_df["Difference (C)"] = (
        table_df["OceanEmbed Prediction (C)"]
        - table_df["Temperature (C)"]
    ).round(3)

    table_df["OceanEmbed Prediction (C)"] = (
        table_df["OceanEmbed Prediction (C)"].round(3)
    )

else:

    table_df["OceanEmbed Prediction (C)"] = "Unavailable"
    table_df["Difference (C)"] = "Unavailable"

table_df["Depth (m)"] = table_df["Depth (m)"].round(3)
table_df["Temperature (C)"] = table_df["Temperature (C)"].round(3)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
    height=700,
)


# ============================================================
# DATA SUMMARY
# ============================================================

st.html("""
<div class="section-title">8. Current development dataset</div>
""")

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric("Temperature points", f"{len(profile):,}")

with s2:
    st.metric(
        "Depth range",
        f"{profile['Depth (m)'].min():.2f} - "
        f"{profile['Depth (m)'].max():.2f} m",
    )

with s3:
    st.metric("Latitude", f"{selected_lat:.4f}")

with s4:
    st.metric("Longitude", f"{selected_lon:.4f}")


# ============================================================
# SCIENTIFIC NOTE
# ============================================================

st.html("""
<div class="card">
    <div class="note">

        <b>Scientific status:</b><br><br>

        The blue temperature profile is real Copernicus
        Marine data.

        The red marker is shown only when the OceanEmbed
        Random Forest model is available.

        <br><br>

        The present development model was trained on a
        development dataset and is not yet a global
        deep-ocean prediction system.

        <br><br>

        The next scientific expansion is to download deeper
        Copernicus observations and retrain the model across
        a substantially larger ocean region.

    </div>
</div>
""")