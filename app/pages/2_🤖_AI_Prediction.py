import sys
from pathlib import Path
from datetime import date

import streamlit as st
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_temperature

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="OceanEmbed 2.0 | AI Prediction",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------
# STYLING
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(0, 150, 255, 0.16), transparent 30%),
            radial-gradient(circle at 90% 20%, rgba(0, 220, 200, 0.10), transparent 30%),
            linear-gradient(135deg, #03111f 0%, #061d31 45%, #02101c 100%);
        color: #f5f9ff;
    }

    .hero {
        padding: 2.5rem 2rem;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            rgba(9, 72, 112, 0.75),
            rgba(4, 34, 58, 0.85)
        );
        border: 1px solid rgba(120, 210, 255, 0.20);
        margin-bottom: 1.5rem;
        box-shadow: 0 15px 45px rgba(0,0,0,0.25);
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1.15rem;
        color: #b9d9ed;
        max-width: 800px;
    }

    .result-card {
        padding: 2rem;
        border-radius: 22px;
        background: rgba(7, 43, 67, 0.80);
        border: 1px solid rgba(100, 200, 255, 0.20);
        text-align: center;
        margin-top: 1rem;
    }

    .prediction {
        font-size: 3.2rem;
        font-weight: 800;
        color: #72d8ff;
    }

    .label {
        color: #9fc3d9;
        font-size: 1rem;
    }

    .info-card {
        padding: 1.3rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
NC_FILES = list((PROJECT_ROOT / "data" / "raw" / "expanded").glob("*.nc"))

if not NC_FILES:
    st.error("No expanded Copernicus NetCDF dataset was found.")
    st.stop()

NC_FILE = NC_FILES[0]


@st.cache_resource
def load_dataset():
    return xr.open_dataset(NC_FILE)


ds = load_dataset()


# ---------------------------------------------------------
# DATA LIMITS
# ---------------------------------------------------------
lat_min = float(ds.latitude.min())
lat_max = float(ds.latitude.max())

lon_min = float(ds.longitude.min())
lon_max = float(ds.longitude.max())

depth_min = float(ds.depth.min())
depth_max = float(ds.depth.max())

time_values = ds.time.values
date_min = time_values[0].astype("datetime64[D]").astype(date)
date_max = time_values[-1].astype("datetime64[D]").astype(date)


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🤖 AI Ocean Temperature Prediction</h1>
        <p>
            Use the OceanEmbed machine-learning model to estimate
            ocean temperature at a selected location, date, and depth.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="info-card">
            <b>🧠 Model</b><br>
            Random Forest Regression
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <b>🌊 Target</b><br>
            Ocean potential temperature
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="info-card">
            <b>📐 Inputs</b><br>
            Date · Depth · Latitude · Longitude
        </div>
        """,
        unsafe_allow_html=True,
    )


st.subheader("Choose Prediction Conditions")


# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------
left, right = st.columns(2)

with left:
    selected_date = st.date_input(
        "📅 Date",
        value=date_min,
        min_value=date_min,
        max_value=date_max,
    )

    selected_depth = st.number_input(
        "🌊 Depth (meters)",
        min_value=depth_min,
        max_value=depth_max,
        value=min(10.0, depth_max),
        step=1.0,
    )

with right:
    selected_lat = st.number_input(
        "📍 Latitude (°N)",
        min_value=lat_min,
        max_value=lat_max,
        value=(lat_min + lat_max) / 2,
        step=0.1,
    )

    selected_lon = st.number_input(
        "📍 Longitude (°E)",
        min_value=lon_min,
        max_value=lon_max,
        value=(lon_min + lon_max) / 2,
        step=0.1,
    )


st.markdown("---")


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

prediction = None

if st.button("🚀 Predict Ocean Temperature", use_container_width=True):

    with st.spinner("Running OceanEmbed AI model..."):

        try:
            prediction = predict_temperature(
                year=selected_date.year,
                month=selected_date.month,
                day=selected_date.day,
                depth_m=selected_depth,
                latitude=selected_lat,
                longitude=selected_lon,
            )
        except Exception as e:
         st.error(f"Prediction failed: {type(e).__name__}: {e}")


# ---------------------------------------------------------
# COPERNICUS OBSERVATION
# ---------------------------------------------------------

observed_temperature = float(
    ds["thetao"]
    .sel(
        time=selected_date,
        depth=selected_depth,
        latitude=selected_lat,
        longitude=selected_lon,
        method="nearest",
    )
    .values
)


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

if prediction is None:
    prediction_display = "Unavailable"
else:
    prediction_display = f"{prediction:.2f} °C"


st.markdown(
    f"""
    <div class="result-card">
        <div class="label">🤖 OceanEmbed AI Prediction</div>
        <div class="prediction">{prediction_display}</div>
        <div class="label">
            {selected_depth:.1f} m depth ·
            {selected_lat:.2f}°N ·
            {selected_lon:.2f}°E ·
            {selected_date}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    f"""
    <div class="result-card">
        <div class="label">🌊 Copernicus Observation</div>
        <div class="prediction">{observed_temperature:.2f} °C</div>
        <div class="label">
            Real Copernicus ocean temperature ·
            {selected_depth:.1f} m depth ·
            {selected_lat:.2f}°N ·
            {selected_lon:.2f}°E ·
            {selected_date}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# EXPLANATION
# ---------------------------------------------------------
st.markdown("---")

st.subheader("How the AI Prediction Works")

a, b, c = st.columns(3)

with a:
    st.markdown(
        """
        ### 1️⃣ Location
        The model receives the selected latitude and longitude
        to understand where the prediction is being made.
        """
    )

with b:
    st.markdown(
        """
        ### 2️⃣ Depth & Date
        Depth and calendar information provide the vertical
        and temporal context for the prediction.
        """
    )

with c:
    st.markdown(
        """
        ### 3️⃣ Temperature
        The trained regression model estimates the ocean
        temperature for those conditions.
        """
    )


st.info(
    f"""
    **Current training/data coverage**

    Latitude: {lat_min:.1f}°N to {lat_max:.1f}°N  
    Longitude: {lon_min:.1f}°E to {lon_max:.1f}°E  
    Depth: approximately {depth_min:.1f}–{depth_max:.1f} m  
    Dates: {date_min} to {date_max}
    """
)