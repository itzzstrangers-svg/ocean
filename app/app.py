from pathlib import Path
import streamlit as st
import xarray as xr

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Ocean",
    page_icon="🌊",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NC_DIR = PROJECT_ROOT / "data" / "raw" / "expanded"
NC_FILES = list(NC_DIR.glob("*.nc"))

if not NC_FILES:
    st.error("No Copernicus dataset found.")
    st.stop()

NC_FILE = NC_FILES[0]


@st.cache_resource
def load_dataset():
    return xr.open_dataset(NC_FILE)


ds = load_dataset()

lat_min = float(ds.latitude.min())
lat_max = float(ds.latitude.max())
lon_min = float(ds.longitude.min())
lon_max = float(ds.longitude.max())
depth_max = float(ds.depth.max())

data_points = 1
for size in ds.sizes.values():
    data_points *= size

start_date = str(ds.time.values[0])[:10]
end_date = str(ds.time.values[-1])[:10]


# --------------------------------------------------
# STYLE
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0,150,220,.18), transparent 30%),
        linear-gradient(135deg, #02101b, #03283c, #02111e);
}

.hero-box {
    padding: 45px;
    border-radius: 25px;
    background: linear-gradient(135deg, #063f5c, #032238);
    border: 1px solid rgba(100,210,255,.3);
    margin-bottom: 30px;
}

.hero-small {
    color: #55d9ff;
    font-size: 14px;
    font-weight: bold;
    letter-spacing: 3px;
}

.hero-title {
    color: white;
    font-size: 48px;
    font-weight: 800;
    margin-top: 15px;
}

.hero-text {
    color: #b9d8e8;
    font-size: 18px;
    line-height: 1.6;
}

.card {
    padding: 25px;
    border-radius: 20px;
    background: rgba(10,65,90,.7);
    border: 1px solid rgba(100,210,255,.2);
    min-height: 180px;
}

.card-title {
    color: white;
    font-size: 22px;
    font-weight: bold;
}

.card-text {
    color: #b8d5e2;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown("""
<div class="hero-box">
    <div class="hero-small">
        OCEAN INTELLIGENCE · COPERNICUS MARINE · MACHINE LEARNING
    </div>
    <div class="hero-title">
        Explore the Ocean.<br>
        Understand the Deep.
    </div>
    <div class="hero-text">
        OceanEmbed 2.0 combines real ocean observations
        with machine learning to explore temperature
        across latitude, longitude, depth and time.
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATASET
# --------------------------------------------------
st.header("🌊 Current OceanEmbed Dataset")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Latitude Coverage",
    f"{lat_min:.0f}–{lat_max:.0f}°N"
)

c2.metric(
    "Longitude Coverage",
    f"{lon_min:.0f}–{lon_max:.0f}°E"
)

c3.metric(
    "Maximum Depth",
    f"{depth_max:.0f} m"
)

c4.metric(
    "Ocean Data Points",
    f"{data_points:,}"
)


# --------------------------------------------------
# FEATURES
# --------------------------------------------------
st.header("🚀 Explore OceanEmbed")

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("""
    <div class="card">
        <div class="card-title">🌊 Ocean Explorer</div>
        <div class="card-text">
            Explore ocean locations on an interactive map
            and compare real Copernicus observations with
            OceanEmbed predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)

with f2:
    st.markdown("""
    <div class="card">
        <div class="card-title">🤖 AI Prediction</div>
        <div class="card-text">
            Select a date, location and depth to generate
            an ocean temperature prediction using the
            trained machine-learning model.
        </div>
    </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("""
    <div class="card">
        <div class="card-title">📊 Data Explorer</div>
        <div class="card-text">
            Inspect temperature profiles, visualize ocean
            data and explore the OceanEmbed dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------
st.header("🔬 From Ocean Observations to Machine Learning")

st.info(
    """
    **01 — Observe**

    Copernicus Marine provides ocean temperature observations.

    **02 — Prepare**

    The observations are transformed into structured
    latitude, longitude, depth and time features.

    **03 — Learn**

    OceanEmbed trains a machine-learning regression model
    using those observations.

    **04 — Explore**

    The application compares real observations with
    OceanEmbed predictions at selected locations and depths.
    """
)


# --------------------------------------------------
# COVERAGE
# --------------------------------------------------
st.header("📍 Dataset Coverage")

a, b = st.columns(2)

with a:
    st.success(
        f"""
        **Geographic Coverage**

        Latitude: **{lat_min:.1f}°N – {lat_max:.1f}°N**

        Longitude: **{lon_min:.1f}°E – {lon_max:.1f}°E**
        """
    )

with b:
    st.success(
        f"""
        **Ocean Coverage**

        Dates: **{start_date} → {end_date}**

        Maximum depth: **{depth_max:.1f} m**
        """
    )


st.divider()

st.caption(
    "🌊 Ocean · Ocean observations · Machine learning · Ocean exploration"
)
