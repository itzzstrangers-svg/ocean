from pathlib import Path
import sys

import streamlit as st
import pandas as pd
import plotly.express as px


# --------------------------------------------------
# PROJECT
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="OceanEmbed | Data Explorer",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
CSV_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temperature_samples.csv"
)

if not CSV_FILE.exists():
    st.error(f"Dataset not found: {CSV_FILE}")
    st.stop()


@st.cache_data
def load_data():
    return pd.read_csv(CSV_FILE)


df = load_data()


# --------------------------------------------------
# FIND TEMPERATURE COLUMN
# --------------------------------------------------
temperature_candidates = [
    "temperature_c",
    "temperature",
    "temp",
    "temp_C",
    "thetao",
    "sea_water_potential_temperature",
]

temperature_column = None

for column in temperature_candidates:
    if column in df.columns:
        temperature_column = column
        break

if temperature_column is None:
    st.error(
        "Could not find the temperature column in the dataset."
    )
    st.write("Columns found in the CSV:")
    st.write(list(df.columns))
    st.stop()


# --------------------------------------------------
# FIND DEPTH COLUMN
# --------------------------------------------------
if "depth_m" in df.columns:
    depth_column = "depth_m"
elif "depth" in df.columns:
    depth_column = "depth"
else:
    st.error("Could not find the depth column.")
    st.write(list(df.columns))
    st.stop()


# --------------------------------------------------
# PAGE
# --------------------------------------------------
st.title("📊 Data Explorer")

st.write(
    "Explore the ocean temperature observations used by "
    "OceanEmbed 2.0."
)

st.success(
    f"Loaded {len(df):,} ocean observations."
)


# --------------------------------------------------
# DATASET INFO
# --------------------------------------------------
with st.expander("🔎 Dataset columns"):
    st.write(list(df.columns))

    st.write(
        f"Temperature column: `{temperature_column}`"
    )

    st.write(
        f"Depth column: `{depth_column}`"
    )


# --------------------------------------------------
# FILTERS
# --------------------------------------------------
st.header("🔎 Filter Data")

c1, c2, c3 = st.columns(3)

with c1:
    latitude_range = st.slider(
        "Latitude",
        float(df["latitude"].min()),
        float(df["latitude"].max()),
        (
            float(df["latitude"].min()),
            float(df["latitude"].max()),
        ),
    )

with c2:
    longitude_range = st.slider(
        "Longitude",
        float(df["longitude"].min()),
        float(df["longitude"].max()),
        (
            float(df["longitude"].min()),
            float(df["longitude"].max()),
        ),
    )

with c3:
    depth_range = st.slider(
        "Depth (m)",
        float(df[depth_column].min()),
        float(df[depth_column].max()),
        (
            float(df[depth_column].min()),
            float(df[depth_column].max()),
        ),
    )


filtered = df[
    df["latitude"].between(*latitude_range)
    & df["longitude"].between(*longitude_range)
    & df[depth_column].between(*depth_range)
]


st.metric(
    "Filtered observations",
    f"{len(filtered):,}"
)


# --------------------------------------------------
# TEMPERATURE SUMMARY
# --------------------------------------------------
st.header("🌡️ Temperature Summary")

a, b, c, d = st.columns(4)

a.metric(
    "Minimum",
    f"{filtered[temperature_column].min():.2f} °C",
)

b.metric(
    "Maximum",
    f"{filtered[temperature_column].max():.2f} °C",
)

c.metric(
    "Average",
    f"{filtered[temperature_column].mean():.2f} °C",
)

d.metric(
    "Median",
    f"{filtered[temperature_column].median():.2f} °C",
)


# --------------------------------------------------
# DISTRIBUTION
# --------------------------------------------------
st.header("📈 Temperature Distribution")

fig = px.histogram(
    filtered,
    x=temperature_column,
    nbins=50,
    title="Ocean Temperature Distribution",
)

fig.update_layout(
    xaxis_title="Temperature (°C)",
    yaxis_title="Number of Observations",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# --------------------------------------------------
# TEMPERATURE VS DEPTH
# --------------------------------------------------
st.header("🌊 Temperature vs Depth")

profile = (
    filtered
    .groupby(depth_column, as_index=False)[temperature_column]
    .mean()
)

fig2 = px.line(
    profile,
    x=temperature_column,
    y=depth_column,
    markers=True,
    title="Average Ocean Temperature Profile",
)

fig2.update_yaxes(
    autorange="reversed"
)

fig2.update_layout(
    xaxis_title="Temperature (°C)",
    yaxis_title="Depth (m)",
)

st.plotly_chart(
    fig2,
    use_container_width=True,
)


# --------------------------------------------------
# DATA TABLE
# --------------------------------------------------
st.header("🗃️ Ocean Observations")

st.dataframe(
    filtered.head(5000),
    use_container_width=True,
    hide_index=True,
    height=600,
)


# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------
st.header("⬇️ Download")

csv_data = filtered.to_csv(index=False)

st.download_button(
    label="Download Filtered CSV",
    data=csv_data,
    file_name="oceanembed_filtered_data.csv",
)
   