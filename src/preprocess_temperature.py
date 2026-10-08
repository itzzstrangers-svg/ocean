from pathlib import Path
import xarray as xr
import pandas as pd

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def prepare_temperature_data():
    # Prefer the expanded deep-ocean dataset.
    expanded_files = list(
        (RAW_DATA_DIR / "expanded").glob("*.nc")
    )

    if expanded_files:
        input_file = expanded_files[0]
    else:
        files = list(RAW_DATA_DIR.glob("*.nc"))

        if not files:
            raise FileNotFoundError(
                "No NetCDF files found in data/raw"
            )

        input_file = files[0]

    output_file = (
        PROCESSED_DATA_DIR / "temperature_samples.csv"
    )

    print(f"Using NetCDF file: {input_file}")

    ds = xr.open_dataset(input_file)

    df = (
        ds["thetao"]
        .to_dataframe()
        .reset_index()
    )

    df = df.rename(
        columns={
            "time": "date",
            "depth": "depth_m",
            "latitude": "latitude",
            "longitude": "longitude",
            "thetao": "temperature_c",
        }
    )

    df = df.dropna()

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print(f"Output: {output_file}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(
        f"Latitude: {df['latitude'].min()} "
        f"to {df['latitude'].max()}"
    )
    print(
        f"Longitude: {df['longitude'].min()} "
        f"to {df['longitude'].max()}"
    )
    print(
        f"Depth: {df['depth_m'].min():.3f} "
        f"to {df['depth_m'].max():.3f} m"
    )


if __name__ == "__main__":
    prepare_temperature_data()
