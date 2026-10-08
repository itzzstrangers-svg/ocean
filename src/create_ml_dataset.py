from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/processed/temperature_samples.csv")
OUTPUT_FILE = Path("data/processed/ml_dataset.csv")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

ml_df = df[
    [
        "year",
        "month",
        "day",
        "depth_m",
        "latitude",
        "longitude",
        "temperature_c",
    ]
].copy()

ml_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
print(f"Rows: {len(ml_df)}")
print(f"Features: {list(ml_df.columns[:-1])}")
print("Target: temperature_c")
