from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from the raw data folder."""
    file_path = RAW_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)