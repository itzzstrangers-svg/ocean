from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def prepare_csv(input_filename: str, output_filename: str) -> None:
    """Read a raw CSV and save a cleaned copy."""
    input_path = RAW_DATA_DIR / input_filename
    output_path = PROCESSED_DATA_DIR / output_filename

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved processed data: {output_path}")