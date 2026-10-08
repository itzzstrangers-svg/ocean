from pathlib import Path


RAW_DATA_DIR = Path("data/raw")


def prepare_raw_data_folder() -> Path:
    """Create the raw data folder if it does not exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DATA_DIR


if __name__ == "__main__":
    folder = prepare_raw_data_folder()
    print(f"Raw data folder ready: {folder}")