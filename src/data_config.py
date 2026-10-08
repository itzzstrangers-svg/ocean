from pathlib import Path


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Initial geographic area for development/testing.
# These are deliberately broad and can be changed later.
LAT_MIN = -10.0
LAT_MAX = 10.0
LON_MIN = 60.0
LON_MAX = 100.0

# Initial depth range in metres.
DEPTH_MIN = 0.0
DEPTH_MAX = 1000.0