from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np

DATA_FILE = Path("data/processed/ml_dataset.csv")
MODEL_DIR = Path("models/trained")
MODEL_FILE = MODEL_DIR / "temperature_model.joblib"

df = pd.read_csv(DATA_FILE)

features = [
    "year",
    "month",
    "day",
    "depth_m",
    "latitude",
    "longitude",
]

X = df[features]
y = df["temperature_c"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_FILE)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"MAE: {mae:.4f} C")
print(f"RMSE: {rmse:.4f} C")
print(f"Model saved: {MODEL_FILE}")
