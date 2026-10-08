from sklearn.ensemble import RandomForestRegressor


def create_model() -> RandomForestRegressor:
    """Create the OceanEmbed 2.0 baseline model."""
    return RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )