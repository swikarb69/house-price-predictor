import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = r"data/housing.csv"
MODEL_PATH = "model.pkl"
FEATURES_PATH = "features.pkl"
METRICS_PATH = "metrics.json"


def load_and_prepare_data(path: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the dataset, impute missing values, and one-hot encode categoricals."""
    housing_df = pd.read_csv(path)
    housing_df["total_bedrooms"] = housing_df["total_bedrooms"].fillna(
        housing_df["total_bedrooms"].median()
    )
    housing_df = pd.get_dummies(housing_df, columns=["ocean_proximity"])

    X = housing_df.drop("median_house_value", axis=1)
    y = housing_df["median_house_value"]
    return X, y, X.columns.tolist()


def train_model(X_train, y_train) -> RandomForestRegressor:
    """Train a Random Forest with parallel workers for speed."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def main() -> None:
    X, y, feature_names = load_and_prepare_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    rmse = float(np.sqrt(np.mean((y_test - predictions) ** 2)))

    print(f"MAE: {mae:.2f}")
    print(f"R^2: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")

    joblib.dump(model, MODEL_PATH, compress=9)
    joblib.dump(feature_names, FEATURES_PATH)

    with open(METRICS_PATH, "w") as file:
        json.dump(
            {
                "mae": round(mae, 2),
                "r2": round(r2, 4),
                "rmse": round(rmse, 2),
                "n_estimators": model.n_estimators,
                "n_samples_train": int(len(X_train)),
            },
            file,
            indent=2,
        )

    print(f"Model saved to {MODEL_PATH}")
    print(f"Features saved to {FEATURES_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()