import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import pickle

# Loading the dataset from 'data/' directory
housing_df = pd.read_csv(r"data/housing.csv")

# Filling the missing values (NaNs) in 'total_bedrooms'
housing_df['total_bedrooms'] = housing_df['total_bedrooms'].fillna(housing_df['total_bedrooms'].median())

# Converting 'ocean_proximity' into numeric values
housing_df = pd.get_dummies(housing_df, columns = ['ocean_proximity'])

# Separating Features and Target
X = housing_df.drop('median_house_value', axis = 1)
y = housing_df['median_house_value']

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size = 0.2,
    random_state = 42
)

# Train model
model = RandomForestRegressor(
    n_estimators = 100,
    random_state = 42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Metrices
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.2f}")
print(f"R_squared: {r2:.2f}")

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save features name
with open("features.pkl", "wb") as file:
    pickle.dump(X.columns.tolist(), file)

print("Model saved successfully!")