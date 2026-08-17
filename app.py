import json

import joblib
import pandas as pd
import streamlit as st

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="centered",
)


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_features():
    return joblib.load("features.pkl")


@st.cache_data
def load_metrics():
    try:
        with open("metrics.json") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


model = load_model()
features = load_features()
metrics = load_metrics()

# =========================
# Sidebar
# =========================

st.sidebar.title("🏠 About")

st.sidebar.info(
    """
    This application predicts California house prices
    using a Random Forest Regressor trained on the
    California Housing dataset.
    """
)

st.sidebar.subheader("📊 Model Performance")

st.sidebar.metric(label="R² Score", value=f"{metrics.get('r2', 0.77):.4f}")
st.sidebar.metric(label="MAE", value=f"${metrics.get('mae', 37507):,.0f}")

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Tech Stack**

    - Python
    - Scikit-Learn
    - Pandas
    - Streamlit
    """
)

# =========================
# Main Page
# =========================

st.title("🏠 California House Price Predictor - Swikarb69")

st.markdown(
    """
    Predict California housing prices using machine learning.

    Enter property details below and click **Predict Price**.
    """
)

st.markdown("---")

# =========================
# Input Form
# =========================

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        median_income = st.number_input(
            "Median Income (tens of thousands)",
            min_value=0.5,
            max_value=16.0,
            value=3.5,
            step=0.1,
        )
        housing_median_age = st.number_input(
            "Housing Median Age",
            min_value=1,
            max_value=52,
            value=29,
        )
        total_rooms = st.number_input(
            "Total Rooms",
            min_value=1,
            max_value=40000,
            value=2000,
            step=100,
        )
        total_bedrooms = st.number_input(
            "Total Bedrooms",
            min_value=1,
            max_value=6000,
            value=400,
            step=50,
        )

    with col2:
        population = st.number_input(
            "Population",
            min_value=1,
            max_value=36000,
            value=1400,
            step=100,
        )
        households = st.number_input(
            "Households",
            min_value=1,
            max_value=6000,
            value=500,
            step=50,
        )
        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.0,
            value=35.5,
            step=0.1,
        )
        longitude = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-113.0,
            value=-119.5,
            step=0.1,
        )

    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"],
    )

    submitted = st.form_submit_button("🔮 Predict Price", type="primary")

# =========================
# Prediction
# =========================

if submitted:
    input_data = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
    }

    selected_feature = f"ocean_proximity_{ocean_proximity}"

    for feature in features:
        if feature.startswith("ocean_proximity_"):
            input_data[feature] = 1 if feature == selected_feature else 0

    input_df = pd.DataFrame([input_data]).reindex(columns=features, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.markdown("---")

    st.metric(
        label="🏡 Estimated House Price",
        value=f"${prediction:,.0f}",
    )

    st.success("Prediction generated successfully!")

# =========================
# Footer
# =========================

st.markdown("---")

st.caption("Built with Swikarb69 using Streamlit, Pandas, and Scikit-Learn")