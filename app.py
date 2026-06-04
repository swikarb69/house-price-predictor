import streamlit as st
import pandas as pd
import pickle

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# =========================
# Load Model
# =========================

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("features.pkl", "rb") as file:
    features = pickle.load(file)

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

st.sidebar.metric(
    label="R² Score",
    value="0.77"
)

st.sidebar.metric(
    label="MAE",
    value="$37,507"
)

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

st.title("🏠 California House Price Predictor -Swikarb69")

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

col1, col2 = st.columns(2)

with col1:

    median_income = st.number_input(
        "Median Income",
        min_value=0.0,
        value=8.3252
    )

    housing_median_age = st.number_input(
        "Housing Median Age",
        min_value=1,
        value=41
    )

    total_rooms = st.number_input(
        "Total Rooms",
        min_value=1,
        value=880
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        min_value=1,
        value=129
    )

with col2:

    population = st.number_input(
        "Population",
        min_value=1,
        value=322
    )

    households = st.number_input(
        "Households",
        min_value=1,
        value=126
    )

    latitude = st.number_input(
        "Latitude",
        value=37.88
    )

    longitude = st.number_input(
        "Longitude",
        value=-122.23
    )

# =========================
# Ocean Proximity
# =========================

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    [
        "<1H OCEAN",
        "INLAND",
        "NEAR OCEAN",
        "NEAR BAY",
        "ISLAND"
    ]
)

# =========================
# Prediction
# =========================

if st.button("🔮 Predict Price"):

    input_data = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income
    }

    # Initialize ocean features
    for feature in features:
        if feature.startswith("ocean_proximity_"):
            input_data[feature] = 0

    selected_feature = f"ocean_proximity_{ocean_proximity}"

    if selected_feature in features:
        input_data[selected_feature] = 1

    input_df = pd.DataFrame([input_data])

    input_df = input_df.reindex(
        columns=features,
        fill_value=0
    )

    prediction = model.predict(input_df)[0]

    st.markdown("---")

    st.metric(
        label="🏡 Estimated House Price",
        value=f"${prediction:,.0f}"
    )

    st.success(
        "Prediction generated successfully!"
    )

# =========================
# Footer
# =========================

st.markdown("---")

st.caption(
    "Built with Swikarb69 using Streamlit, Pandas, and Scikit-Learn"
)