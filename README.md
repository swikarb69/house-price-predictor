# 🏠 California House Price Predictor

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)]()

## 📌 Overview

California House Price Predictor is an end-to-end Machine Learning web application that estimates residential property values based on housing characteristics and geographical information.

The application uses a **Random Forest Regressor** trained on the California Housing dataset and provides real-time predictions through an interactive Streamlit interface.

---

## 🚀 Live Demo

🔗 https://house-price-predictor-swikarb69.streamlit.app

---

## 📊 Model Performance

| Metric                    | Score   |
| ------------------------- | ------- |
| R² Score                  | 0.77    |
| Mean Absolute Error (MAE) | $37,507 |

The model explains approximately **77% of the variance** in housing prices while maintaining a reasonable prediction error.

---

## ✨ Features

* Interactive web interface
* Real-time house price prediction
* Ocean proximity selection
* Random Forest Regression model
* Responsive Streamlit dashboard
* Clean and user-friendly UI
* Model performance metrics display

---

## 🛠️ Tech Stack

### Machine Learning

* Scikit-Learn
* Pandas
* NumPy

### Web Application

* Streamlit

### Development

* Python 3.11
* Git
* GitHub

---

## 📂 Project Structure

```text
House-Price-Predictor/
└── data/
    └── housing.csv
│
├── README.md
├── app.py
├── features.pkl
├── model.pkl
├── requirements.txt
├── train.py

```

---

## 📈 Dataset

This project uses the California Housing dataset containing:

* Longitude
* Latitude
* Housing Median Age
* Total Rooms
* Total Bedrooms
* Population
* Households
* Median Income
* Ocean Proximity
* Median House Value (Target)

Dataset Source:
https://www.kaggle.com/datasets/camnugent/california-housing-prices

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/swikarb69/house-price-predictor.git

cd house-price-predictor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train.py
```

### Run Application

```bash
streamlit run app.py
```

---

## 🖥️ Example Prediction

Input:

* Median Income: 8.32
* House Age: 41
* Total Rooms: 880
* Population: 322
* Ocean Proximity: Near Bay

Output:

```text
Estimated House Price:
$452,000
```

---

## 🎯 Future Improvements

* Feature Importance Visualization
* Interactive California Housing Map
* SHAP Explainability
* Batch CSV Predictions
* Hyperparameter Optimization
* Model Monitoring Dashboard

---

## 👨‍💻 Author

**Swikar Bhattarai**

- GitHub: [@swikarb69](https://github.com/swikarb69)
- LinkedIn: [Swikar Bhattarai](https://linkedin.com/in/swikar-bhattarai-11178b240)

---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub.
