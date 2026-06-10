# EV Price Prediction and Market Forecasting System Using Machine Learning, Facebook Prophet, and Nixtla

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Forecasting](https://img.shields.io/badge/Forecasting-Nixtla-green)
![Prophet](https://img.shields.io/badge/Facebook-Prophet-purple)
![License](https://img.shields.io/badge/License-Academic-red)

---

## Project Overview

The **EV Price Prediction and Market Forecasting System** is a hybrid machine learning and time-series forecasting project developed to estimate electric vehicle (EV) prices and forecast future EV market trends in Cambodia.

The project combines:

- Machine Learning Regression Models
- Facebook Prophet Forecasting
- Nixtla Forecasting Framework
- Real-World EV Market Data
- Synthetic EV Data Generation

Unlike traditional vehicle valuation systems, this project considers technical vehicle specifications, market behavior, and historical pricing trends to provide accurate EV price estimation and future market forecasting.

---

## Objectives

- Predict EV market prices using machine learning techniques.
- Identify the most influential factors affecting EV prices.
- Forecast future EV prices using advanced time-series methods.
- Compare machine learning and forecasting approaches.
- Analyze EV depreciation trends between 2026 and 2030.
- Support data-driven decision-making for EV buyers, sellers, and businesses.

---

## Problem Statement

Traditional vehicle valuation methods primarily focus on:

- Vehicle Age
- Mileage

However, EV pricing is affected by many additional factors:

- Battery Capacity
- Driving Range
- Power Output
- Charging Technology
- Battery Aging
- Brand Reputation
- Market Demand

These factors create complex pricing relationships that require advanced machine learning and forecasting techniques.

This project aims to build a comprehensive framework capable of:

1. Predicting current EV prices.
2. Forecasting future EV market trends.

---

## Dataset

### Data Sources

| Dataset | Records |
| ------- | ------- |
| Khmer24 EV Listings (Real Data) | 1,187 |
| Synthetic EV Dataset | 5,000 |
| **Total** | **6,187** |

The Khmer24 dataset reflects real Cambodian EV market conditions, while the synthetic dataset improves training coverage and model robustness.

---

## Features Used

### Vehicle Features

- Brand
- Model
- Mileage (km)
- Battery Capacity (kWh)
- Driving Range (km)
- Power Output (kW)
- Acceleration (0-100 km/h)
- Charging Type
- Seller Type
- Vehicle Condition
- City
- Color

### Engineered Features

- Age
- Age_Squared
- Log_Mileage
- Mileage_per_Year
- Efficiency_km_per_kWh
- Power_to_Battery
- Range_per_Power
- Condition_Score
- Fast_Charging
- Listing_Age_Days

---

## Data Preprocessing

### Data Cleaning

- Duplicate Removal
- Invalid Price Removal
- Datetime Conversion
- Numerical Conversion

### Missing Value Handling

#### Numerical Variables

```python
Median Imputation
```

#### Categorical Variables

```python
Most Frequent Imputation
```

### Feature Transformation

```python
StandardScaler()
OneHotEncoder()
ColumnTransformer()
Pipeline()
```

### Outlier Analysis

```python
Interquartile Range (IQR)
```

---

## Feature Engineering

### Age Features

```text
Age
Age_Squared
```

### Usage Features

```text
Log_Mileage
Mileage_per_Year
```

### Efficiency Features

```text
Efficiency_km_per_kWh
Power_to_Battery
Range_per_Power
```

### Market Features

```text
Condition_Score
Fast_Charging
Listing_Age_Days
```

---

## Models Implemented

### Machine Learning Models

#### Linear Regression

Baseline predictive model used to estimate EV prices based on vehicle attributes.

#### Random Forest Regressor

Ensemble learning model used to capture complex non-linear pricing relationships.

```python
RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

### Facebook Prophet Models

#### Baseline Prophet

Forecasting model using historical EV prices.

#### Prophet with Auxiliary Data (Early Fusion)

Prophet enhanced with external regressors:

- Mileage
- Vehicle Age
- Battery Capacity
- Driving Range
- Power Output
- Condition Score
- Brand Premium

Features:

- Logistic Growth Trend
- Yearly Seasonality
- Cambodia Holiday Effects
- Changepoint Detection

### Nixtla Forecasting Framework

#### StatsForecast

##### AutoARIMA

```python
AutoARIMA(season_length=12)
```

Purpose:

- Statistical forecasting baseline
- Automatic parameter optimization
- Monthly EV price forecasting

#### MLForecast

##### Linear Regression Forecasting

```python
LinearRegression()
```

##### Random Forest Forecasting

```python
RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
```

##### Gradient Boosting Regression Trees (GBRT)

```python
GradientBoostingRegressor(
    n_estimators=499,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
```

---

## Forecast Configuration

### Lag Features

```python
[1, 2, 3, 4, 5, 6]
```

### Forecast Horizon

```text
6 Months
```

### Validation Strategy

```text
Rolling Window Cross Validation
```

---

## Evaluation Metrics

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score

---

## Results

### 1. Linear Regression

#### Test Performance

| Metric | Value |
| ------ | ----- |
| MAE | $1,295.12 |
| RMSE | $1,715.24 |
| R² | 0.9911 |

#### Cross Validation

| Metric | Value |
| ------ | ----- |
| Mean R² | 0.9892 |
| Mean RMSE | $1,830.39 |

#### Findings

- Best regression model on synthetic data.
- Strong generalization capability.
- Minimal overfitting.
- Explained over 99% of EV price variation.

### 2. Random Forest Regressor

#### Test Performance

| Metric | Value |
| ------ | ----- |
| MAE | $1,284.30 |
| RMSE | $1,775.13 |
| R² | 0.9905 |

#### Top Important Features

| Feature | Importance |
| ------- | ---------- |
| Power Output | 0.6987 |
| Acceleration | 0.0891 |
| Battery Capacity | 0.0447 |
| Efficiency | 0.0260 |

#### Findings

- Strong non-linear learning capability.
- Slightly better MAE than Linear Regression.
- Strong performance on Khmer24 real-world data.

### 3. Facebook Prophet

#### Khmer24 Dataset

| Metric | Value |
| ------ | ----- |
| MAE | $7,707.63 |
| RMSE | $8,937.15 |
| MAPE | 17.11% |
| R² | 0.7029 |

#### Synthetic Dataset

| Metric | Value |
| ------ | ----- |
| MAE | $6,614.48 |
| RMSE | $8,786.22 |
| MAPE | 15.25% |
| R² | 0.7969 |

#### Findings

- Successfully modeled market trends.
- Captured seasonality and holiday effects.
- Better performance on synthetic data.

---

## Nixtla Forecasting Results

| Method | Dataset | R² | MAE |
| ------ | ------- | -- | --- |
| Baseline Prophet | Historical + Synthetic | 0.7587 | $6,777 |
| Early Fusion (+ Auxiliary Data) | Historical + Synthetic | 0.7655 | $6,416 |
| Direct Global Fusion (Random Forest) | Combined Dataset | **0.9825** | **$2,145** |
| Late Fusion (Prophet + RF) | Combined Dataset | 0.8118 | $6,615 |
| Aggregated Prophet | Combined Dataset | 0.7328 | $6,606 |

### Best Forecasting Model

#### Direct Global Fusion (Nixtla + Random Forest)

| Metric | Value |
| ------ | ----- |
| R² | **0.9825** |
| MAE | **$2,145** |

This model achieved the highest forecasting accuracy by combining:

- Historical EV price trends
- Lag-based forecasting features
- Vehicle specifications
- Auxiliary market variables
- Random Forest forecasting

---

## Future EV Price Forecast (2026-2030)

| Vehicle | 2026 | 2027 | 2028 | 2029 | 2030 |
| ------- | ---- | ---- | ---- | ---- | ---- |
| BYD Atto 3 | $23,869 | $20,747 | $17,928 | $17,914 | $18,362 |
| MG ZS EV | $22,655 | $19,539 | $17,114 | $17,451 | $17,902 |
| Nissan Leaf | $8,442 | $7,414 | $7,414 | $7,414 | $7,414 |
| Tesla Model 3 | $37,346 | $34,308 | $31,575 | $29,262 | $29,747 |
| Tesla Model Y | $51,783 | $48,808 | $46,140 | $43,514 | $41,063 |

### Key Forecast Insights

- Tesla Model Y retains the highest market value.
- Tesla Model 3 shows strong long-term resale performance.
- BYD Atto 3 and MG ZS EV experience faster depreciation.
- Nissan Leaf remains relatively stable at a lower price range.
- EV prices generally decline due to battery aging, technological advancement, and increased market competition.

---

## Key Findings

- Power Output is the strongest predictor of EV price.
- Battery Capacity and Driving Range significantly influence resale value.
- Linear Regression achieved excellent predictive accuracy.
- Random Forest effectively captured non-linear relationships.
- Prophet successfully modeled long-term market trends.
- Nixtla provided advanced forecasting capabilities.
- Direct Global Fusion achieved the highest forecasting performance.
- Combining Machine Learning and Forecasting produced the most comprehensive EV valuation framework.

---

## Technology Stack

### Programming Language

- Python 3.10+

### Libraries

- Pandas
- NumPy
- Scikit-Learn
- Facebook Prophet
- Nixtla
- StatsForecast
- MLForecast
- Matplotlib
- Seaborn

### Development Environment

- Jupyter Notebook
- Google Colab

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Jupyter Notebook or Google Colab

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/EV-Price-Prediction.git
cd EV-Price-Prediction

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Launch Jupyter Notebook
jupyter notebook

# Open the desired notebook from the notebooks/ directory:
# - data_preprocessing.ipynb
# - linear_regression.ipynb
# - random_forest.ipynb
# - prophet_forecasting.ipynb
# - nixtla_forecasting.ipynb
```

---

## Project Structure

```text
EV-Price-Prediction/
│
├── data/
│   ├── khmer24_ev_data.csv
│   ├── synthetic_ev_data.csv
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── linear_regression.ipynb
│   ├── random_forest.ipynb
│   ├── prophet_forecasting.ipynb
│   ├── nixtla_forecasting.ipynb
│
├── models/
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│
├── results/
│   ├── evaluation_metrics/
│   ├── visualizations/
│   ├── forecasts/
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Authors

- Cheang Yornphavorak
- Kruy Monychotakna
- Bun David
- Uon Kimmeng
- Khut Buntha

### Institution

Royal University of Phnom Penh (RUPP)

Bachelor of Engineering in Data Science and Engineering

---

## License

This project was developed for academic and research purposes. It may be used for educational and non-commercial research with proper attribution.

---

## Conclusion

This project presents a comprehensive EV valuation framework that integrates machine learning prediction models, Facebook Prophet forecasting, and the Nixtla forecasting ecosystem. The results demonstrate that combining vehicle-specific features with advanced time-series forecasting techniques can significantly improve EV price estimation and market trend prediction. Among all forecasting approaches, the Direct Global Fusion model achieved the strongest forecasting performance, highlighting the effectiveness of combining machine learning with temporal forecasting methods for real-world EV market analysis.
