# EV Price Prediction and Market Forecasting System

## Overview

A machine learning and time-series forecasting system for estimating electric vehicle (EV) resale prices and analyzing future market trends in Cambodia.

Combines **regression-based ML models** (Linear Regression, Random Forest, XGBoost) with **time-series forecasting** (Prophet) to provide accurate price estimation and future market insights, leveraging both real-world Khmer24 listings and synthetic EV data.

---

## Objectives

- Predict EV prices using machine learning
- Identify key factors influencing EV market value
- Forecast future price trends using Prophet + macroeconomic covariates
- Compare traditional regression models with forecasting approaches
- Provide data-driven insights for EV buyers, sellers, and businesses

---

## Problem Statement

Traditional vehicle valuation relies on age and mileage, but EVs introduce additional pricing factors: battery capacity and degradation, driving range, charging technology, power output, brand reputation, and government incentives. Conventional depreciation models are insufficient — this project develops a predictive system capturing these complex relationships.

---

## Datasets

| Dataset | Source | Records | Usage |
|---|---|---|---|
| Khmer24 EV Listings | Real market (Cambodia) | 1,187 | Real-world validation |
| Synthetic EV Records | Generated (New Formula) | 5,000 | Model training |
| **Combined** | | **6,187** | |

---

## Methodology

### Feature Engineering

| Category | Features |
|---|---|
| Age | `Age`, `Age_Squared` |
| Usage | `Log_Mileage`, `Mileage_per_Year` |
| Efficiency | `Efficiency_km_per_kWh`, `Power_to_Battery`, `Range_per_Power` |
| Market | `Condition_Score`, `Fast_Charging`, `Listing_Age_Days` |

### Preprocessing Pipeline

```
Median Imputation (numeric) -> Most Frequent Imputation (categorical)
-> StandardScaler -> OneHotEncoder -> ColumnTransformer -> sklearn Pipeline
```

### Auxiliary Macroeconomic Covariates (Prophet)

Nine Cambodia-specific monthly features (2020–2030): inflation, fuel price, USD/KHR rate, charging stations, EV policy score, electricity price, battery material index, interest rate, EV import tax rate.

---

## Models

### Regression Models (sklearn)

| Model | Configuration |
|---|---|
| **Linear Regression** | Baseline |
| **Random Forest** | 400 trees, max_depth=None, min_samples_leaf=2 |
| **XGBoost** | 500 estimators, max_depth=8, lr=0.05 |

### Forecasting Models

| Model | Approach |
|---|---|
| **Prophet** | Logistic growth, explicit changepoints, Cambodia holidays |
| **Prophet + Aux Fusion** | Baseline + 9 macroeconomic regressors (Early Fusion) |
| **Direct Global Fusion** | Random Forest on all features + auxiliary (bypasses Prophet) |
| **Late Fusion** | Prophet trend + RF residual correction |
| **Aggregated Prophet** | Daily aggregation + 48 hyperparameter combinations |

---

## Evaluation Metrics

```
MAE, MSE, RMSE, MAPE, R-squared, Accuracy@10%
```

---

## Results

### Synthetic Data (5,000 records) - 80/20 Split

| Model | R-squared | MAE | RMSE |
|---|---|---|---|
| **Linear Regression** | **0.9911** | $1,295 | $1,715 |
| XGBoost | 0.9908 | $1,279 | $1,750 |
| Random Forest | 0.9905 | $1,284 | $1,775 |

### Real-World Khmer24 Data (1,187 records)

| Model | R-squared | MAE | RMSE |
|---|---|---|---|
| **XGBoost** | **0.9461** | $2,359 | $4,149 |
| Random Forest | 0.9328 | $2,806 | $4,635 |
| Linear Regression | 0.9251 | $3,246 | $4,892 |

### Grouped Cross-Validation (5-fold GroupKFold)

Prevents data leakage from repeated car signatures (same model/year across train and test).

| Model | R-squared | RMSE |
|---|---|---|
| **Linear Regression** | **0.9892** | $1,830 |
| XGBoost | 0.9884 | $1,883 |
| Random Forest | 0.9843 | $2,203 |

### Prophet Models (Combined Synthetic + Historical)

| Model | R-squared | MAE | MAPE |
|---|---|---|---|
| Baseline Prophet | ~0.75 | ~$7,161 | ~16.2% |
| Direct Global Fusion (RF + Aux) | **0.94+** | -- | -- |

---

## Feature Importance (XGBoost)

Top predictors from synthetic data:

| Feature | Importance |
|---|---|
| `Model Name_BYD Sealion 6` | 21.8% |
| `Model Name_AITO M7` | 17.9% |
| `Brand_Hyundai` | 6.4% |
| `Power_Avg` | 5.8% |
| `Model Name_Hyundai Ioniq 5` | 4.5% |
| `Acceleration` | 3.2% |
| `Battery_Avg` | 3.2% |

Model-level identity dominates -- specific models encode bundled specifications (power, battery, range, brand perception) that drive price more than any individual numeric feature alone.

---

## Key Insights

- **Synthetic-to-Real Gap**: R-squared drops from ~0.99 to ~0.93--0.95 when moving to real Khmer24 data, a ~5--7% performance decline from real-world noise and missing values.
- **GroupKFold Confirms Robustness**: R-squared remains above 0.98 even under strict grouped cross-validation.
- **Model Name** is the strongest predictor -- it bundles all specifications and brand perception.
- **Power Output, Battery Capacity, and Acceleration** are the top continuous drivers.
- Adding Cambodia macroeconomic covariates improves Prophet accuracy, but sklearn models already saturate performance.

---

## Tech Stack

**Language:** Python  
**Libraries:** pandas, NumPy, scikit-learn, XGBoost, Prophet, matplotlib, seaborn  
**Environment:** Jupyter Notebook

---

## Project Structure

```
EV_prediction/
├── Take_work/
│   ├── EV_price+AUX.ipynb                         # Main ML pipeline (LR, RF, XGBoost)
│   ├── Prophet_with_Cambodia_Auxiliary_Data.ipynb  # Prophet + macro fusion
│   └── *.csv                                       # Dataset files
├── README.md
└── requirements.txt
```

---

## Authors

**Cheang Yornphavorak, Kruy Monychotakna, Bun David, Uon Kimmeng, Khut Buntha**

Royal University of Phnom Penh -- Bachelor of Engineering in Data Science and Engineering

---

## License

Academic and research purposes. May be used for educational and non-commercial research with proper attribution.
