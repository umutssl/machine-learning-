📘 Telco Customer Churn Prediction



🧠 Project Overview

This project aims to predict customer churn (whether a customer leaves the company) using the Telco Customer Churn dataset.
A complete end-to-end machine learning pipeline is implemented — including data preprocessing, feature engineering, model training, hyperparameter tuning, and feature importance visualization.

📊 Dataset

Dataset name: Telco-Customer-Churn.csv

Each record represents a telecom customer and includes information about:

Demographics (gender, senior citizen, tenure)

Services subscribed (internet, phone, streaming, etc.)

Contract type and payment method

Monthly and total charges

Churn (target variable: whether the customer left or stayed)

Target Variable:
Churn → 1 if customer left, 0 otherwise.

⚙️ Steps and Workflow
1. Data Preprocessing

Removed unnecessary columns (customerID).

Converted TotalCharges to numeric and filled missing values with the median.

Encoded the Churn column into binary format (Yes → 1, No → 0).

Outlier detection and capping were applied to numerical variables.

2. Exploratory Data Analysis

Custom functions (check_df, grab_col_names, cat_summary, num_summary) were used to:

Inspect data structure

Summarize categorical and numerical variables

Detect cardinal features

3. Feature Engineering

New informative features were created to enhance model performance:

NEW_TENURE_YEAR → Categorized tenure into year groups

NEW_Engaged → Identified customers with long-term contracts

NEW_noProt → Flagged users without support or protection services

NEW_Young_Not_Engaged → Detected young customers with short contracts

NEW_TotalServices → Counted number of subscribed services

NEW_FLAG_ANY_STREAMING → Flagged streaming service usage

NEW_FLAG_AutoPayment → Identified automatic payment users

NEW_AVG_Charges, NEW_Increase, NEW_AVG_Service_Fee → Created financial ratios

4. Encoding

Label Encoding was applied to binary categorical features.

One-Hot Encoding was applied to multi-class categorical variables.

5. Feature Scaling

All numerical features were standardized using StandardScaler.

6. Base Model Comparison

A variety of classifiers were trained and compared using 10-fold cross-validation:
| Model                  | Library      |
| ---------------------- | ------------ |
| Logistic Regression    | scikit-learn |
| K-Nearest Neighbors    | scikit-learn |
| Decision Tree          | scikit-learn |
| Random Forest          | scikit-learn |
| Support Vector Machine | scikit-learn |
| XGBoost                | xgboost      |
| LightGBM               | lightgbm     |
| CatBoost               | catboost     |

Each model was evaluated using:

Accuracy

ROC-AUC

Precision

Recall

F1-score
7. Hyperparameter Tuning

Best-performing models were further optimized using GridSearchCV:
| Model        | Parameters Tuned                                                 |
| ------------ | ---------------------------------------------------------------- |
| RandomForest | `max_depth`, `max_features`, `min_samples_split`, `n_estimators` |
| XGBoost      | `learning_rate`, `max_depth`, `n_estimators`, `colsample_bytree` |
| LightGBM     | `learning_rate`, `n_estimators`, `colsample_bytree`              |
| CatBoost     | `iterations`, `learning_rate`, `depth`                           |

📈 Results Summary

All models achieved strong predictive results after tuning, with ensemble methods (especially XGBoost, LightGBM, and CatBoost) outperforming others in ROC-AUC and accuracy metrics.


🧩 Technologies Used
| Library                           | Purpose                                   |
| --------------------------------- | ----------------------------------------- |
| `pandas`, `numpy`                 | Data manipulation                         |
| `matplotlib`, `seaborn`           | Visualization                             |
| `scikit-learn`                    | Preprocessing, model training, evaluation |
| `xgboost`, `lightgbm`, `catboost` | Advanced gradient boosting models         |
| `warnings`                        | Ignore unnecessary logs                   |

