# Insurance Claims Fraud Risk Analytics

## Project Overview

This project analyzes auto insurance claims data to identify potentially fraudulent claims using machine learning and business-oriented risk reporting.

The goal of this project is not only to build classification models, but also to translate model outputs into practical insights that can support insurance claim review, fraud-risk detection, and investigation prioritization.

The workflow includes data cleaning, feature engineering, categorical encoding, class imbalance handling, model training, model evaluation, and feature importance analysis.

---

## Dataset

The dataset used in this project is an auto insurance claims fraud detection dataset from Kaggle.

It contains policy, customer, vehicle, and incident-level information related to insurance claims. The target variable is `fraud_reported`, which indicates whether a claim was reported as fraudulent.

### Target Variable

| Variable | Description |
|---|---|
| `fraud_reported` | Indicates whether an insurance claim was reported as fraudulent |

Target encoding:

| Original Value | Encoded Value | Meaning |
|---|---:|---|
| `Y` | 1 | Fraudulent claim |
| `N` | 0 | Non-fraudulent claim |

---

## Tools and Libraries

This project was developed using:

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- matplotlib
- seaborn

---

## Project Workflow

## 1. Data Loading

The notebook automatically searches the Kaggle input directory for CSV files and loads the first available dataset.

```python
csv_files = glob.glob('/kaggle/input/**/*.csv', recursive=True)
file_path = csv_files[0]
df = pd.read_csv(file_path)
```

This makes the notebook easier to run in a Kaggle environment without manually entering a dataset path.

---

## 2. Data Cleaning

Missing values represented by `?` are replaced with `NaN`.

```python
df.replace('?', np.nan, inplace=True)
```

Missing categorical values are filled using the mode for the following columns:

- `collision_type`
- `property_damage`
- `police_report_available`

These fields are related to claim and incident characteristics, so filling them helps preserve useful business information instead of removing rows.

---

## 3. Feature Engineering

Two business-relevant features are created from the original dataset.

### Policy Age

`policy_age_days` measures the number of days between the policy binding date and the incident date.

```python
df['policy_age_days'] = (df['incident_date'] - df['policy_bind_date']).dt.days
```

This feature may help capture whether claims occur soon after a policy begins, which can be useful in fraud-risk analysis.

### Vehicle Age

`vehicle_age` measures the age of the vehicle at the time of the incident.

```python
df['vehicle_age'] = df['incident_year'] - df['auto_year']
```

This feature may help analyze whether vehicle age contributes to claim patterns or fraud-risk behavior.

After these features are created, redundant identifiers and raw date columns are removed.

---

## 4. Encoding and Train-Test Split

Categorical variables are transformed using one-hot encoding.

```python
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
```

One-hot encoding avoids assigning artificial ordinal meaning to categorical variables.

The dataset is then split into training and testing sets using a stratified split.

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

Stratification helps preserve the fraud and non-fraud class distribution in both the training and testing sets.

---

## 5. Feature Scaling

Numerical features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
```

The scaler is fitted only on the training set and then applied to the test set to avoid data leakage.

---

## 6. Model Training

Two classification models are trained and compared:

### Random Forest Classifier

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    class_weight='balanced'
)
```

The Random Forest model uses `class_weight='balanced'` to help address class imbalance.

### XGBoost Classifier

```python
XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=pos_weight
)
```

The XGBoost model uses `scale_pos_weight` to account for the imbalance between fraudulent and non-fraudulent claims.

---

## 7. Model Evaluation

The models are evaluated using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix

Since insurance fraud detection is usually an imbalanced classification problem, accuracy alone is not enough. Recall, precision, F1-score, ROC-AUC, and PR-AUC provide a more complete view of model performance.

### Evaluation Focus

In a fraud-risk setting:

- **Recall** is important because missing fraudulent claims can be costly.
- **Precision** is important because too many false alarms can increase investigation workload.
- **PR-AUC** is especially useful for imbalanced datasets.
- **Confusion matrices** help show how many fraud and non-fraud cases are correctly or incorrectly classified.

---

## 8. Feature Importance Analysis

The XGBoost model is used to generate feature importance scores.

```python
importances = xgb.feature_importances_
```

The top 15 features are visualized to help explain which variables contribute most to fraud-risk prediction.

This step supports model interpretability and helps translate technical results into business insights.

---

## Key Business Value

This project demonstrates how machine learning can support insurance fraud-risk analytics by:

- Identifying potentially suspicious claims
- Prioritizing claims for manual investigation
- Reducing unnecessary review workload
- Supporting data-driven claim review decisions
- Explaining key factors associated with fraud-risk prediction
- Translating model results into business-oriented insights

---

## Results and Interpretation

The project compares Random Forest and XGBoost models using multiple performance metrics. The confusion matrix and classification report provide insight into how well each model identifies fraudulent and non-fraudulent claims.

The feature importance chart helps identify the strongest predictors in the model, which can be used to support fraud-risk reporting and claim investigation prioritization.

---

## Project Limitations

This project is based on a public dataset and is intended for learning, portfolio, and analytical demonstration purposes.

In a real insurance business environment, additional validation would be required before using the model for operational decision-making.

Possible limitations include:

- The dataset may not fully represent real-world insurance claim behavior.
- Public datasets may contain simplified or historical patterns.
- Feature importance does not always imply direct causation.
- Model thresholds may need to be adjusted based on business costs.
- Further explainability techniques would be needed for production use.

---

## Future Improvements

Possible future improvements include:

- Hyperparameter tuning using cross-validation
- Threshold optimization based on investigation cost
- SHAP analysis for stronger model explainability
- Additional business feature engineering
- Model monitoring for data drift
- Power BI or Tableau dashboard for fraud-risk reporting
- Deployment as a simple web application or API

---

## Skills Demonstrated

This project demonstrates the following skills:

- Data cleaning
- Missing-value handling
- Feature engineering
- One-hot encoding
- Stratified train-test splitting
- Feature scaling
- Classification modeling
- Handling imbalanced data
- Random Forest modeling
- XGBoost modeling
- Model evaluation
- ROC-AUC and PR-AUC interpretation
- Confusion matrix analysis
- Feature importance visualization
- Business-oriented analytics reporting

---

## Repository Structure

```text
insurance-claims-fraud-risk-analytics/
│
├── README.md
├── insurance_fraud_analysis.ipynb
├── requirements.txt
└── images/
    ├── random_forest_confusion_matrix.png
    ├── xgboost_confusion_matrix.png
    └── xgboost_feature_importance.png
```

---

## How to Run

1. Clone the repository.

```bash
git clone https://github.com/your-username/insurance-claims-fraud-risk-analytics.git
```

2. Install the required libraries.

```bash
pip install -r requirements.txt
```

3. Open the notebook.

```bash
jupyter notebook insurance_fraud_analysis.ipynb
```

4. Run all cells in order.

---

## Example Project Summary

This project builds an end-to-end machine learning workflow for auto insurance claims fraud-risk analysis. It prepares mixed-type insurance claim data, engineers business-relevant features, handles class imbalance, trains Random Forest and XGBoost classifiers, evaluates performance using fraud-focused metrics, and visualizes feature importance to support claim investigation prioritization.

---

## Author

Jason Chan
