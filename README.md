# Insurance Claims Fraud Risk Analytics

## Project Overview

This project builds a machine learning workflow to identify potentially fraudulent auto insurance claims using policy, vehicle, incident, and claim-related information.

The goal of this project is not only to train classification models, but also to support business-oriented fraud risk analysis. The model results can help prioritize suspicious claims for further manual review and support data-driven decision-making in insurance claim investigation.

## Dataset

The dataset used in this project is an auto insurance claims fraud detection dataset from Kaggle.

The target variable is:

- `fraud_reported`
  - `Y`: Fraudulent claim
  - `N`: Non-fraudulent claim

The dataset includes information such as:

- Policy details
- Customer information
- Incident details
- Vehicle information
- Claim amounts
- Fraud reporting status

## Tools and Technologies

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- matplotlib
- seaborn
- Kaggle Notebook environment

## Project Workflow

### 1. Data Loading

The notebook automatically searches the Kaggle input directory and loads the first available CSV file.

```python
csv_files = glob.glob('/kaggle/input/**/*.csv', recursive=True)
file_path = csv_files[0]
df = pd.read_csv(file_path)
```

This makes the notebook easier to run in the Kaggle environment without manually typing the dataset path.

### 2. Data Cleaning

Missing values represented by `?` are replaced with `NaN`.

```python
df.replace('?', np.nan, inplace=True)
```

The following categorical columns are filled using their most frequent values:

- `collision_type`
- `property_damage`
- `police_report_available`

This helps preserve useful claim-related information instead of removing rows with missing values.

### 3. Feature Engineering

Two additional business-relevant features are created.

#### Policy Age

```python
policy_age_days = incident_date - policy_bind_date
```

This feature represents the number of days between the policy binding date and the incident date. It may help capture suspicious claims that occur shortly after policy activation.

#### Vehicle Age

```python
vehicle_age = incident_year - auto_year
```

This feature represents the age of the vehicle at the time of the incident.

After creating these features, redundant identifiers and raw date columns are removed, including:

- `policy_number`
- `policy_bind_date`
- `incident_date`
- `incident_location`
- `auto_year`
- `incident_year`
- `_c39`

### 4. Target Encoding

The target variable `fraud_reported` is converted into binary format:

```python
Y -> 1
N -> 0
```

Where:

- `1` represents a fraudulent claim
- `0` represents a normal claim

### 5. Categorical Encoding

Categorical variables are converted using one-hot encoding.

```python
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
```

One-hot encoding is used instead of label encoding to avoid assigning artificial ordinal relationships to categorical values.

### 6. Train-Test Split

The dataset is split into training and testing sets using an 80/20 split.

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

A stratified split is used to maintain a similar fraud/non-fraud ratio in both the training and testing sets.

### 7. Feature Scaling

Numerical features are standardized using Z-score normalization.

```python
StandardScaler()
```

This ensures that numerical features are on a comparable scale before model training.

### 8. Model Training

Two classification models are trained and compared:

#### Random Forest Classifier

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    class_weight='balanced'
)
```

The `class_weight='balanced'` parameter is used to help address class imbalance.

#### XGBoost Classifier

```python
XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=pos_weight
)
```

The `scale_pos_weight` parameter is used to help XGBoost handle the imbalanced fraud classification problem.

## Model Evaluation

The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

Since fraud detection is usually an imbalanced classification problem, accuracy alone is not enough. This project also considers recall, F1-score, ROC-AUC, and PR-AUC to better evaluate model performance on fraudulent claims.

## Feature Importance

The XGBoost model is used to generate feature importance scores.

The top 15 most important features are visualized to understand which claim characteristics contribute most to the fraud prediction model.

```python
importances = xgb.feature_importances_
```

This helps translate model outputs into more interpretable business insights.

## Business Value

This project demonstrates how machine learning can support insurance fraud risk analysis by:

- Identifying potentially suspicious claims
- Prioritizing claims for manual investigation
- Supporting fraud review teams with data-driven insights
- Reducing unnecessary review workload
- Explaining important risk drivers through feature importance analysis

## How to Run

### Option 1: Run on Kaggle

1. Open Kaggle Notebook.
2. Add the auto insurance fraud dataset as an input.
3. Copy the notebook code into Kaggle.
4. Run all cells in order.

The notebook automatically searches for CSV files inside:

```python
/kaggle/input/
```

### Option 2: Run Locally

If running locally, update the file path manually:

```python
file_path = "your_local_dataset_path.csv"
df = pd.read_csv(file_path)
```

Then install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

Run the Python script or Jupyter Notebook.

## Project Structure

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

## Requirements

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

## Limitations and Future Improvements

This project is designed as a machine learning analytics project using a public dataset. In a real insurance business environment, additional validation, governance, and monitoring would be required before using the model in production.

Possible future improvements include:

- Hyperparameter tuning using cross-validation
- Threshold adjustment based on fraud investigation cost
- SHAP analysis for stronger model explainability
- More advanced feature engineering
- Model monitoring for data drift
- Power BI or Tableau dashboard for business reporting
- Comparison with logistic regression as a more interpretable baseline model

## Skills Demonstrated

- Data cleaning
- Missing value handling
- Feature engineering
- One-hot encoding
- Train-test splitting
- Feature scaling
- Imbalanced classification handling
- Random Forest modeling
- XGBoost modeling
- Model evaluation
- Confusion matrix visualization
- Feature importance analysis
- Business-oriented fraud risk interpretation

## Summary

This project presents an end-to-end machine learning workflow for auto insurance claims fraud detection. It combines data preprocessing, feature engineering, classification modeling, performance evaluation, and feature importance analysis to support fraud risk review and insurance claim investigation prioritization.
