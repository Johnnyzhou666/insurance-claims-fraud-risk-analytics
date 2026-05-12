import os
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')

# 1. Load and Clean Data
print("Locating dataset...")
csv_files = glob.glob('/kaggle/input/**/*.csv', recursive=True)

if not csv_files:
    raise FileNotFoundError("No CSV file found. Please check your Kaggle inputs.")

file_path = csv_files[0]
print(f"Loading data from: {file_path}")
df = pd.read_csv(file_path)

# Handle missing values denoted by '?'
df.replace('?', np.nan, inplace=True)

# Impute missing categoricals with mode
cols_to_fill = ['collision_type', 'property_damage', 'police_report_available']
for col in cols_to_fill:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)


# 2. Feature Engineering
print("Engineering features...")

# Calculate days between policy binding and incident
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
df['incident_date'] = pd.to_datetime(df['incident_date'])
df['policy_age_days'] = (df['incident_date'] - df['policy_bind_date']).dt.days

# Calculate vehicle age at the time of the incident
df['incident_year'] = df['incident_date'].dt.year
df['vehicle_age'] = df['incident_year'] - df['auto_year']

# Drop redundant or ID columns
cols_to_drop = ['policy_number', 'policy_bind_date', 'incident_date', 
                'incident_location', 'auto_year', 'incident_year', '_c39']
df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

# Map target to binary
df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})


# 3. Encoding and Splitting
# One-hot encode categoricals to avoid arbitrary ordinal weights
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']

# Stratified split to maintain fraud ratio in train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric features
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


# 4. Modeling & Evaluation
def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] 
    
    print(f"\n--- {name} Results ---")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probs):.4f}")
    print(f"PR-AUC:    {average_precision_score(y_test, probs):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, preds))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('Actual (0: Normal, 1: Fraud)')
    plt.xlabel('Predicted')
    plt.show()

# Random Forest using balanced class weights
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced')
rf.fit(X_train, y_train)
evaluate(rf, X_test, y_test, "Random Forest")

# XGBoost handling imbalance natively
print("Training XGBoost...")
pos_weight = sum(y_train == 0) / sum(y_train == 1)
xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42, 
                    eval_metric='logloss', scale_pos_weight=pos_weight)
xgb.fit(X_train, y_train)
evaluate(xgb, X_test, y_test, "XGBoost")


# 5. Feature Importance
print("Plotting feature importances...")
importances = xgb.feature_importances_
idx = np.argsort(importances)[::-1] 

top_n = 15
top_features = X.columns[idx][:top_n]
top_scores = importances[idx][:top_n]

plt.figure(figsize=(10, 6))
sns.barplot(x=top_scores, y=top_features, palette="viridis")
plt.title('Top 15 Feature Importances (XGBoost)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
