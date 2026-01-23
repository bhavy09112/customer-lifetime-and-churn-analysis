import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Redirect stdout to file
output_file = open(
    "results/random_forest_results.txt",
    "w"
)
sys.stdout = output_file

# Load data
df = pd.read_csv(r"C:\Users\OMEN\customer-lifetime-and-churn-analysis\Model\data\processed\customer_features.csv")

# Create churn label
CHURN_THRESHOLD_DAYS = 90
df["churn"] = np.where(df["recency"] > CHURN_THRESHOLD_DAYS, 1, 0)

# Feature selection
features = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "tenure_days"
]

X = df[features]
y = df["churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("===== Random Forest Results =====\n")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# Restore stdout
sys.stdout = sys.__stdout__
output_file.close()

print("Random Forest results saved successfully.")
