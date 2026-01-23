import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load data
# -----------------------------
def load_data(path):
    return pd.read_csv(path)

# -----------------------------
# Label churn
# -----------------------------
def label_churn(df, churn_threshold_days=90):
    df = df.copy()
    df["churn"] = np.where(df["recency"] > churn_threshold_days, 1, 0)
    return df

# -----------------------------
# Train churn model (LR)
# -----------------------------
def train_churn_model(df):
    features = [
        "recency",
        "frequency",
        "monetary",
        "avg_order_value",
        "tenure_days"
    ]

    X = df[features]
    y = df["churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    churn_prob = model.predict_proba(X_scaled)[:, 1]

    df["churn_probability"] = churn_prob
    return df

# -----------------------------
# Calculate CLV
# -----------------------------
def calculate_clv(df):
    df = df.copy()

    # Purchase frequency per day
    df["purchase_frequency"] = df["frequency"] / df["tenure_days"]

    # Expected lifetime (inverse of churn probability)
    df["expected_lifetime"] = 1 / (df["churn_probability"] + 1e-6)

    # CLV formula
    df["CLV"] = (
        df["avg_order_value"]
        * df["purchase_frequency"]
        * df["expected_lifetime"]
    )

    return df

# -----------------------------
# Main pipeline
# -----------------------------
if __name__ == "__main__":
    input_path = r"C:\Users\OMEN\customer-lifetime-and-churn-analysis\Model\data\processed\customer_features.csv"
    output_path = r"C:\Users\OMEN\customer-lifetime-and-churn-analysis\Model\customer_clv.csv"

    df = load_data(input_path)
    df = label_churn(df)
    df = train_churn_model(df)
    df = calculate_clv(df)

    df.to_csv(output_path, index=False)

    print("CLV calculation complete.")
    print(df[["customer_id", "CLV"]].sort_values("CLV", ascending=False).head(10))
