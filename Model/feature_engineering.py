import pandas as pd
import numpy as np

# Load cleaned transaction data
def load_data(path):
    return pd.read_csv(path, parse_dates=["invoicedate"])

# Create customer-level features
def engineer_customer_features(df):
    df = df.copy()

    # Reference date = last date in dataset + 1 day calculating recency
    snapshot_date = df["invoicedate"].max() + pd.Timedelta(days=1)

    customer_features = (
        df.groupby("customer_id")
        .agg(
            recency=("invoicedate", lambda x: (snapshot_date - x.max()).days),
            frequency=("invoice", "nunique"),
            monetary=("total_amount", "sum"),
            avg_order_value=("total_amount", "mean"),
            first_purchase=("invoicedate", "min"),
            last_purchase=("invoicedate", "max")
        )
        .reset_index()
    )

    # Customer tenure (days between first and last purchase)
    customer_features["tenure_days"] = (
        customer_features["last_purchase"]
        - customer_features["first_purchase"]
    ).dt.days

    # Purchase span safety (avoid zero-division logic later)
    customer_features["tenure_days"] = customer_features["tenure_days"].clip(lower=1)

    # Drop raw date columns (not model-friendly)
    customer_features = customer_features.drop(
        columns=["first_purchase", "last_purchase"]
    )

    return customer_features

# Save engineered features
def save_features(df, path):
    df.to_csv(path, index=False)

# Main pipeline
if __name__ == "__main__":
    input_path = r"C:\Users\OMEN\customer-lifetime-and-churn-analysis\Model\data\processed\transactions_clean.csv"
    output_path = r"C:\Users\OMEN\customer-lifetime-and-churn-analysis\Model\data\processed\customer_features.csv"

    df_transactions = load_data(input_path)
    df_features = engineer_customer_features(df_transactions)
    save_features(df_features, output_path)

    print("Feature engineering complete.")
    print(f"Customer feature table shape: {df_features.shape}")

# we are saving file in data\processed