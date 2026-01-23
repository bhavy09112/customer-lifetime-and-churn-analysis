import pandas as pd

# Load customer features
def load_features(path):
    return pd.read_csv(path)

# Create RFM scores
def compute_rfm_scores(df, n_bins=5):
    df = df.copy()

    # Rank values to avoid duplicate bin edges
    r_rank = df["recency"].rank(method="first")
    f_rank = df["frequency"].rank(method="first")
    m_rank = df["monetary"].rank(method="first")

    # Recency: lower is better
    df["R_score"] = pd.qcut(
        r_rank,
        q=n_bins,
        labels=range(n_bins, 0, -1)
    ).astype(int)

    # Frequency: higher is better
    df["F_score"] = pd.qcut(
        f_rank,
        q=n_bins,
        labels=range(1, n_bins + 1)
    ).astype(int)

    # Monetary: higher is better
    df["M_score"] = pd.qcut(
        m_rank,
        q=n_bins,
        labels=range(1, n_bins + 1)
    ).astype(int)

    df["RFM_score"] = (
        df["R_score"].astype(str)
        + df["F_score"].astype(str)
        + df["M_score"].astype(str)
    )

    df["RFM_sum"] = df[["R_score", "F_score", "M_score"]].sum(axis=1)

    return df

# Assign customer segments
def assign_rfm_segment(df):
    df = df.copy()

    def segment(row):
        if row["RFM_sum"] >= 13:
            return "Champions"
        elif row["RFM_sum"] >= 10:
            return "Loyal Customers"
        elif row["R_score"] >= 4 and row["F_score"] <= 2:
            return "New Customers"
        elif row["R_score"] <= 2 and row["F_score"] >= 4:
            return "At Risk"
        elif row["RFM_sum"] <= 6:
            return "Lost Customers"
        else:
            return "Potential Loyalists"

    df["RFM_segment"] = df.apply(segment, axis=1)
    return df

# Save RFM data
def save_rfm(df, path):
    df.to_csv(path, index=False)

# Main pipeline
if __name__ == "__main__":
    input_path = r"C:\Users\OMEN\customer-lifetime-and-churn-analysis\Model\data\processed\customer_features.csv"
    output_path = r"C:\Users\OMEN\customer-lifetime-and-churn-analysis\Model\data\customer_rfm.csv"

    df_features = load_features(input_path)
    df_rfm = compute_rfm_scores(df_features)
    df_rfm = assign_rfm_segment(df_rfm)
    save_rfm(df_rfm, output_path)

    print("RFM segmentation complete.")
    print(df_rfm["RFM_segment"].value_counts())
