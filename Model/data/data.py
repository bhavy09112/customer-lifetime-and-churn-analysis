import pandas as pd
import numpy as np

# Loading raw data
def load_data(path):
    return pd.read_excel(path)

# Clean transaction data
def clean_transactions(df):
    df = df.copy()

    # 1. Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Expected columns after rename:
    # invoice, stockcode, description, quantity,
    # invoicedate, price, customer_id, country

    # 2. Remove rows without Customer ID
    df = df.dropna(subset=["customer_id"])

    # 3. Convert Customer ID to integer
    df["customer_id"] = df["customer_id"].astype(int)

    # 4. Convert InvoiceDate to datetime
    df["invoicedate"] = pd.to_datetime(df["invoicedate"])

    # 5. Remove cancelled transactions
    # Cancelled invoices starts with 'C'
    df = df[~df["invoice"].astype(str).str.startswith("C")]

    # 6. Remove invalid quantities and prices
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]

    # 7. Handle description text
    df["description"] = (
        df["description"]
        .str.strip()
        .str.upper()
    )

    # 8. Remove duplicates
    df = df.drop_duplicates()

    # 9. Create total transaction value
    df["total_amount"] = df["quantity"] * df["price"]

    # 10. Reset index
    df = df.reset_index(drop=True)

    return df

# Save cleaned data
def save_clean_data(df, path):
    df.to_csv(path, index=False)

# Main pipeline
if __name__ == "__main__":
    raw_path = r"C:\Users\OMEN\customer-lifetime-and-churn-analysis\Model\data\raw\online_retail.xlsx"
    clean_path = r"C:\Users\OMEN\customer-lifetime-and-churn-analysis\Model\data\processed\transactions_clean.csv"

    df_raw = load_data(raw_path)
    df_clean = clean_transactions(df_raw)
    save_clean_data(df_clean, clean_path)

    print("Data cleaning complete.")
    print(f"Cleaned dataset shape: {df_clean.shape}")
